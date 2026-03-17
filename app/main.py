from pathlib import Path
import os
import threading
import json
import hashlib
import html
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.services.pipeline_runner import (
    create_job_from_upload,
    write_job_metadata,
    write_job_status,
    read_job_metadata,
    read_job_status,
    read_summary,
    read_crossing_events,
    run_pipeline_for_job,
)
from app.services.genai_chat import ask_gemini_about_job
from app.services.video_scene_analyzer import analyze_video_scene

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
JOBS_DIR = STORAGE_DIR / "jobs"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)


class JobChatRequest(BaseModel):
    message: str


app = FastAPI(title="Vehicle Volume Analyzer Dashboard")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "application/octet-stream",
}
ALLOWED_DIRECTIONS = {"top_to_bottom"}
MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
MAX_CHAT_MESSAGE_LENGTH = 2000


def build_error_page(title: str, heading: str, subtext: str, status_code: int) -> HTMLResponse:
    safe_title = html.escape(title)
    safe_heading = html.escape(heading)
    safe_subtext = html.escape(subtext)

    return HTMLResponse(
        content=f"""
        <html>
          <head>
            <title>{safe_title}</title>
            <link rel="stylesheet" href="/static/styles.css" />
          </head>
          <body>
            <main class="container">
              <section class="card">
                <h1>{safe_heading}</h1>
                <p class="subtext">{safe_subtext}</p>
                <p><a href="/">Go back</a></p>
              </section>
            </main>
          </body>
        </html>
        """,
        status_code=status_code,
    )


def validate_video_upload(video: UploadFile) -> tuple[bool, str]:
    filename = (video.filename or "").strip()
    suffix = Path(filename).suffix.lower()
    content_type = (video.content_type or "").strip().lower()

    if not filename:
        return False, "No video file was provided."

    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        return False, "Unsupported video format. Please upload mp4, mov, avi, or mkv."

    if content_type and content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
        return False, "Unsupported upload content type for video file."

    try:
        video.file.seek(0, os.SEEK_END)
        size_bytes = video.file.tell()
        video.file.seek(0)
    except Exception:
        return False, "Unable to inspect uploaded file."

    if size_bytes <= 0:
        return False, "Uploaded video appears to be empty."

    if size_bytes > MAX_UPLOAD_SIZE_BYTES:
        return False, "Uploaded video exceeds the 500 MB limit."

    return True, ""


def validate_runtime_params(line_frac: float, direction: str) -> tuple[bool, str]:
    try:
        line_frac_value = float(line_frac)
    except (TypeError, ValueError):
        return False, "Invalid line position value."

    if not (0.05 <= line_frac_value <= 0.95):
        return False, "line_frac must be between 0.05 and 0.95."

    direction_value = (direction or "").strip()
    if direction_value not in ALLOWED_DIRECTIONS:
        return False, "direction must be: top_to_bottom."

    return True, ""


def sanitize_job_info_for_llm(job_info: dict) -> dict:
    if not isinstance(job_info, dict):
        return {}

    allowed_keys = {
        "job_id",
        "safe_name",
        "line_frac",
        "direction",
        "status",
        "created_at",
        "updated_at",
    }

    sanitized = {}
    for key, value in job_info.items():
        if key in allowed_keys:
            sanitized[key] = value

    return sanitized


def sanitize_chat_result_for_client(result: dict) -> dict:
    if not isinstance(result, dict):
        return {
            "ok": False,
            "error": "Invalid chat response.",
            "answer": "",
        }

    sanitized = dict(result)
    sanitized.pop("debug_error", None)

    if not sanitized.get("ok"):
        error_text = str(sanitized.get("error", "") or "").strip()
        if not error_text:
            sanitized["error"] = "Unable to answer the question right now."

    return sanitized


def read_job_summary(job_id: str) -> dict:
    job_dir = JOBS_DIR / job_id
    summary_path = job_dir / "summary.json"

    if not summary_path.exists():
        return {}

    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_visual_inference(job_id: str) -> dict:
    job_dir = JOBS_DIR / job_id
    inference_path = job_dir / "visual_inference.json"

    if not inference_path.exists():
        return {}

    try:
        return json.loads(inference_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_chat_cache_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / "chat_cache.json"


CHAT_CACHE_VERSION = "v2"


def normalize_chat_cache_key(message: str) -> str:
    normalized = " ".join((message or "").strip().lower().split())
    versioned = f"{CHAT_CACHE_VERSION}:{normalized}"
    return hashlib.sha256(versioned.encode("utf-8")).hexdigest()


def read_cached_chat_answer(job_id: str, message: str) -> dict:
    cache_path = get_chat_cache_path(job_id)

    if not cache_path.exists():
        return {}

    try:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    cache_key = normalize_chat_cache_key(message)
    cached_entry = cache_data.get(cache_key)

    if not isinstance(cached_entry, dict):
        return {}

    return cached_entry


def write_cached_chat_answer(job_id: str, message: str, result: dict) -> None:
    cache_path = get_chat_cache_path(job_id)

    try:
        if cache_path.exists():
            cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cache_data, dict):
                cache_data = {}
        else:
            cache_data = {}
    except Exception:
        cache_data = {}

    cache_key = normalize_chat_cache_key(message)

    cache_data[cache_key] = {
        "ok": bool(result.get("ok", False)),
        "error": result.get("error", ""),
        "answer": result.get("answer", ""),
        "model": result.get("model", ""),
        "response_mode": result.get("response_mode", ""),
        "evidence": result.get("evidence", ""),
        "fallback_reason": result.get("fallback_reason", ""),
        "cached": True,
    }

    try:
        cache_path.write_text(
            json.dumps(cache_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[chat-cache] job_id={job_id} status=write_ok "
            f"path={cache_path.name} question={message!r}"
        )
    except Exception as exc:
        print(
            f"[chat-cache] job_id={job_id} status=write_failed "
            f"path={cache_path.name} error={exc}"
        )


def get_or_create_visual_inference(
    job_id: str,
    job_summary: dict,
    job_info: dict,
) -> dict:
    job_dir = JOBS_DIR / job_id
    inference_path = job_dir / "visual_inference.json"

    if inference_path.exists():
        try:
            return json.loads(inference_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    processed_video_path = job_dir / "annotated_output.mp4"

    result = analyze_video_scene(
        video_path=processed_video_path,
        job_summary=job_summary,
        job_info=job_info,
    )

    try:
        inference_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

    return result


def read_job_crossing_events(job_id: str) -> list[dict]:
    job_dir = JOBS_DIR / job_id
    csv_path = job_dir / "crossing_events.csv"

    if not csv_path.exists():
        return []

    events = []

    try:
        import csv

        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            for row in reader:
                frame_raw = row.get("frame", "")
                time_raw = row.get("time_sec", "")
                track_raw = row.get("track_id", "")
                class_name = str(row.get("class_name", "")).strip()

                try:
                    frame_value = int(float(frame_raw)) if str(frame_raw).strip() else 0
                except (TypeError, ValueError):
                    frame_value = 0

                try:
                    time_value = float(time_raw) if str(time_raw).strip() else 0.0
                except (TypeError, ValueError):
                    time_value = 0.0

                try:
                    track_value = int(float(track_raw)) if str(track_raw).strip() else None
                except (TypeError, ValueError):
                    track_value = None

                events.append(
                    {
                        "track_id": track_value,
                        "vehicle_class": class_name,
                        "direction": "",
                        "crossing_frame": frame_value,
                        "timestamp_seconds": time_value,
                    }
                )

    except Exception:
        return []

    return events


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_line_frac": 0.75,
            "default_direction": "top_to_bottom",
        },
    )


@app.post("/api/jobs")
async def create_job_api(
    video: UploadFile = File(...),
    line_frac: float = Form(...),
    direction: str = Form(...),
):
    is_valid_video, video_error = validate_video_upload(video)
    if not is_valid_video:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": video_error,
            },
        )

    is_valid_params, params_error = validate_runtime_params(line_frac, direction)
    if not is_valid_params:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": params_error,
            },
        )

    line_frac = float(line_frac)
    direction = direction.strip()

    job_id, job_dir, input_path, safe_name = create_job_from_upload(JOBS_DIR, video)

    write_job_metadata(
        job_dir=job_dir,
        job_id=job_id,
        safe_name=safe_name,
        input_path=input_path,
        line_frac=line_frac,
        direction=direction,
    )

    write_job_status(
        job_dir=job_dir,
        status="queued",
        message="Job accepted and queued for processing.",
        progress_pct=2,
        stage_key="queued",
        stage_label="Queued",
    )

    worker = threading.Thread(
        target=run_pipeline_for_job,
        kwargs={
            "job_dir": job_dir,
            "line_frac": line_frac,
            "direction": direction,
        },
        daemon=True,
    )
    worker.start()

    return JSONResponse(
        {
            "job_id": job_id,
            "status_url": f"/api/jobs/{job_id}/status",
            "results_url": f"/results/{job_id}",
        }
    )


@app.get("/api/jobs/{job_id}/status")
async def job_status_api(job_id: str):
    job_dir = JOBS_DIR / job_id

    if not job_dir.exists():
        return JSONResponse(
            {"error": f"Job not found: {job_id}"},
            status_code=404,
        )

    status_data = read_job_status(job_dir)
    status_data["job_id"] = job_id
    status_data["results_url"] = f"/results/{job_id}"

    return JSONResponse(status_data)


@app.post("/process")
async def process_video(
    request: Request,
    video: UploadFile = File(...),
    line_frac: float = Form(...),
    direction: str = Form(...),
):
    is_valid_video, video_error = validate_video_upload(video)
    if not is_valid_video:
        return build_error_page(
            title="Invalid Upload",
            heading="Invalid upload",
            subtext=video_error,
            status_code=400,
        )

    is_valid_params, params_error = validate_runtime_params(line_frac, direction)
    if not is_valid_params:
        return build_error_page(
            title="Invalid Processing Settings",
            heading="Invalid processing settings",
            subtext=params_error,
            status_code=400,
        )

    line_frac = float(line_frac)
    direction = direction.strip()

    job_id, job_dir, input_path, safe_name = create_job_from_upload(JOBS_DIR, video)

    write_job_metadata(
        job_dir=job_dir,
        job_id=job_id,
        safe_name=safe_name,
        input_path=input_path,
        line_frac=line_frac,
        direction=direction,
    )

    run_pipeline_for_job(
        job_dir=job_dir,
        line_frac=line_frac,
        direction=direction,
    )

    return RedirectResponse(url=f"/results/{job_id}", status_code=303)


@app.get("/results/{job_id}", response_class=HTMLResponse)
async def results_page(request: Request, job_id: str):
    job_dir = JOBS_DIR / job_id

    if not job_dir.exists():
        return build_error_page(
            title="Job Not Found",
            heading="Job not found",
            subtext=f"No job exists for ID: {job_id}",
            status_code=404,
        )

    files_in_job = sorted([p.name for p in job_dir.iterdir() if p.is_file()])
    job_info = read_job_metadata(job_dir)
    summary = read_summary(job_dir)
    events = read_crossing_events(job_dir)
    status = read_job_status(job_dir)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "job_id": job_id,
            "files_in_job": files_in_job,
            "job_info": job_info,
            "summary": summary,
            "events": events,
            "status": status,
            "input_video_url": f"/files/{job_id}/input",
            "video_url": f"/files/{job_id}/video",
            "video_download_url": f"/files/{job_id}/video/download",
            "csv_url": f"/files/{job_id}/csv",
        },
    )


@app.get("/files/{job_id}/input")
async def get_input_video(job_id: str):
    job_dir = JOBS_DIR / job_id

    video_candidates = [
        p for p in job_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        and p.name != "annotated_output.mp4"
    ]

    if not video_candidates:
        return build_error_page(
            title="Input Video Not Found",
            heading="Input video not found",
            subtext=f"No input video exists for job ID: {job_id}",
            status_code=404,
        )

    input_video_path = sorted(video_candidates)[0]

    return FileResponse(
        path=input_video_path,
        media_type="video/mp4",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/files/{job_id}/csv")
async def download_csv(job_id: str):
    job_dir = JOBS_DIR / job_id
    csv_path = job_dir / "crossing_events.csv"

    if not csv_path.exists():
        return build_error_page(
            title="CSV Not Found",
            heading="CSV not found",
            subtext=f"No CSV exists for job ID: {job_id}",
            status_code=404,
        )

    return FileResponse(
        path=csv_path,
        filename="crossing_events.csv",
        media_type="text/csv",
    )


@app.get("/api/results/{job_id}")
async def api_results(job_id: str):
    job_dir = JOBS_DIR / job_id

    if not job_dir.exists():
        return {"error": f"Job not found: {job_id}"}

    job_info = read_job_metadata(job_dir)
    summary = read_summary(job_dir)
    files_in_job = sorted([p.name for p in job_dir.iterdir() if p.is_file()])

    return {
        "job_id": job_id,
        "job_info": job_info,
        "summary": summary,
        "files_in_job": files_in_job,
    }


@app.get("/files/{job_id}/video")
async def get_annotated_video(job_id: str):
    job_dir = JOBS_DIR / job_id
    video_path = job_dir / "annotated_output.mp4"

    if not video_path.exists():
        return build_error_page(
            title="Video Not Found",
            heading="Annotated video not found",
            subtext=f"No video exists for job ID: {job_id}",
            status_code=404,
        )

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/files/{job_id}/video/download")
async def download_annotated_video(job_id: str):
    job_dir = JOBS_DIR / job_id
    video_path = job_dir / "annotated_output.mp4"

    if not video_path.exists():
        return build_error_page(
            title="Video Not Found",
            heading="Annotated video not found",
            subtext=f"No video exists for job ID: {job_id}",
            status_code=404,
        )

    return FileResponse(
        path=video_path,
        filename="annotated_output.mp4",
        media_type="video/mp4",
    )


@app.post("/api/jobs/{job_id}/chat")
async def chat_about_job(job_id: str, payload: JobChatRequest):
    job_dir = JOBS_DIR / job_id

    if not job_dir.exists():
        return JSONResponse(
            status_code=404,
            content={
                "ok": False,
                "error": f"Job '{job_id}' not found.",
                "answer": "",
            },
        )

    user_message = (payload.message or "").strip()
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": "Message cannot be empty.",
                "answer": "",
            },
        )

    if len(user_message) > MAX_CHAT_MESSAGE_LENGTH:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": "Message is too long.",
                "answer": "",
            },
        )

    summary_data = read_job_summary(job_id)
    crossing_events = read_job_crossing_events(job_id)
    job_info = read_job_metadata(job_dir)
    safe_job_info = sanitize_job_info_for_llm(job_info)

    cached_result = read_cached_chat_answer(job_id=job_id, message=user_message)
    if cached_result:
        print(
            f"[chat] job_id={job_id} source=chat_cache "
            f"mode={cached_result.get('response_mode', '')} "
            f"evidence={cached_result.get('evidence', '')} "
            f"question={user_message!r}"
        )
        return JSONResponse(
            status_code=200,
            content=sanitize_chat_result_for_client(cached_result),
        )

    visual_inference_result = get_or_create_visual_inference(
        job_id=job_id,
        job_summary=summary_data,
        job_info=safe_job_info,
    )

    result = ask_gemini_about_job(
        user_message=user_message,
        summary_data=summary_data,
        crossing_events=crossing_events,
        job_info=safe_job_info,
        metadata_context={
            "visual_inference_result": visual_inference_result,
        },
    )

    if result.get("ok") and result.get("answer"):
        print(
            f"[chat] job_id={job_id} source=fresh_response "
            f"mode={result.get('response_mode', 'gemini')} "
            f"evidence={result.get('evidence', '')} "
            f"fallback_reason={result.get('fallback_reason', '')} "
            f"question={user_message!r}"
        )

        write_cached_chat_answer(
            job_id=job_id,
            message=user_message,
            result=result,
        )

    if not result.get("ok"):
        print(
            f"[chat] job_id={job_id} source=failed_response "
            f"mode={result.get('response_mode', '')} "
            f"error_type={result.get('error_type', '')} "
            f"fallback_reason={result.get('fallback_reason', '')} "
            f"question={user_message!r}"
        )

    safe_result = sanitize_chat_result_for_client(result)
    status_code = 200 if safe_result.get("ok") else 500
    return JSONResponse(status_code=status_code, content=safe_result)