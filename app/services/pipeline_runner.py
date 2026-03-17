from pathlib import Path
import shutil
from uuid import uuid4
import json
import subprocess
import sys
import os
from datetime import datetime, timezone

def create_job_from_upload(jobs_dir: Path, uploaded_file) -> tuple[str, Path, Path, str]:
    safe_name = Path(uploaded_file.filename).name if uploaded_file.filename else "uploaded_video.mp4"
    job_id = uuid4().hex[:8]
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / safe_name

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    return job_id, job_dir, input_path, safe_name


def write_job_metadata(
    job_dir: Path,
    job_id: str,
    safe_name: str,
    input_path: Path,
    line_frac: float,
    direction: str,
) -> Path:
    metadata_path = job_dir / "job_info.txt"
    metadata_path.write_text(
        f"job_id={job_id}\n"
        f"filename={safe_name}\n"
        f"line_frac={line_frac}\n"
        f"direction={direction}\n"
        f"input_path={input_path}\n",
        encoding="utf-8",
    )
    return metadata_path


def read_job_metadata(job_dir: Path) -> dict:
    metadata_path = job_dir / "job_info.txt"

    if not metadata_path.exists():
        return {}

    data = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value

    return data


def read_job_status(job_dir: Path) -> dict:
    status_path = job_dir / "status.json"

    if not status_path.exists():
        return {
            "status": "unknown",
            "message": "No job status available yet.",
            "progress_pct": 0,
            "stage_key": "unknown",
            "stage_label": "Unknown",
        }

    return json.loads(status_path.read_text(encoding="utf-8"))


def read_summary(job_dir: Path) -> dict:
    summary_path = job_dir / "summary.json"

    if not summary_path.exists():
        return {}

    return json.loads(summary_path.read_text(encoding="utf-8"))


def read_crossing_events(job_dir: Path) -> list[dict]:
    csv_path = job_dir / "crossing_events.csv"

    if not csv_path.exists():
        return []

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []

    headers = lines[0].split(",")
    rows = []

    for line in lines[1:]:
        values = line.split(",")
        raw_row = dict(zip(headers, values))

        normalized_row = {
            "track_id": raw_row.get("track_id", ""),
            "vehicle_class": raw_row.get("class_name", raw_row.get("vehicle_class", "")),
            "direction": raw_row.get("direction", "top_to_bottom"),
            "crossing_frame": raw_row.get("frame", raw_row.get("crossing_frame", "")),
            "timestamp_seconds": raw_row.get("time_sec", raw_row.get("timestamp_seconds", "")),
        }
        rows.append(normalized_row)

    return rows


def build_summary_from_events(crossing_events: list[dict]) -> dict:
    class_order = [
        "auto",
        "bus",
        "car",
        "light_motor_vehicle",
        "motorcycle",
        "multi-axle",
        "tractor",
        "truck",
    ]

    counts_by_class = {class_name: 0 for class_name in class_order}
    normalized_events = []

    for event in crossing_events:
        vehicle_class = str(event.get("vehicle_class", "")).strip()
        timestamp_raw = event.get("timestamp_seconds", 0)

        try:
            timestamp_seconds = float(timestamp_raw)
        except (TypeError, ValueError):
            timestamp_seconds = 0.0

        if vehicle_class in counts_by_class:
            counts_by_class[vehicle_class] += 1

        normalized_events.append({
            "timestamp_seconds": max(0.0, timestamp_seconds),
            "vehicle_class": vehicle_class,
        })

    total_count = len(crossing_events)

    def get_bucket_options(max_seconds: float) -> list[int]:
        if max_seconds < 60:
            return [1, 5, 10]
        if max_seconds <= 300:
            return [10, 30, 60]
        if max_seconds <= 1800:
            return [30, 60, 300]
        if max_seconds <= 7200:
            return [60, 300, 600]
        return [60, 300, 600, 900, 1800]

    def get_default_bucket(max_seconds: float) -> int:
        if max_seconds < 60:
            return 5
        if max_seconds <= 300:
            return 30
        if max_seconds <= 1800:
            return 60
        return 300

    def build_bucket_label(bucket_start_sec: int, bucket_size_sec: int) -> str:
        bucket_end_sec = bucket_start_sec + bucket_size_sec

        if bucket_size_sec < 60:
            start_label = int(bucket_start_sec)
            end_label = int(bucket_end_sec)
            return f"{start_label}-{end_label} sec"

        start_min = int(bucket_start_sec // 60)
        end_min = int(bucket_end_sec // 60)
        return f"{start_min}-{end_min} min"

    def compute_traffic_highlight(
        events: list[dict],
        bucket_size_sec: int,
        mode: str,
    ) -> dict:
        empty_result = {
            "mode": mode,
            "bucket_size_seconds": bucket_size_sec,
            "time_range": "No data",
            "vehicle_class": "No data",
            "vehicle_count": 0,
            "bucket_start_seconds": None,
        }

        if not events:
            return empty_result

        max_seconds = max((event["timestamp_seconds"] for event in events), default=0.0)
        bucket_count = max(1, int((max_seconds // bucket_size_sec) + 1))

        bucket_totals = [0] * bucket_count
        bucket_class_counts = [{} for _ in range(bucket_count)]

        for event in events:
            timestamp_seconds = event["timestamp_seconds"]
            vehicle_class = event["vehicle_class"] or "unknown"

            bucket_index = min(int(timestamp_seconds // bucket_size_sec), bucket_count - 1)

            bucket_totals[bucket_index] += 1
            bucket_class_counts[bucket_index][vehicle_class] = (
                bucket_class_counts[bucket_index].get(vehicle_class, 0) + 1
            )

        candidate_indexes = [
            index for index, count in enumerate(bucket_totals) if count > 0
        ]

        if not candidate_indexes:
            return empty_result

        if mode == "peak":
            target_index = max(candidate_indexes, key=lambda index: bucket_totals[index])
        else:
            target_index = min(candidate_indexes, key=lambda index: bucket_totals[index])

        target_total = bucket_totals[target_index]
        target_class_counts = bucket_class_counts[target_index]

        if target_class_counts:
            dominant_class = max(
                target_class_counts.items(),
                key=lambda item: (item[1], item[0]),
            )[0]
        else:
            dominant_class = "No data"

        bucket_start_seconds = target_index * bucket_size_sec

        return {
            "mode": mode,
            "bucket_size_seconds": bucket_size_sec,
            "time_range": build_bucket_label(bucket_start_seconds, bucket_size_sec),
            "vehicle_class": dominant_class,
            "vehicle_count": target_total,
            "bucket_start_seconds": bucket_start_seconds,
        }

    max_timestamp_seconds = max(
        (event["timestamp_seconds"] for event in normalized_events),
        default=0.0,
    )

    bucket_options = get_bucket_options(max_timestamp_seconds)
    default_bucket = get_default_bucket(max_timestamp_seconds)

    traffic_highlights = {
        "bucket_options_seconds": bucket_options,
        "default_bucket_seconds": default_bucket,
        "peak": compute_traffic_highlight(normalized_events, default_bucket, "peak"),
        "trough": compute_traffic_highlight(normalized_events, default_bucket, "trough"),
    }

    return {
        "total_count": total_count,
        "counts_by_class": counts_by_class,
        "traffic_highlights": traffic_highlights,
    }

def write_job_status(
    job_dir: Path,
    status: str,
    message: str,
    progress_pct: int = 0,
    stage_key: str = "queued",
    stage_label: str = "Queued",
) -> Path:
    status_path = job_dir / "status.json"

    safe_progress = max(0, min(100, int(progress_pct)))

    status_data = {
        "status": status,
        "message": message,
        "progress_pct": safe_progress,
        "stage_key": stage_key,
        "stage_label": stage_label,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    status_path.write_text(json.dumps(status_data, indent=2), encoding="utf-8")
    return status_path


def run_pipeline_for_job(
    job_dir: Path,
    line_frac: float,
    direction: str,
) -> dict:
    base_dir = Path(__file__).resolve().parents[2]

    input_videos = [
        p for p in job_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
    ]

    if not input_videos:
        write_job_status(
            job_dir=job_dir,
            status="failed",
            message="No input video found in job folder.",
            progress_pct=0,
            stage_key="failed",
            stage_label="Input missing",
        )
        raise RuntimeError("No input video found in job folder.")

    input_video_path = input_videos[0]
    csv_path = job_dir / "crossing_events.csv"
    video_path = job_dir / "annotated_output.mp4"
    summary_path = job_dir / "summary.json"
    model_path = base_dir / "models" / "veh8_v2_best.pt"

    write_job_status(
        job_dir=job_dir,
        status="processing",
        message="Preparing pipeline execution.",
        progress_pct=5,
        stage_key="initializing",
        stage_label="Preparing job",
    )

    count_cmd = [
        sys.executable,
        "scripts/count_video.py",
        "--video", str(input_video_path),
        "--model", str(model_path),
        "--tracker_config", "configs/tracker.yaml",
        "--counter_config", "configs/counter.yaml",
        "--output_csv", str(csv_path),
        "--line_frac", str(line_frac),
        "--direction", str(direction),
    ]

    render_cmd = [
        sys.executable,
        "scripts/render_annotated_video.py",
        "--video", str(input_video_path),
        "--model", str(model_path),
        "--tracker_config", "configs/tracker.yaml",
        "--counter_config", "configs/counter.yaml",
        "--output", str(video_path),
        "--line_frac", str(line_frac),
        "--direction", str(direction),
    ]

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{base_dir}" if not existing_pythonpath else f"{base_dir}:{existing_pythonpath}"
    )

    try:
        write_job_status(
            job_dir=job_dir,
            status="processing",
            message="Running vehicle detection and counting.",
            progress_pct=28,
            stage_key="counting",
            stage_label="Counting vehicles",
        )

        subprocess.run(
            count_cmd,
            cwd=base_dir,
            env=env,
            check=True,
        )

        write_job_status(
            job_dir=job_dir,
            status="processing",
            message="Building summary from detected crossing events.",
            progress_pct=62,
            stage_key="summarizing",
            stage_label="Building analytics",
        )

        crossing_events = read_crossing_events(job_dir)
        summary_data = build_summary_from_events(crossing_events)
        summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

        write_job_status(
            job_dir=job_dir,
            status="processing",
            message="Rendering annotated output video.",
            progress_pct=82,
            stage_key="rendering",
            stage_label="Rendering video",
        )

        subprocess.run(
            render_cmd,
            cwd=base_dir,
            env=env,
            check=True,
        )

        write_job_status(
            job_dir=job_dir,
            status="completed",
            message="Pipeline completed successfully.",
            progress_pct=100,
            stage_key="completed",
            stage_label="Done",
        )

    except subprocess.CalledProcessError as exc:
        write_job_status(
            job_dir=job_dir,
            status="failed",
            message=f"Pipeline command failed with exit code {exc.returncode}.",
            progress_pct=100,
            stage_key="failed",
            stage_label="Failed",
        )
        raise

    return {
        "csv_path": csv_path,
        "summary_path": summary_path,
        "video_path": video_path,
        "line_frac": line_frac,
        "direction": direction,
        "status": "completed",
    }