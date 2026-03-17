import base64
import json
import os
from pathlib import Path
from typing import Any

import cv2
from google import genai
from google.genai import types


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_SAMPLED_FRAMES = 4
JPEG_QUALITY = 80
MAX_DIMENSION = 1280
MAX_SCENE_CUES = 5


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return "{}"


def _sanitize_job_info(job_info: dict[str, Any] | None) -> dict[str, Any]:
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
    for key in allowed_keys:
        if key in job_info:
            sanitized[key] = job_info.get(key)

    return sanitized


def _sanitize_job_summary(job_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(job_summary, dict):
        return {}

    allowed_keys = {
        "total_count",
        "counts_by_class",
        "traffic_highlights",
        "timeline",
        "video_duration_seconds",
    }

    sanitized = {}
    for key in allowed_keys:
        if key in job_summary:
            sanitized[key] = job_summary.get(key)

    return sanitized


def _resize_frame_if_needed(frame):
    height, width = frame.shape[:2]
    longest_side = max(height, width)

    if longest_side <= MAX_DIMENSION:
        return frame

    scale = MAX_DIMENSION / float(longest_side)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _encode_frame_jpeg_base64(frame) -> str:
    resized = _resize_frame_if_needed(frame)
    ok, buffer = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        raise ValueError("Failed to encode frame to JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _sample_video_frames(video_path: Path, max_frames: int = MAX_SAMPLED_FRAMES) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            return []

        sample_indexes = []
        for ratio in [0.15, 0.35, 0.6, 0.85]:
            idx = min(frame_count - 1, max(0, int(frame_count * ratio)))
            sample_indexes.append(idx)

        sample_indexes = sorted(set(sample_indexes))[:max_frames]

        sampled = []
        for idx in sample_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            timestamp_seconds = (idx / fps) if fps > 0 else None
            sampled.append(
                {
                    "frame_index": idx,
                    "timestamp_seconds": round(timestamp_seconds, 2) if timestamp_seconds is not None else None,
                    "jpeg_base64": _encode_frame_jpeg_base64(frame),
                }
            )

        return sampled
    finally:
        cap.release()


def _build_scene_system_instruction() -> str:
    return """
You are analyzing sampled frames from a processed traffic video.

Your task is to infer broad scene context from visible cues only.

Rules:
1. Distinguish clearly between verified facts and visual assumptions.
2. Do not claim an exact location unless explicit evidence is visible.
3. If no hard location metadata exists, any country or region statement must be labeled as a hypothesis.
4. Focus on visible cues such as:
   - side of road traffic
   - road width and markings
   - vehicle mix
   - signage style or language if visible
   - urban vs rural context
   - weather or lighting
   - roadside environment
5. Be conservative. Prefer "possibly" over certainty.
6. Return structured JSON only.
7. Use "unknown" when the evidence is not strong enough.
8. Keep scene_cues brief and evidence-based.
9. Do not infer precise addresses, organizations, or identities.
""".strip()


def _build_scene_prompt(job_summary: dict[str, Any] | None, job_info: dict[str, Any] | None) -> str:
    payload = {
        "job_summary": _sanitize_job_summary(job_summary),
        "job_info": _sanitize_job_info(job_info),
        "required_output_schema": {
            "possible_region": "string",
            "possible_country": "string",
            "road_context": "string",
            "traffic_side_hint": "string",
            "environment_type": "string",
            "scene_cues": ["string"],
            "confidence": "low | medium | high",
            "assumption_only": True,
            "disclaimer": "string",
        },
    }

    return f"""
Analyze the sampled processed-video frames and return one compact JSON object.

Important:
- If location is not visually inferable, say "unknown".
- If country is guessed, it must remain a hypothesis.
- Keep scene_cues short and evidence-based.
- Use only the provided visual frames and sanitized context.
- Do not rely on hidden metadata.

CONTEXT_JSON:
{_safe_json(payload)}
""".strip()


def _sanitize_visual_inference_output(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {
            "possible_region": "unknown",
            "possible_country": "unknown",
            "road_context": "",
            "traffic_side_hint": "",
            "environment_type": "",
            "scene_cues": [],
            "confidence": "low",
            "assumption_only": True,
            "disclaimer": "Model output was not in the expected object format.",
        }

    raw_cues = parsed.get("scene_cues", [])
    if isinstance(raw_cues, list):
        scene_cues = [str(item).strip() for item in raw_cues[:MAX_SCENE_CUES] if str(item).strip()]
    else:
        scene_cues = []

    confidence = str(parsed.get("confidence", "low") or "low").strip().lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    return {
        "possible_region": str(parsed.get("possible_region", "unknown") or "unknown").strip(),
        "possible_country": str(parsed.get("possible_country", "unknown") or "unknown").strip(),
        "road_context": str(parsed.get("road_context", "") or "").strip(),
        "traffic_side_hint": str(parsed.get("traffic_side_hint", "") or "").strip(),
        "environment_type": str(parsed.get("environment_type", "") or "").strip(),
        "scene_cues": scene_cues,
        "confidence": confidence,
        "assumption_only": bool(parsed.get("assumption_only", True)),
        "disclaimer": str(parsed.get("disclaimer", "") or "").strip(),
    }


def analyze_video_scene(
    video_path: Path,
    job_summary: dict[str, Any] | None,
    job_info: dict[str, Any] | None,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "ok": False,
            "error": "Missing Gemini API key.",
            "visual_inference": {},
            "model": model_name,
        }

    if not video_path.exists():
        return {
            "ok": False,
            "error": "Processed video not found.",
            "visual_inference": {},
            "model": model_name,
        }

    sampled_frames = _sample_video_frames(video_path)
    if not sampled_frames:
        return {
            "ok": False,
            "error": "Could not sample frames from processed video.",
            "visual_inference": {},
            "model": model_name,
        }

    try:
        client = genai.Client(api_key=api_key)

        contents: list[Any] = [_build_scene_prompt(job_summary=job_summary, job_info=job_info)]

        for item in sampled_frames:
            contents.append(
                types.Part.from_bytes(
                    data=base64.b64decode(item["jpeg_base64"]),
                    mime_type="image/jpeg",
                )
            )

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_build_scene_system_instruction(),
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
            ),
        )

        text = (response.text or "").strip()
        if not text:
            return {
                "ok": False,
                "error": "Gemini returned empty scene analysis.",
                "visual_inference": {},
                "model": model_name,
            }

        parsed = json.loads(text)
        sanitized_output = _sanitize_visual_inference_output(parsed)

        return {
            "ok": True,
            "error": "",
            "visual_inference": sanitized_output,
            "sampled_frame_count": len(sampled_frames),
            "sampled_timestamps_seconds": [
                item["timestamp_seconds"] for item in sampled_frames if item["timestamp_seconds"] is not None
            ],
            "model": model_name,
        }

    except Exception as exc:
        normalized_error = str(exc).lower()

        if "429" in normalized_error or "resource_exhausted" in normalized_error or "quota" in normalized_error:
            return {
                "ok": False,
                "error": "Scene analysis is temporarily unavailable due to model quota limits.",
                "visual_inference": {},
                "model": model_name,
                "retryable": True,
            }

        return {
            "ok": False,
            "error": "Scene analysis failed.",
            "visual_inference": {},
            "model": model_name,
            "retryable": True,
        }