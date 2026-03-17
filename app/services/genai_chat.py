import json
import os
import re
from typing import Any

from google import genai
from google.genai import types


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MAX_USER_MESSAGE_CHARS = 2000
MAX_CROSSING_EVENTS_IN_PROMPT = 100
MAX_VISUAL_CUES = 5


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        return "{}"


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    return " ".join(cleaned.split())


def _truncate_text(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()


def _sanitize_visual_inference_result(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    visual = data.get("visual_inference", {})
    if not isinstance(visual, dict):
        visual = {}

    raw_cues = visual.get("scene_cues", [])
    if isinstance(raw_cues, list):
        cues = [str(item).strip() for item in raw_cues[:MAX_VISUAL_CUES] if str(item).strip()]
    else:
        cues = []

    return {
        "ok": bool(data.get("ok", False)),
        "error": "",
        "visual_inference": {
            "possible_region": str(visual.get("possible_region", "") or "").strip(),
            "possible_country": str(visual.get("possible_country", "") or "").strip(),
            "road_context": str(visual.get("road_context", "") or "").strip(),
            "traffic_side_hint": str(visual.get("traffic_side_hint", "") or "").strip(),
            "environment_type": str(visual.get("environment_type", "") or "").strip(),
            "scene_cues": cues,
            "confidence": str(visual.get("confidence", "") or "").strip(),
            "assumption_only": bool(visual.get("assumption_only", False)),
            "disclaimer": str(visual.get("disclaimer", "") or "").strip(),
        },
        "sampled_frame_count": int(data.get("sampled_frame_count", 0) or 0),
        "sampled_timestamps_seconds": data.get("sampled_timestamps_seconds", []) or [],
        "model": str(data.get("model", "") or "").strip(),
    }


def _sanitize_verified_video_metadata(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    allowed_keys = {
        "duration_seconds",
        "fps",
        "frame_count",
        "width",
        "height",
        "created_at",
        "captured_at",
        "timezone",
        "location_name",
        "country",
        "region",
        "city",
    }

    sanitized = {}
    for key in allowed_keys:
        if key in data:
            sanitized[key] = data.get(key)

    return sanitized


def _sanitize_metadata_context(metadata_context: dict[str, Any] | None) -> dict[str, Any]:
    metadata_context = metadata_context or {}

    return {
        "verified_video_metadata": _sanitize_verified_video_metadata(
            metadata_context.get("verified_video_metadata", {})
        ),
        "visual_inference_result": _sanitize_visual_inference_result(
            metadata_context.get("visual_inference_result", {})
        ),
    }


def _sanitize_crossing_events_for_prompt(
    crossing_events: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    sanitized_events: list[dict[str, Any]] = []

    for event in (crossing_events or [])[:MAX_CROSSING_EVENTS_IN_PROMPT]:
        try:
            timestamp_seconds = float(event.get("timestamp_seconds", 0) or 0)
        except (TypeError, ValueError):
            timestamp_seconds = 0.0

        try:
            crossing_frame = int(float(event.get("crossing_frame", 0) or 0))
        except (TypeError, ValueError):
            crossing_frame = 0

        track_id_raw = event.get("track_id")
        try:
            track_id = int(float(track_id_raw)) if track_id_raw is not None else None
        except (TypeError, ValueError):
            track_id = None

        sanitized_events.append(
            {
                "track_id": track_id,
                "vehicle_class": str(event.get("vehicle_class", "") or "").strip(),
                "direction": str(event.get("direction", "") or "").strip(),
                "crossing_frame": crossing_frame,
                "timestamp_seconds": round(max(0.0, timestamp_seconds), 3),
            }
        )

    return sanitized_events


def _build_system_instruction() -> str:
    return """
You are an AI traffic analytics assistant for a Vehicle Volume Analyzer dashboard.

Your role:
- Answer user questions about one processed traffic-video job.
- Be practical, analytical, and conversational.
- Prefer direct answers over rigid report formatting.

Evidence hierarchy:
1. Verified job analytics from summary data and crossing-event data are the primary source of truth.
2. File or embedded metadata, if explicitly provided in context, counts as verified metadata.
3. Visual scene inference from processed-video frames is hypothesis-level evidence only unless independently verified by metadata.
4. General traffic knowledge may be used for interpretation, but must not be presented as job-specific fact.

Core rules:
1. Use the provided job context as the source of truth.
2. You may derive secondary insights from the provided data when they can be computed reasonably.
3. If timestamps are available in crossing events, you should estimate peaks, troughs, clustering, density windows, dominant vehicle classes, and other simple traffic patterns from those timestamps.
4. Do not refuse just because a value is not pre-aggregated. If it can be estimated from available event-level data, do the estimation and clearly state the basis.
5. If a result is derived or approximate, say so briefly and explain the assumption in one sentence.
6. If the answer truly cannot be determined from the available data, say that clearly and briefly.
7. Do not invent location, date, weather, season, camera metadata, or legal violations.
8. If verified location/date/time metadata is missing, say that geography-aware or seasonal analysis is limited.
9. Separate job-specific facts from general domain reasoning.
10. For simple user questions, answer directly in 1 to 4 short paragraphs. Do not force section headers unless they improve clarity.
11. Avoid repeating the same caveats in every response. Mention limitations only when they materially affect the answer.
12. If the user challenges your earlier refusal or asks whether you can calculate something, attempt the calculation from the available data before declining.
13. Be concise, useful, and non-robotic.
14. If visual scene inference suggests a possible region or country, label it clearly as a visual hypothesis, not a verified fact.
15. When mentioning possible geography from visuals, mention the cues briefly, such as traffic side, vehicle mix, road context, signage style, or roadside environment.
16. Never present a guessed country, city, or region as confirmed unless explicit verified metadata supports it.
17. Use cautious phrasing for visual geography, such as:
    - "may be from..."
    - "one plausible hypothesis is..."
    - "this is not verified"
    - "based on visual cues only..."
18. Avoid assertive geography phrasing such as:
    - "this is from..."
    - "this appears to be from..."
    - "likely from..." unless immediately softened as unverified hypothesis
19. If the user asks where the video is from, explicitly state whether the answer is verified metadata or visual inference.
20. If verified location metadata is absent, say so in one short sentence.
21. Country guesses should be framed as one plausible option, not the default conclusion.
22. Distinguish clearly between:
    - verified metadata
    - visual inference
    - computed analytics
    - general traffic knowledge

Answer style:
- For short questions, lead with the answer immediately.
- For analytical questions, use this pattern when useful:
  answer -> brief reasoning -> limitation if relevant
- Avoid boilerplate like repeating "What the data shows" every turn.
- Sound like a sharp traffic analyst, not a compliance bot.

Allowed examples of derivation from current job data:
- estimate the busiest short interval from crossing timestamps
- identify dominant vehicle class in the densest interval
- compare concentration across time windows
- comment on whether the traffic mix is highly skewed
- infer whether the clip suggests bursty flow vs steady sparse flow
- discuss visually plausible geography while labeling it as assumption-only

Not allowed:
- claiming exact geography-specific traffic rules without verified location context
- claiming seasonal patterns without date/time/location context
- claiming accident, violation, or enforcement conclusions without explicit evidence
- presenting visual scene guesses as verified metadata
""".strip()


def _build_user_prompt(
    user_message: str,
    summary_data: dict[str, Any] | None,
    crossing_events: list[dict[str, Any]] | None,
    job_info: dict[str, Any] | None,
    metadata_context: dict[str, Any] | None = None,
) -> str:
    safe_metadata_context = _sanitize_metadata_context(metadata_context)
    safe_crossing_events = _sanitize_crossing_events_for_prompt(crossing_events)

    prompt_payload = {
        "user_question": _truncate_text(user_message, MAX_USER_MESSAGE_CHARS),
        "job_summary": summary_data or {},
        "job_info": job_info or {},
        "verified_video_metadata": safe_metadata_context.get("verified_video_metadata", {}),
        "visual_inference_result": safe_metadata_context.get("visual_inference_result", {}),
        "crossing_events_sample": safe_crossing_events,
        "crossing_event_count": len(crossing_events or []),
    }

    return f"""
Answer the user's question using the traffic video result context below.

Important instructions for this specific task:
- If the user asks about peak, trough, highest density, clustering, or dominant class over time, compute a reasonable answer from the provided crossing timestamps when possible.
- Do not refuse only because the data is not already aggregated.
- If needed, estimate short traffic concentration windows from event timestamps.
- If the answer is approximate, say so briefly.
- Keep the response direct and natural.
- Avoid repeating generic caveats unless they are relevant to the specific question.
- If visual_inference_result is present, you may use it to discuss visually plausible scene context or possible geography, but you must label it clearly as visual inference or hypothesis unless verified metadata supports it.
- If verified_video_metadata is empty or missing location/date/time fields, do not present geography or seasonality as confirmed fact.
- When discussing possible location from visuals, prefer wording like "may be from..." or "one plausible hypothesis is...".
- Do not use confident wording like "this is from..." or "it is likely from..." unless you immediately clarify that it is unverified visual inference.
- If the user asks whether the location is verified, answer that directly and explicitly.
- When discussing possible location from visuals, mention the specific cues briefly.

RESULT_CONTEXT_JSON:
{_safe_json(prompt_payload)}
""".strip()


def _build_local_fallback_response(
    user_message: str,
    summary_data: dict[str, Any] | None,
    crossing_events: list[dict[str, Any]] | None,
    job_info: dict[str, Any] | None,
    metadata_context: dict[str, Any] | None,
    model_name: str,
    fallback_reason: str,
    retryable: bool = False,
) -> dict[str, Any]:
    fallback = local_fallback_answer(
        user_message=user_message,
        summary_data=summary_data or {},
        crossing_events=crossing_events or [],
        job_info=job_info or {},
        metadata_context=_sanitize_metadata_context(metadata_context),
    )

    response = {
        "ok": True,
        "error": "",
        "answer": fallback["answer"],
        "model": model_name,
        "response_mode": "local_fallback",
        "evidence": fallback["evidence"],
        "fallback_reason": fallback_reason,
    }

    if retryable:
        response["retryable"] = True

    return response


def ask_gemini_about_job(
    user_message: str,
    summary_data: dict[str, Any] | None,
    crossing_events: list[dict[str, Any]] | None,
    job_info: dict[str, Any] | None,
    metadata_context: dict[str, Any] | None = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> dict[str, Any]:
    clean_user_message = _truncate_text(user_message, MAX_USER_MESSAGE_CHARS)

    if not clean_user_message:
        return {
            "ok": False,
            "error": "Empty user message.",
            "answer": "",
            "model": model_name,
        }

    if should_use_local_fallback_first(clean_user_message):
        return _build_local_fallback_response(
            user_message=clean_user_message,
            summary_data=summary_data,
            crossing_events=crossing_events,
            job_info=job_info,
            metadata_context=metadata_context,
            model_name=model_name,
            fallback_reason="local_first_router",
        )

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return _build_local_fallback_response(
            user_message=clean_user_message,
            summary_data=summary_data,
            crossing_events=crossing_events,
            job_info=job_info,
            metadata_context=metadata_context,
            model_name=model_name,
            fallback_reason="missing_api_key",
        )

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model_name,
            contents=_build_user_prompt(
                user_message=clean_user_message,
                summary_data=summary_data,
                crossing_events=crossing_events,
                job_info=job_info,
                metadata_context=metadata_context,
            ),
            config=types.GenerateContentConfig(
                system_instruction=_build_system_instruction(),
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

        answer_text = (response.text or "").strip()
        if not answer_text:
            answer_text = "I could not generate a response for this result."

        return {
            "ok": True,
            "error": "",
            "answer": answer_text,
            "model": model_name,
            "response_mode": "gemini",
        }

    except Exception as exc:
        normalized_error = str(exc).lower()

        if "429" in normalized_error or "resource_exhausted" in normalized_error or "quota" in normalized_error:
            return _build_local_fallback_response(
                user_message=clean_user_message,
                summary_data=summary_data,
                crossing_events=crossing_events,
                job_info=job_info,
                metadata_context=metadata_context,
                model_name=model_name,
                fallback_reason="quota_exceeded",
                retryable=True,
            )

        return _build_local_fallback_response(
            user_message=clean_user_message,
            summary_data=summary_data,
            crossing_events=crossing_events,
            job_info=job_info,
            metadata_context=metadata_context,
            model_name=model_name,
            fallback_reason="request_failed",
            retryable=True,
        )


def should_use_local_fallback_first(user_message: str) -> bool:
    """
    Route simple, deterministic analytics questions to the local fallback
    before calling Gemini. This reduces Gemini usage during testing.
    """

    text = _normalize_text(user_message)

    if not text:
        return True

    local_patterns = [
        "total vehicle",
        "vehicle count",
        "counts by class",
        "count by class",
        "vehicle types",
        "class breakdown",
        "show class counts",
        "show me class counts",
        "highest vehicle class",
        "highest class",
        "top vehicle class",
        "top class",
        "most common class",
        "which class was highest",
        "which vehicle class was highest",
        "which class highest",
        "which vehicle class highest",
        "highest vehicle count",
        "maximum vehicle count",
        "lowest vehicle class",
        "lowest class",
        "least common class",
        "which class was lowest",
        "which vehicle class was lowest",
        "which class lowest",
        "which vehicle class lowest",
        "lowest vehicle count",
        "minimum vehicle count",
        "dominant class",
        "dominant vehicle class",
        "highest cumulative",
        "lowest cumulative",
        "cumulative class",
        "cummulative",
        "busiest period",
        "peak",
        "peak time",
        "trough",
        "quietest period",
        "least busy",
        "dominated the peak",
        "bursty",
        "steady",
        "sparse",
        "flow",
        "where might this video be from",
        "is that verified",
        "visual cues",
        "left side or right side traffic",
        "right side or left side traffic",
        "location guess",
    ]

    if text in {"highest", "lowest", "peak", "trough"}:
        return True

    return any(pattern in text for pattern in local_patterns)


def local_fallback_answer(
    user_message: str,
    summary_data: dict,
    crossing_events: list,
    job_info: dict,
    metadata_context: dict,
):
    """
    Deterministic fallback answers when Gemini is unavailable.

    Supports:
    - total vehicle count
    - counts by class
    - highest / lowest cumulative class
    - highest / lowest cumulative count with class names
    - peak / trough
    - flow characterization
    - cached visual inference
    - compound questions that ask for multiple supported facts at once
    """

    text = _normalize_text(user_message)

    total_count = int(summary_data.get("total_count", 0) or 0)
    class_counts = summary_data.get("counts_by_class", {}) or {}
    highlights = summary_data.get("traffic_highlights", {}) or {}

    peak = highlights.get("peak", {}) or {}
    trough = highlights.get("trough", {}) or {}

    safe_metadata_context = _sanitize_metadata_context(metadata_context)
    visual = safe_metadata_context.get("visual_inference_result", {}) or {}
    visual_data = visual.get("visual_inference", {}) or {}

    response_parts: list[str] = []
    evidence_kinds: set[str] = set()

    def humanize_class_name(name: str) -> str:
        return str(name or "").replace("_", " ").strip()

    def vehicle_word(count: int) -> str:
        return "vehicle" if int(count or 0) == 1 else "vehicles"

    def nonzero_class_counts() -> dict[str, int]:
        return {
            cls: int(count or 0)
            for cls, count in class_counts.items()
            if int(count or 0) > 0
        }

    def format_class_list(items: list[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"

    def compute_bucket_highlight(events: list, mode: str = "peak") -> dict:
        normalized_events = []

        for event in events or []:
            try:
                ts = float(event.get("timestamp_seconds", 0) or 0)
            except (TypeError, ValueError):
                ts = 0.0

            normalized_events.append(
                {
                    "timestamp_seconds": max(0.0, ts),
                    "vehicle_class": str(event.get("vehicle_class", "") or "").strip() or "unknown",
                }
            )

        if not normalized_events:
            return {}

        max_ts = max(event["timestamp_seconds"] for event in normalized_events)

        if max_ts < 60:
            bucket_size = 5
        elif max_ts <= 300:
            bucket_size = 30
        elif max_ts <= 1800:
            bucket_size = 60
        else:
            bucket_size = 300

        bucket_count = max(1, int(max_ts // bucket_size) + 1)
        bucket_totals = [0] * bucket_count
        bucket_class_counts = [{} for _ in range(bucket_count)]

        for event in normalized_events:
            bucket_index = min(int(event["timestamp_seconds"] // bucket_size), bucket_count - 1)
            bucket_totals[bucket_index] += 1

            cls = event["vehicle_class"]
            bucket_class_counts[bucket_index][cls] = bucket_class_counts[bucket_index].get(cls, 0) + 1

        candidate_indexes = [idx for idx, count in enumerate(bucket_totals) if count > 0]
        if not candidate_indexes:
            return {}

        if mode == "peak":
            target_index = max(candidate_indexes, key=lambda idx: bucket_totals[idx])
        else:
            target_index = min(candidate_indexes, key=lambda idx: bucket_totals[idx])

        target_total = bucket_totals[target_index]
        target_class_counts = bucket_class_counts[target_index]

        dominant_class = "No data"
        if target_class_counts:
            dominant_class = max(
                target_class_counts.items(),
                key=lambda item: (item[1], item[0]),
            )[0]

        start_sec = target_index * bucket_size
        end_sec = start_sec + bucket_size

        if bucket_size < 60:
            time_range = f"{int(start_sec)}-{int(end_sec)} sec"
        else:
            time_range = f"{int(start_sec // 60)}-{int(end_sec // 60)} min"

        return {
            "time_range": time_range,
            "vehicle_class": dominant_class,
            "vehicle_count": target_total,
        }

    def flow_character(events: list) -> str:
        timestamps = []

        for event in events or []:
            try:
                timestamps.append(float(event.get("timestamp_seconds", 0) or 0))
            except (TypeError, ValueError):
                continue

        timestamps = sorted(ts for ts in timestamps if ts >= 0)

        if len(timestamps) < 3:
            return "too sparse to characterize confidently"

        gaps = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)

        if avg_gap <= 2.0:
            return "fairly dense"
        if max_gap >= avg_gap * 4:
            return "bursty"
        if avg_gap >= 8.0:
            return "sparse"
        return "moderately steady"

    asks_total = "total" in text and "vehicle" in text

    asks_counts_by_class = (
        "counts by class" in text
        or "count by class" in text
        or "vehicle types" in text
        or "class breakdown" in text
        or "show class counts" in text
        or "show me class counts" in text
    )

    asks_highest_class = (
        "highest vehicle class" in text
        or "highest class" in text
        or "top vehicle class" in text
        or "top class" in text
        or "most common class" in text
        or "which class was highest" in text
        or "which vehicle class was highest" in text
        or "which class highest" in text
        or "which vehicle class highest" in text
        or text == "highest"
    )

    asks_lowest_class = (
        "lowest vehicle class" in text
        or "lowest class" in text
        or "least common class" in text
        or "which class was lowest" in text
        or "which vehicle class was lowest" in text
        or "which class lowest" in text
        or "which vehicle class lowest" in text
        or text == "lowest"
    )

    asks_highest_count = (
        "highest vehicle count" in text
        or "maximum vehicle count" in text
        or "highest count" in text
        or "maximum count" in text
    )

    asks_lowest_count = (
        "lowest vehicle count" in text
        or "minimum vehicle count" in text
        or "lowest count" in text
        or "minimum count" in text
    )

    asks_peak = "peak" in text or "busiest" in text
    asks_trough = "trough" in text or "quiet" in text or "least busy" in text
    asks_flow = "bursty" in text or "steady" in text or "sparse" in text or "flow" in text
    asks_visual = (
        "where" in text
        or "location" in text
        or "country" in text
        or "verified" in text
        or "visual cues" in text
    )

    counts_nonzero = nonzero_class_counts()

    if asks_total:
        resolved_total = total_count if total_count > 0 else len(crossing_events or [])
        response_parts.append(
            f"A total of **{resolved_total} {vehicle_word(resolved_total)}** were counted crossing the line in this video."
        )
        evidence_kinds.add("events")

    if asks_counts_by_class:
        if not class_counts:
            response_parts.append("No class breakdown data is available for this job.")
        else:
            parts = []
            for cls, count in class_counts.items():
                count_int = int(count or 0)
                if count_int > 0:
                    parts.append(f"{humanize_class_name(cls)}: {count_int}")

            if parts:
                response_parts.append(f"Vehicle counts by class: {', '.join(parts)}.")
            else:
                response_parts.append("The class breakdown is available, but all counted classes are currently zero.")

        evidence_kinds.add("events")

    if asks_highest_class or asks_highest_count:
        if not counts_nonzero:
            response_parts.append("No non-zero class totals are available for this job.")
        else:
            top_count = max(counts_nonzero.values())
            top_classes = sorted(
                humanize_class_name(cls)
                for cls, count in counts_nonzero.items()
                if count == top_count
            )

            if asks_highest_count and not asks_highest_class:
                response_parts.append(
                    f"The highest vehicle count was **{top_count} {vehicle_word(top_count)}**, for **{format_class_list(top_classes)}**."
                )
            elif len(top_classes) == 1:
                response_parts.append(
                    f"The highest cumulative vehicle class was **{top_classes[0]}**, with **{top_count} {vehicle_word(top_count)}**."
                )
            else:
                response_parts.append(
                    f"The highest cumulative vehicle classes were **{format_class_list(top_classes)}**, each with **{top_count} {vehicle_word(top_count)}**."
                )

        evidence_kinds.add("events")

    if asks_lowest_class or asks_lowest_count:
        if not counts_nonzero:
            response_parts.append("No non-zero class totals are available for this job.")
        else:
            low_count = min(counts_nonzero.values())
            low_classes = sorted(
                humanize_class_name(cls)
                for cls, count in counts_nonzero.items()
                if count == low_count
            )

            if asks_lowest_count and not asks_lowest_class:
                response_parts.append(
                    f"The lowest vehicle count was **{low_count} {vehicle_word(low_count)}**, for **{format_class_list(low_classes)}**."
                )
            elif len(low_classes) == 1:
                response_parts.append(
                    f"The lowest cumulative vehicle class was **{low_classes[0]}**, with **{low_count} {vehicle_word(low_count)}**."
                )
            else:
                response_parts.append(
                    f"The lowest cumulative vehicle classes were **{format_class_list(low_classes)}**, each with **{low_count} {vehicle_word(low_count)}**."
                )

        evidence_kinds.add("events")

    if asks_peak:
        resolved_peak = peak or compute_bucket_highlight(crossing_events, mode="peak")

        if not resolved_peak:
            response_parts.append("The busiest period could not be determined from the available event data.")
        else:
            time_range = resolved_peak.get("time_range", "No data")
            vehicle_class = humanize_class_name(resolved_peak.get("vehicle_class", "No data"))
            count = int(resolved_peak.get("vehicle_count", 0) or 0)

            response_parts.append(
                f"The busiest interval was **{time_range}**, with **{count} {vehicle_word(count)}**, dominated by **{vehicle_class}**."
            )

        evidence_kinds.add("events")

    if asks_trough:
        resolved_trough = trough or compute_bucket_highlight(crossing_events, mode="trough")

        if not resolved_trough:
            response_parts.append("The quietest period could not be determined from the available event data.")
        else:
            time_range = resolved_trough.get("time_range", "No data")
            vehicle_class = humanize_class_name(resolved_trough.get("vehicle_class", "No data"))
            count = int(resolved_trough.get("vehicle_count", 0) or 0)

            response_parts.append(
                f"The quietest interval was **{time_range}**, with **{count} {vehicle_word(count)}**, mainly **{vehicle_class}**."
            )

        evidence_kinds.add("events")

    if asks_flow:
        character = flow_character(crossing_events)
        response_parts.append(f"From the crossing-event timings, the flow looks **{character}**.")
        evidence_kinds.add("events")

    if asks_visual:
        region = visual_data.get("possible_region")
        country = visual_data.get("possible_country")
        cues = visual_data.get("scene_cues", []) or []

        if not region and not country:
            response_parts.append(
                "There is no cached visual inference available for this job, so I cannot suggest a plausible location."
            )
        else:
            cues_text = ", ".join(cues[:3]) if cues else "general road layout and vehicle mix"
            answer_parts = []

            if region and country:
                answer_parts.append(
                    f"The scene **may be from {region}**, with **{country} as one plausible hypothesis**."
                )
            elif region:
                answer_parts.append(f"The scene **may be from {region}** based on visual cues.")
            elif country:
                answer_parts.append(f"**{country}** is one plausible visual hypothesis for this scene.")

            answer_parts.append("This is **not verified metadata** and should be treated as visual inference only.")
            answer_parts.append(f"Some of the cues used were: {cues_text}.")

            response_parts.append(" ".join(answer_parts))

        evidence_kinds.add("visual")

    if response_parts:
        evidence = "visual" if evidence_kinds == {"visual"} else "events"
        return {
            "answer": " ".join(response_parts),
            "evidence": evidence,
        }

    return {
        "answer": (
            "AI is temporarily unavailable, but I can still answer basic analytics questions "
            "such as total vehicle count, class distribution, highest or lowest class totals, highest or lowest vehicle counts, "
            "busiest period, quietest period, flow pattern, or cached visual scene cues."
        ),
        "evidence": "events",
    }