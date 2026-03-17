# Version Notes

## Current Phase

This repository has now moved beyond the training-only stage and includes the first working web application layer for the Vehicle Volume Analyzer.

## Current Version

**Vehicle Volume Analyzer - Web Dashboard Phase 1**

## What is now included

### Core CV pipeline
- YOLO-based vehicle detection
- centroid tracking
- direction-aware line crossing
- event logging to CSV
- annotated video rendering

### Web dashboard
- FastAPI backend
- server-rendered frontend
- local preview before upload
- draggable horizontal counting line
- async job processing with progress polling
- input and processed video preview
- timeline analytics
- class breakdown
- event table
- CSV download
- annotated video download
- AI chat on results page

### AI layer
- Gemini-first conversational analytics
- local-first routing for simple deterministic questions
- local fallback on missing key, quota exhaustion, or request failure
- cached visual scene inference per job
- cached chat responses per job

## Important runtime notes

- current supported runtime direction: `top_to_bottom`
- `line_frac` is passed through to both counting and annotated rendering
- per-job artifacts are stored locally under `storage/jobs/<job_id>/`
- `.env` and generated artifacts are ignored by Git

## Security hardening added in this phase

- upload validation in FastAPI routes
- runtime validation for `line_frac` and `direction`
- safer escaped HTML error pages
- reduced metadata exposure to LLM requests
- safer chat response sanitization
- secret and artifact ignore rules improved in `.gitignore`

## Repo onboarding improvements added in this phase

- `requirements.txt` updated for webapp runtime
- `run_demo.sh` now launches the local FastAPI dashboard
- `README.md` updated for local tester quick-start
- `SECURITY.md` updated for current project scope

## Suggested tag / release label

`v0.2.0-web-dashboard-local`

## Summary

This phase marks the transition from a model-training repository into a usable local traffic-analysis application that GitHub users can run and test on their own devices.