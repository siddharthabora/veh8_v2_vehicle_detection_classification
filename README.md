# Vehicle Volume Analyzer

A lightweight traffic video analytics system built around a real computer vision pipeline and a FastAPI web dashboard.

It detects vehicles in video, tracks them across frames, counts line crossings, logs crossing events, renders an annotated output video, and exposes the results through a browser-based dashboard with charts, downloads, and an AI chat assistant.

## Current Scope

This repo now includes two connected parts:

1. **Core CV pipeline**
   - YOLO-based vehicle detection
   - Centroid-based tracking
   - Direction-aware line crossing
   - Event logging to CSV
   - Annotated output video rendering

2. **Web dashboard**
   - Local video preview before upload
   - Draggable horizontal counting line
   - Async processing workflow
   - Progress overlay with stage polling
   - Input and processed video preview
   - Timeline analytics and class breakdown
   - Event log table
   - CSV and annotated video download
   - AI chat over processed results with Gemini-first plus local fallback

## Vehicle Classes

The current trained model supports these classes:

- auto
- bus
- car
- light_motor_vehicle
- motorcycle
- multi-axle
- tractor
- truck

## Why this project exists

This project is designed for lightweight traffic volume analysis, especially for mixed-road contexts where vehicle diversity matters.

It is currently optimized for:
- uploaded video analysis
- line-crossing based vehicle counting
- per-job result packaging
- simple local deployment through FastAPI

## Project Structure

```text
veh8_v2_vehicle_detection_classification/
├── app/
│   ├── main.py
│   ├── services/
│   │   ├── genai_chat.py
│   │   ├── pipeline_runner.py
│   │   ├── video_metadata.py
│   │   └── video_scene_analyzer.py
│   ├── static/
│   │   ├── progress-overlay.css
│   │   ├── progress-overlay.js
│   │   └── styles.css
│   └── templates/
│       ├── index.html
│       └── results.html
├── configs/
│   ├── counter.yaml
│   └── tracker.yaml
├── models/
│   └── veh8_v2_best.pt
├── scripts/
│   ├── count_video.py
│   └── render_annotated_video.py
├── src/
│   ├── counting/
│   ├── detection/
│   ├── evaluation/
│   ├── geometry/
│   ├── tracking/
│   └── utils/
├── storage/
│   ├── jobs/
│   └── uploads/
├── tests/
├── .env
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
└── run_demo.sh
````

## Web App Features

The dashboard currently supports:

* upload video from browser
* local preview before processing
* draggable horizontal counting line
* runtime line position passed into real counting and real annotated rendering
* async job creation through API
* progress polling and overlay stages
* setup preview and processed preview
* traffic highlights cards
* volume timeline with multiple modes
* class breakdown pie chart
* crossing event table
* CSV download
* annotated video download
* AI chat on results page

## Current Runtime Behavior

Current runtime assumptions:

* supported direction: `top_to_bottom`
* line position is controlled through `line_frac`
* uploaded videos are processed into per-job folders under `storage/jobs/<job_id>/`

Each job folder can contain:

* original uploaded input video
* `crossing_events.csv`
* `summary.json`
* `annotated_output.mp4`
* `status.json`
* `job_info.txt`
* `visual_inference.json`
* `chat_cache.json`

These generated artifacts are intentionally ignored by Git.

## FastAPI Routes

Current main routes include:

* `GET /`
* `POST /process`
* `GET /results/{job_id}`
* `GET /files/{job_id}/input`
* `GET /files/{job_id}/csv`
* `GET /files/{job_id}/video`
* `GET /files/{job_id}/video/download`
* `GET /api/results/{job_id}`
* `POST /api/jobs`
* `GET /api/jobs/{job_id}/status`
* `POST /api/jobs/{job_id}/chat`

## AI Chat

The results page includes an AI chat assistant.

Current behavior:

* primary model: `gemini-2.5-flash`
* API key loaded from repo-root `.env`
* local-first routing is used for simple deterministic analytics queries
* local fallback is used when the API key is missing, quota is exhausted, or the Gemini request fails
* visual scene inference is cached per job and treated as unverified hypothesis, not confirmed metadata

The chat can currently answer questions about:

* total vehicle count
* class distribution
* highest or lowest class totals
* peak or trough traffic periods
* basic flow patterns
* cached visual scene cues

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Current `requirements.txt` includes both:

* CV pipeline dependencies
* web dashboard/runtime dependencies

## Environment Setup

Create a repo-root `.env` file if you want AI chat enabled:

```env
GEMINI_API_KEY=your_key_here
```

Notes:

* `.env` is ignored by Git
* the app still works without Gemini, but chat will rely on local fallback behavior

## Quick Start for Local Testing

This repo is now set up so a GitHub visitor can run the web dashboard locally and test it with their own videos.

### 1. Clone the repo

```bash
git clone https://github.com/siddharthabora/veh8_v2_vehicle_detection_classification.git
cd veh8_v2_vehicle_detection_classification
```

### 2. Create and activate a Python environment

Example with `venv`:

```bash
python3 -m venv venv
source venv/bin/activate
```

If you already use Conda:

```bash
conda activate venv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: add Gemini API key

Create `.env` in repo root:

```env
GEMINI_API_KEY=your_key_here
```

### 5. Run the local web app

```bash
./run_demo.sh
```

If port `8000` is already in use:

```bash
PORT=8001 ./run_demo.sh
```

Then open:

* `http://127.0.0.1:8000`
* or `http://127.0.0.1:8001` if you changed the port

## How to Use the Dashboard

1. Open the home page
2. Select a traffic video from your device
3. Review the local preview before upload
4. Drag the horizontal counting line to the desired position
5. Click **Process Video**
6. Wait for the progress overlay to complete
7. Review results:

   * input preview
   * processed preview
   * traffic highlights
   * timeline chart
   * class breakdown
   * crossing events
   * CSV and video downloads
   * AI chat

## Core Pipeline Notes

Main model:

* `models/veh8_v2_best.pt`

Key scripts:

* `scripts/count_video.py`
* `scripts/render_annotated_video.py`

Important source areas:

* `src/detection/yolo_detector.py`
* `src/tracking/centroid_tracker.py`
* `src/counting/line_counter.py`
* `src/geometry/line.py`
* `src/evaluation/event_logger.py`
* `src/evaluation/video_evaluator.py`
* `src/utils/config_loader.py`

Configs:

* `configs/tracker.yaml`
* `configs/counter.yaml`

## Tests

Run tests with:

```bash
pytest
```

Current tracked tests include:

* config loading
* line geometry
* line crossing behavior

## Security Notes

Current hardening includes:

* `.env` ignored by Git
* generated storage folders ignored by Git
* upload validation in FastAPI
* runtime parameter validation for `line_frac` and `direction`
* safer HTML error rendering
* sanitized LLM-bound metadata
* reduced leakage of provider/debug errors to the browser

See `SECURITY.md` for a focused security overview.

## Known Current Constraints

* current runtime direction support is only `top_to_bottom`
* local hosting is the primary supported mode right now
* no auth or multi-user isolation layer is included yet
* generated job data is stored locally in `storage/jobs/`
* public-cloud deployment has not yet been hardened as a production SaaS setup

## Roadmap Direction

Current likely next areas:

* deeper security and vulnerability testing
* additional automated tests for app routes
* better app packaging for new users
* controlled GitHub publishing of the web dashboard phase
* future conversational analytics improvements

## License

See `LICENSE`.

````

Then save it and run:

```bash
python -m py_compile app/main.py
````
