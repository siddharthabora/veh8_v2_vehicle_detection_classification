#!/usr/bin/env bash
set -e

INPUT_VIDEO="${1:-}"
MODEL_PATH="models/veh8_v2_best.pt"
CSV_OUT="outputs/crossing_events.csv"
VIDEO_OUT="outputs/annotated_output.mp4"

if [ -z "$INPUT_VIDEO" ]; then
  echo "Usage:"
  echo "  ./run_demo.sh <path_to_video>"
  echo ""
  echo "Example:"
  echo "  ./run_demo.sh outputs/count_test_4.mp4"
  exit 1
fi

if [ ! -f "$INPUT_VIDEO" ]; then
  echo "Error: video file not found: $INPUT_VIDEO"
  exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Error: model file not found: $MODEL_PATH"
  exit 1
fi

echo "Vehicle Traffic Counter Demo"
echo "--------------------------------"
echo "Input video: $INPUT_VIDEO"
echo "Model: $MODEL_PATH"
echo ""

echo "Creating Python environment..."
python3.11 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Running vehicle counting pipeline..."
echo ""

python -m scripts.count_video \
  --video "$INPUT_VIDEO" \
  --model "$MODEL_PATH"

echo ""
echo "Rendering annotated video..."
echo ""

python -m scripts.render_annotated_video \
  --video "$INPUT_VIDEO" \
  --model "$MODEL_PATH" \
  --output "$VIDEO_OUT"

echo ""
echo "Results generated:"
echo "$CSV_OUT"
echo "$VIDEO_OUT"
echo ""
echo "Demo complete."
