#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "========================================"
echo " Vehicle Volume Analyzer - Local Web App"
echo "========================================"
echo ""
echo "Starting FastAPI dashboard on http://${HOST}:${PORT}"
echo ""
echo "Before running:"
echo "1. Activate your Python environment"
echo "2. Ensure dependencies are installed:"
echo "   pip install -r requirements.txt"
echo "3. Ensure .env exists in repo root if AI chat is needed"
echo ""

python -m uvicorn app.main:app --host "${HOST}" --port "${PORT}" --reload
