#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/lf/conda/envs/my_ocr_env/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export RECSYS_TEXT_DEVICE="${RECSYS_TEXT_DEVICE:-cpu}"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" -m uvicorn api.main:app --host "${HOST}" --port "${PORT}"
