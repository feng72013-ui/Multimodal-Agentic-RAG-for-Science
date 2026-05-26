#!/usr/bin/env bash
set -euo pipefail

# Start a DotsOCR vLLM OpenAI-compatible server for OCR batch processing.
#
# The batch OCR script expects:
#   endpoint: http://localhost:6006/v1
#   model:    dots_ocr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/lf/mount/LLM/project/test_project/MutilModel_RAG/DotsOCR}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dots_ocr}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}:$(dirname "$MODEL_PATH"):${PYTHONPATH:-}"
export DOTSOCR_VLLM_AUTO_REGISTER=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PACKAGE_NAME="$(basename "$MODEL_PATH")"
export DOTSOCR_MODEL_PACKAGE_NAME="${MODEL_PACKAGE_NAME}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found. Activate/install the environment that contains vLLM first." >&2
  exit 127
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 127
fi

echo "Starting DotsOCR vLLM server"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  HOST=${HOST}"
echo "  PORT=${PORT}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  PYTHON_BIN=${PYTHON_BIN}"
echo "  MODEL_PACKAGE_NAME=${MODEL_PACKAGE_NAME}"
echo "  DOTSOCR_VLLM_AUTO_REGISTER=${DOTSOCR_VLLM_AUTO_REGISTER}"

"${PYTHON_BIN}" -c '
import importlib
import sys

module_name = sys.argv[1]
sys.argv = ["vllm"] + sys.argv[2:]

importlib.import_module(f"{module_name}.modeling_dots_ocr_vllm")
from vllm.entrypoints.cli.main import main

main()
' "${MODEL_PACKAGE_NAME}" serve "${MODEL_PATH}" \
   --host "${HOST}" \
   --port "${PORT}" \
   --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
   --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
   --chat-template-content-format string \
   --served-model-name "${SERVED_MODEL_NAME}" \
   --trust-remote-code
