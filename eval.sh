#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${PROJECT_ROOT}/code"
EVAL_ROOT="${CODE_ROOT}/evaluation"
OUTPUT_DIR="${QWEN2_5_VL_EVAL_OUTPUT_DIR:-${EVAL_ROOT}/output}"

export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH}"
export QWEN2_5_VL_VARIANT="${QWEN2_5_VL_VARIANT:-origin}"

mkdir -p "${OUTPUT_DIR}"

python "${EVAL_ROOT}/eval.py" \
  --variant "${QWEN2_5_VL_VARIANT}" \
  --model-path "${QWEN2_5_VL_EVAL_MODEL_PATH:-${CODE_ROOT}/qwen-vl-finetune/output}" \
  --data-dir "${QWEN2_5_VL_EVAL_DATA_DIR:-${PROJECT_ROOT}/dataset/test_3_5.jsonl}" \
  --img-root "${QWEN2_5_VL_EVAL_IMG_ROOT:-${PROJECT_ROOT}/dataset/PE-Video/videos/test}" \
  --output-file "${QWEN2_5_VL_EVAL_OUTPUT_FILE:-${OUTPUT_DIR}/${QWEN2_5_VL_VARIANT}_infer.json}" \
  "$@"
