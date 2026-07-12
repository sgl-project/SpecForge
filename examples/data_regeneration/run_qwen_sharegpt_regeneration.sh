#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PROFILE="${MODEL_PROFILE:-qwen3-8b}"
case "${MODEL_PROFILE}" in
    qwen3-8b)
        DEFAULT_MODEL_PATH="Qwen/Qwen3-8B"
        DEFAULT_INPUT_FILE="${ROOT_DIR}/cache/dataset/sharegpt_train.jsonl"
        DEFAULT_OUTPUT_FILE="${ROOT_DIR}/cache/dataset/sharegpt_train_regen_qwen3_8b_temperature0_non_reasoning.jsonl"
        REASONING_MODE=disable
        DEFAULT_MAX_TOKENS=4096
        VALIDATION_MODE=(--expect-non-reasoning)
        ;;
    qwen3.6-27b)
        DEFAULT_MODEL_PATH="Qwen/Qwen3.6-27B"
        DEFAULT_INPUT_FILE="${ROOT_DIR}/cache/dataset/sharegpt_train.jsonl"
        DEFAULT_OUTPUT_FILE="${ROOT_DIR}/cache/dataset/sharegpt_train_regen_qwen3.6-27b_temperature0_reasoning.jsonl"
        REASONING_MODE=save
        DEFAULT_MAX_TOKENS=32768
        VALIDATION_MODE=(--expect-reasoning)
        ;;
    *)
        echo "Unsupported MODEL_PROFILE: ${MODEL_PROFILE}" >&2
        echo "Expected qwen3-8b or qwen3.6-27b." >&2
        exit 1
        ;;
esac

PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
INPUT_FILE="${INPUT_FILE:-${DEFAULT_INPUT_FILE}}"
OUTPUT_FILE="${OUTPUT_FILE:-${DEFAULT_OUTPUT_FILE}}"
SERVER_ADDRESSES="${SERVER_ADDRESSES:-localhost:30000}"
CONCURRENCY="${CONCURRENCY:-64}"
MAX_TOKENS="${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}"

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "Input dataset does not exist: ${INPUT_FILE}" >&2
    exit 1
fi
if [[ "${OUTPUT_FILE}" != *.jsonl ]]; then
    echo "OUTPUT_FILE must end in .jsonl: ${OUTPUT_FILE}" >&2
    exit 1
fi

ERROR_FILE="${OUTPUT_FILE%.jsonl}_error.jsonl"
SKIPPED_FILE="${OUTPUT_FILE%.jsonl}_skipped.jsonl"
for path in "${OUTPUT_FILE}" "${ERROR_FILE}" "${SKIPPED_FILE}"; do
    if [[ -e "${path}" ]]; then
        echo "Refusing to reuse an existing regeneration output: ${path}" >&2
        echo "Choose a fresh OUTPUT_FILE or remove the old run explicitly." >&2
        exit 1
    fi
done

mkdir -p "$(dirname "${OUTPUT_FILE}")"
read -r -a server_args <<< "${SERVER_ADDRESSES}"

"${PYTHON}" scripts/regenerate_train_data.py \
    --model "${MODEL_PATH}" \
    --reasoning "${REASONING_MODE}" \
    --temperature 0 \
    --concurrency "${CONCURRENCY}" \
    --max-tokens "${MAX_TOKENS}" \
    --server-address "${server_args[@]}" \
    --input-file-path "${INPUT_FILE}" \
    --output-file-path "${OUTPUT_FILE}"

INPUT_ROWS=$(awk 'END {print NR}' "${INPUT_FILE}")
SUCCESS_ROWS=$(awk 'END {print NR}' "${OUTPUT_FILE}")
ERROR_ROWS=$(awk 'END {print NR}' "${ERROR_FILE}")
SKIPPED_ROWS=$(awk 'END {print NR}' "${SKIPPED_FILE}")

"${PYTHON}" - \
    "${INPUT_ROWS}" "${SUCCESS_ROWS}" "${ERROR_ROWS}" "${SKIPPED_ROWS}" <<'PY'
import sys

input_rows, success_rows, error_rows, skipped_rows = map(int, sys.argv[1:5])
completed_rows = success_rows + error_rows + skipped_rows

print(f"input rows: {input_rows}")
print(f"success rows: {success_rows}")
print(f"error rows: {error_rows}")
print(f"skipped rows: {skipped_rows}")
if completed_rows != input_rows:
    raise SystemExit(
        "regeneration did not account for every input row: "
        f"completed={completed_rows}, input={input_rows}"
    )
success_fraction = success_rows / completed_rows if completed_rows else 0.0
print(f"success fraction: {success_fraction:.2%}")
PY

"${PYTHON}" scripts/validate_regenerated_data.py \
    --data-path "${OUTPUT_FILE}" \
    "${VALIDATION_MODE[@]}" \
    --strict-think-markers
