#!/usr/bin/env bash
set -euo pipefail

# Download the exact Kimi-K3 Open Perfect Blend regeneration used by the K3
# draft recipes.  The token is read from a protected file and is never printed.

REPO_ID=${REPO_ID:-skx618/Kimi-K3-OpenPerfectBlend-Regen}
REVISION=${REVISION:-439c2fdc9fd2ae92e194bde468d26867b36dd660}
FILENAME=${FILENAME:-data.jsonl}
EXPECTED_SHA256=${EXPECTED_SHA256:-5418f09d1af8ec2e08e8385799f1eeb3c062c669b28407870f7737007bc3eeb9}
EXPECTED_ROWS=${EXPECTED_ROWS:-698316}
OUTPUT=${OUTPUT:-/workspace/k3_dspark/data/kimi-k3-openperfectblend-regen-439c2fdc/data.jsonl}
SMOKE_OUTPUT=${SMOKE_OUTPUT:-/workspace/k3_dspark/data/kimi-k3-openperfectblend-smoke.jsonl}
HF_TOKEN_FILE=${HF_TOKEN_FILE:-/workspace/k3_dspark/secrets/hf_token}

if [[ -z "${HF_TOKEN:-}" && -r "$HF_TOKEN_FILE" ]]; then
    IFS= read -r HF_TOKEN < "$HF_TOKEN_FILE" || [[ -n "$HF_TOKEN" ]]
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    printf 'ERROR: set HF_TOKEN or create mode-600 %s\n' "$HF_TOKEN_FILE" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
downloaded=$(
    HF_TOKEN="$HF_TOKEN" python3 - "$REPO_ID" "$REVISION" "$FILENAME" <<'PY'
import os
import sys

from huggingface_hub import hf_hub_download

print(
    hf_hub_download(
        repo_id=sys.argv[1],
        repo_type="dataset",
        revision=sys.argv[2],
        filename=sys.argv[3],
        token=os.environ["HF_TOKEN"],
    )
)
PY
)
install -m 0644 "$downloaded" "$OUTPUT"

actual_sha256=$(sha256sum "$OUTPUT" | awk '{print $1}')
actual_rows=$(wc -l < "$OUTPUT" | tr -d ' ')
if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    printf 'ERROR: dataset SHA-256 mismatch: %s\n' "$actual_sha256" >&2
    exit 1
fi
if [[ "$actual_rows" != "$EXPECTED_ROWS" ]]; then
    printf 'ERROR: dataset row count mismatch: %s\n' "$actual_rows" >&2
    exit 1
fi
sed -n '1,4p' "$OUTPUT" > "$SMOKE_OUTPUT"
[[ "$(wc -l < "$SMOKE_OUTPUT" | tr -d ' ')" == 4 ]] || {
    printf 'ERROR: failed to create four-row smoke fixture\n' >&2
    exit 1
}
printf 'validated Kimi-K3 Open Perfect Blend: rows=%s sha256=%s path=%s\n' \
    "$actual_rows" "$actual_sha256" "$OUTPUT"
printf 'created four-row smoke fixture: %s\n' "$SMOKE_OUTPUT"
