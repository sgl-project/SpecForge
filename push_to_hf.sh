#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  HF_TOKEN=xxx ./push_to_hf.sh --repo-id <namespace/name> --path <abs_path> [--branch main] [--private true|false] [--repo-type model|dataset|space] [--commit "msg"] [--force true|false]

Examples:
  HF_TOKEN=hf_xxx ./push_to_hf.sh \
    --repo-id yourname/llama3-8b-eagle3-lora-fixed \
    --path /sgl-workspace/SpecForge/outputs/llama3-8b-eagle3-lora-fixed/epoch_0/draft_lora \
    --private true
EOF
}

# defaults
BRANCH="main"
PRIVATE="false"
REPO_TYPE="model"
COMMIT_MSG=""
FORCE="false"

# parse args
REPO_ID=""
SRC_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id) REPO_ID="${2:-}"; shift 2;;
    --path) SRC_PATH="${2:-}"; shift 2;;
    --branch) BRANCH="${2:-}"; shift 2;;
    --private) PRIVATE="${2:-}"; shift 2;;
    --repo-type) REPO_TYPE="${2:-}"; shift 2;;
    --commit) COMMIT_MSG="${2:-}"; shift 2;;
    --force) FORCE="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# validate
: "${HF_TOKEN:?Set HF_TOKEN in env}"
: "${REPO_ID:?--repo-id is required}"
: "${SRC_PATH:?--path is required}"

if [[ ! -d "$SRC_PATH" ]]; then
  echo "Path not found: $SRC_PATH" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git not found. Please install git." >&2
  exit 1
fi
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs not found. Please install git-lfs." >&2
  exit 1
fi
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface CLI not found; attempting to install huggingface_hub..."
  python3 -m pip install --user -U huggingface_hub >/dev/null
fi

# create repo if missing (ignore error if exists)
if command -v hf >/dev/null 2>&1; then
  CREATE_FLAGS=(--repo-type "$REPO_TYPE" --token "$HF_TOKEN")
  [[ "$PRIVATE" == "true" ]] && CREATE_FLAGS+=(--private)
  hf repo create "$REPO_ID" "${CREATE_FLAGS[@]}" -y 2>/dev/null || true
else
  CREATE_FLAGS=(--type "$REPO_TYPE" -y --token "$HF_TOKEN")
  [[ "$PRIVATE" == "true" ]] && CREATE_FLAGS+=(--private)
  huggingface-cli repo create "$REPO_ID" "${CREATE_FLAGS[@]}" 2>/dev/null || true
fi

# commit msg
if [[ -z "$COMMIT_MSG" ]]; then
  COMMIT_MSG="Upload from script on $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
fi

# work in a temp dir to avoid touching source
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# copy content excluding .git
tar -C "$SRC_PATH" --exclude='.git' -cf - . | tar -C "$WORKDIR" -xf -

cd "$WORKDIR"
git init -q
# Some git-lfs versions do not support -q; keep output quiet via redirection
git lfs install --skip-repo >/dev/null 2>&1 || true

# sensible LFS defaults for model assets
git lfs track "*.safetensors" "*.bin" "*.pt" "*.ckpt" "*.h5" "*.gguf" "*.onnx" "*.tflite" "*.tar" "*.zip" 2>/dev/null || true
# track large tokenizer assets to satisfy HF pre-receive hooks (>10 MiB)
git lfs track "tokenizer.json" "tokenizer.model" "spiece.model" "sentencepiece.bpe.model" "*.spm" 2>/dev/null || true
echo ".gitattributes" >> .gitignore || true

git add -A
git commit -m "$COMMIT_MSG" -q

REMOTE="https://oauth2:${HF_TOKEN}@huggingface.co/${REPO_ID}"
git branch -M "$BRANCH"
git remote add origin "$REMOTE"

PUSH_FLAGS=()
[[ "$FORCE" == "true" ]] && PUSH_FLAGS+=("--force")
git push "${PUSH_FLAGS[@]}" -u origin "$BRANCH"

echo "Pushed $SRC_PATH to https://huggingface.co/${REPO_ID} (branch: $BRANCH)"
