#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$ROOT/patches/sglang/v0.5.18/spec-capture-target-kv.patch"
SOURCE="$ROOT/patches/sglang/v0.5.18/target_kv_capture.py"
PARENT="$(python -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).resolve().parent.parent)')"
python -c 'import sglang; assert sglang.__version__.startswith("0.5.18"), sglang.__version__'
test -f "$PARENT/sglang/srt/spec_capture_sink.py"
export GIT_CEILING_DIRECTORIES="$(dirname "$PARENT")"
if git -C "$PARENT" apply -p2 --reverse --check "$PATCH" 2>/dev/null; then
    echo "Target-KV patch already applied"
else
    git -C "$PARENT" apply -p2 --check "$PATCH"
    git -C "$PARENT" apply -p2 "$PATCH"
fi
cp "$SOURCE" "$PARENT/sglang/srt/target_kv_capture.py"
python -m compileall -q "$PARENT/sglang/srt/target_kv_capture.py"
