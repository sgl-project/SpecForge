#!/usr/bin/env bash
# Apply the spec-capture patch to the INSTALLED sglang (site-packages).
#
# The patch file is authored against the sglang source tree (paths like
# python/sglang/srt/...); an installed package drops the python/ prefix, so we
# strip one component and apply from the site-packages parent.
#
# Usage: scripts/apply_sglang_spec_capture_patch.sh [--reverse]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$HERE/patches/sglang/v0.5.14/spec-capture.patch"

SGL_PARENT="$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')"
SGL_VERSION="$(python -c 'import sglang; print(sglang.__version__)')"

if [[ "$SGL_VERSION" != 0.5.14* ]]; then
    echo "WARNING: installed sglang is $SGL_VERSION; the patch targets v0.5.14" >&2
fi

REVERSE=()
if [[ "${1:-}" == "--reverse" ]]; then
    REVERSE=(--reverse)
fi

# Idempotence: skip when already applied (forward mode only).
if [[ ${#REVERSE[@]} -eq 0 ]] && [[ -f "$SGL_PARENT/sglang/srt/spec_capture_sink.py" ]]; then
    echo "spec-capture patch already applied at $SGL_PARENT/sglang"
    exit 0
fi

patch "${REVERSE[@]}" -p1 -d "$SGL_PARENT" < "$PATCH"
echo "spec-capture patch ${REVERSE[0]:-applied} at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
