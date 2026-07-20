#!/usr/bin/env bash
# Apply the spec-capture patch to the INSTALLED sglang (site-packages).
#
# The patch file is authored against the sglang source tree (git-style paths
# a/python/sglang/srt/...); an installed package drops the python/ prefix, so
# strip TWO components (a/ + python/) and apply from the site-packages parent.
#
# Idempotence is CONTENT-aware, not existence-based: the applied patch text is
# recorded next to the tree, so a revised patch reverses the recorded one and
# re-applies instead of silently keeping a stale version alive on cached
# venvs/runners. A tree patched before this record existed is adopted only
# when a reverse dry-run proves it matches the current patch byte-for-byte;
# anything else fails loudly rather than testing against unknown server code.
#
# Usage: scripts/apply_sglang_spec_capture_patch.sh [--reverse]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$HERE/patches/sglang/v0.5.14/spec-capture.patch"

SGL_PARENT="$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')"
SGL_VERSION="$(python -c 'import sglang; print(sglang.__version__)')"
APPLIED_COPY="$SGL_PARENT/sglang/.spec_capture_patch.applied"
SINK="$SGL_PARENT/sglang/srt/spec_capture_sink.py"

if [[ "$SGL_VERSION" != 0.5.14* ]]; then
    echo "WARNING: installed sglang is $SGL_VERSION; the patch targets v0.5.14" >&2
fi

if [[ "${1:-}" == "--reverse" ]]; then
    patch --reverse -p2 --batch -N -d "$SGL_PARENT" < "$PATCH"
    rm -f "$APPLIED_COPY"
    echo "spec-capture patch --reverse at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
    exit 0
fi

if [[ -f "$APPLIED_COPY" ]]; then
    if cmp -s "$APPLIED_COPY" "$PATCH"; then
        echo "spec-capture patch already applied at $SGL_PARENT/sglang"
        exit 0
    fi
    echo "spec-capture patch changed; reversing the recorded version first"
    patch --reverse -p2 --batch -d "$SGL_PARENT" < "$APPLIED_COPY"
elif [[ -f "$SINK" ]]; then
    # Patched before the applied-copy record existed. Adopt only a tree that
    # provably matches the current patch; otherwise demand a clean reinstall.
    # git apply verifies exact content on reverse; BSD patch dry-runs do not.
    if command -v git > /dev/null; then
        matches() { git -C "$SGL_PARENT" apply --reverse --check -p2 "$PATCH" 2> /dev/null; }
    else
        matches() { patch --reverse --dry-run -p2 --batch -d "$SGL_PARENT" < "$PATCH" > /dev/null; }
    fi
    if matches; then
        cp "$PATCH" "$APPLIED_COPY"
        echo "spec-capture patch already applied at $SGL_PARENT/sglang (adopted)"
        exit 0
    fi
    echo "ERROR: $SGL_PARENT/sglang carries an unknown spec-capture patch state" >&2
    echo "reinstall sglang (or clear the cached venv) and re-run this script" >&2
    exit 1
fi

patch -p2 --batch -N -d "$SGL_PARENT" < "$PATCH"
cp "$PATCH" "$APPLIED_COPY"
echo "spec-capture patch applied at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
