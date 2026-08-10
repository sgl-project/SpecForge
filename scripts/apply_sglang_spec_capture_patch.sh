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
# The optional --live layer (patches/sglang/online-live/) adds capture of
# real serving traffic on top of the base v0.5.14 patch. It has its own
# applied-copy record and is applied after / reversed before the base patch.
#
# Usage: scripts/apply_sglang_spec_capture_patch.sh
#          [--target v0.5.14|kimi-k3-ee560a2|kimi-k3-9acd9cb|kimi-k3-f8493a4]
#          [--live] [--reverse]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="v0.5.14"
PATCH_TARGET=""
REVERSE=0
LIVE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --target requires a value" >&2
                exit 2
            fi
            TARGET="$2"
            shift 2
            ;;
        --live)
            LIVE=1
            shift
            ;;
        --reverse)
            REVERSE=1
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ "$LIVE" == 1 && "$TARGET" != "v0.5.14" ]]; then
    echo "ERROR: --live is only supported for the v0.5.14 target" >&2
    exit 2
fi

case "$TARGET" in
    v0.5.14)
        EXPECTED_VERSION_PREFIX="0.5.14"
        PATCH_TARGET="$TARGET"
        ;;
    kimi-k3-ee560a2|kimi-k3-9acd9cb|kimi-k3-f8493a4)
        # Kimi K3's SGLang fork currently reports a base-package version that
        # does not uniquely identify this source revision, so patch --check is
        # the authoritative compatibility gate below.
        EXPECTED_VERSION_PREFIX=""
        # One patch is generated against ee560a2 and compatibility-checked
        # against the original f8493a4 integration point and the 9acd9cb tip.
        # Keep the historical directory name so existing automation remains
        # source-compatible.
        PATCH_TARGET="kimi-k3-f8493a4"
        ;;
    *)
        echo "ERROR: unsupported SGLang patch target: $TARGET" >&2
        exit 2
        ;;
esac
PATCH="$HERE/patches/sglang/$PATCH_TARGET/spec-capture.patch"

SGL_PARENT="$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')"
SGL_VERSION="$(python -c 'import sglang; print(sglang.__version__)')"
APPLIED_COPY="$SGL_PARENT/sglang/.spec_capture_patch.applied"
SINK="$SGL_PARENT/sglang/srt/spec_capture_sink.py"
LIVE_PATCH="$HERE/patches/sglang/online-live/spec-capture-live.patch"
LIVE_APPLIED_COPY="$SGL_PARENT/sglang/.spec_capture_live_patch.applied"

if [[ -n "$EXPECTED_VERSION_PREFIX" && "$SGL_VERSION" != "$EXPECTED_VERSION_PREFIX"* ]]; then
    echo "WARNING: installed sglang is $SGL_VERSION; the patch targets $TARGET" >&2
fi

reverse_live_if_applied() {
    if [[ -f "$LIVE_APPLIED_COPY" ]]; then
        patch --reverse -p2 --batch -d "$SGL_PARENT" < "$LIVE_APPLIED_COPY"
        rm -f "$LIVE_APPLIED_COPY"
        echo "online-live patch --reverse at $SGL_PARENT/sglang"
    fi
}

if [[ "$REVERSE" == 1 ]]; then
    # The live layer sits on top of the base patch: reverse it first.
    reverse_live_if_applied
    if [[ "$LIVE" == 1 ]]; then
        exit 0
    fi
    patch --reverse -p2 --batch -N -d "$SGL_PARENT" < "$PATCH"
    rm -f "$APPLIED_COPY"
    echo "spec-capture patch $TARGET --reverse at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
    exit 0
fi

apply_live_if_requested() {
    if [[ "$LIVE" != 1 ]]; then
        return 0
    fi
    if [[ -f "$LIVE_APPLIED_COPY" ]]; then
        if cmp -s "$LIVE_APPLIED_COPY" "$LIVE_PATCH"; then
            echo "online-live patch already applied at $SGL_PARENT/sglang"
            return 0
        fi
        echo "online-live patch changed; reversing the recorded version first"
        patch --reverse -p2 --batch -d "$SGL_PARENT" < "$LIVE_APPLIED_COPY"
    fi
    patch -p2 --batch -N -d "$SGL_PARENT" < "$LIVE_PATCH"
    cp "$LIVE_PATCH" "$LIVE_APPLIED_COPY"
    echo "online-live patch applied at $SGL_PARENT/sglang"
}

if [[ -f "$APPLIED_COPY" ]]; then
    if cmp -s "$APPLIED_COPY" "$PATCH"; then
        echo "spec-capture patch $TARGET already applied at $SGL_PARENT/sglang"
        apply_live_if_requested
        exit 0
    fi
    echo "spec-capture patch changed; reversing the recorded version first"
    # The live layer sits on top of the base patch: it must come off first.
    reverse_live_if_applied
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
        echo "spec-capture patch $TARGET already applied at $SGL_PARENT/sglang (adopted)"
        exit 0
    fi
    echo "ERROR: $SGL_PARENT/sglang carries an unknown spec-capture patch state" >&2
    echo "reinstall sglang (or clear the cached venv) and re-run this script" >&2
    exit 1
fi

patch -p2 --batch -N -d "$SGL_PARENT" < "$PATCH"
cp "$PATCH" "$APPLIED_COPY"
echo "spec-capture patch $TARGET applied at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
apply_live_if_requested
