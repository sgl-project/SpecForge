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
# venvs/runners. All checks and mutations use git apply, which is exact and
# atomic. This also lets us safely recognize a pip-upgrade cache state: pip
# replaces package-owned files but leaves our added sink and patch record.
#
# Usage: scripts/apply_sglang_spec_capture_patch.sh
#          [--target v0.5.18|kimi-k3-ee560a2|kimi-k3-9acd9cb|kimi-k3-f8493a4]
#          [--reverse]
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="v0.5.18"
PATCH_TARGET=""
REVERSE=0
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

case "$TARGET" in
    v0.5.18)
        EXPECTED_VERSION_PREFIX="0.5.18"
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
PATCH="${SPECFORGE_SPEC_CAPTURE_PATCH:-$HERE/patches/sglang/$PATCH_TARGET/spec-capture.patch}"

SGL_PARENT="${SPECFORGE_SGLANG_ROOT:-$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')}"
SGL_VERSION="${SPECFORGE_SGLANG_VERSION:-$(python -c 'import sglang; print(sglang.__version__)')}"
APPLIED_COPY="$SGL_PARENT/sglang/.spec_capture_patch.applied"
SINK="$SGL_PARENT/sglang/srt/spec_capture_sink.py"

if ! command -v git > /dev/null; then
    echo "ERROR: git is required to verify and apply the spec-capture patch exactly" >&2
    exit 1
fi

check_apply() {
    git -C "$SGL_PARENT" apply --check -p2 "$1" 2> /dev/null
}

check_reverse() {
    git -C "$SGL_PARENT" apply --reverse --check -p2 "$1" 2> /dev/null
}

apply_exact() {
    git -C "$SGL_PARENT" apply -p2 "$1"
}

reverse_exact() {
    git -C "$SGL_PARENT" apply --reverse -p2 "$1"
}

fail_unknown_state() {
    echo "ERROR: $SGL_PARENT/sglang carries an unknown spec-capture patch state" >&2
    echo "reinstall sglang (or clear the cached venv) and re-run this script" >&2
    exit 1
}

# A cached venv may retain files added by our old patch after pip upgrades the
# package-owned files. Move the known added file aside, then require the new
# patch to apply exactly before discarding the stale residue. Restore it on any
# failed check so this recovery path never leaves a partial mutation behind.
recover_pip_upgrade_cache() {
    if [[ -z "$EXPECTED_VERSION_PREFIX" || "$SGL_VERSION" != "$EXPECTED_VERSION_PREFIX"* ]]; then
        return 1
    fi

    local stale_sink=""
    if [[ -f "$SINK" ]]; then
        stale_sink="$SINK.specforge-stale.$$"
        if [[ -e "$stale_sink" ]]; then
            return 1
        fi
        mv "$SINK" "$stale_sink"
    fi

    if check_apply "$PATCH"; then
        [[ -z "$stale_sink" ]] || rm -f "$stale_sink"
        rm -f "$APPLIED_COPY"
        echo "recovered stale spec-capture files left by a cached pip upgrade"
        return 0
    fi

    if [[ -n "$stale_sink" && -f "$stale_sink" ]]; then
        mv "$stale_sink" "$SINK"
    fi
    return 1
}

if [[ -n "$EXPECTED_VERSION_PREFIX" && "$SGL_VERSION" != "$EXPECTED_VERSION_PREFIX"* ]]; then
    echo "WARNING: installed sglang is $SGL_VERSION; the patch targets $TARGET" >&2
fi

if [[ "$REVERSE" == 1 ]]; then
    if ! check_reverse "$PATCH"; then
        fail_unknown_state
    fi
    reverse_exact "$PATCH"
    rm -f "$APPLIED_COPY"
    echo "spec-capture patch $TARGET --reverse at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
    exit 0
fi

if [[ -f "$APPLIED_COPY" ]]; then
    if cmp -s "$APPLIED_COPY" "$PATCH" && check_reverse "$PATCH"; then
        echo "spec-capture patch $TARGET already applied at $SGL_PARENT/sglang"
        exit 0
    fi
    if check_reverse "$APPLIED_COPY"; then
        echo "spec-capture patch changed; reversing the recorded version first"
        reverse_exact "$APPLIED_COPY"
        rm -f "$APPLIED_COPY"
    elif ! recover_pip_upgrade_cache; then
        fail_unknown_state
    fi
elif [[ -f "$SINK" ]]; then
    # Patched before the applied-copy record existed. Adopt only a tree that
    # provably matches the current patch; otherwise demand a clean reinstall.
    if check_reverse "$PATCH"; then
        cp "$PATCH" "$APPLIED_COPY"
        echo "spec-capture patch $TARGET already applied at $SGL_PARENT/sglang (adopted)"
        exit 0
    fi
    fail_unknown_state
fi

if ! check_apply "$PATCH"; then
    fail_unknown_state
fi
apply_exact "$PATCH"
cp "$PATCH" "$APPLIED_COPY"
echo "spec-capture patch $TARGET applied at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
