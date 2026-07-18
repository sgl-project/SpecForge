#!/usr/bin/env bash
# Strictly apply, verify, or reverse the server-capture patch in one isolated
# sglang==0.5.14 installation. The Python executable is mandatory so an active
# shell environment can never redirect this operation to another site-packages.
set -euo pipefail

readonly EXPECTED_PATCH_SHA256="a07c015a584bfff053d1358b1cc3c28ee79f1983ed8298953732c5ebe40647fb"
readonly EXPECTED_V1_TO_V2_SHA256="a41469afa9355a7384f1cd9d700836a9ecc8dc13c8488d73e867730f40ea6c59"
readonly EXPECTED_V2_TO_V3_SHA256="4395b9a515d2fcd4ae101bf6dd2d8e960b3786a0a4e8059aa12113a8c182496a"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PATCH_PATH="$REPO_ROOT/patches/sglang/v0.5.14/spec-capture.patch"
readonly V1_TO_V2_PATH="$REPO_ROOT/patches/sglang/v0.5.14/spec-capture-v1-to-v2.patch"
readonly V2_TO_V3_PATH="$REPO_ROOT/patches/sglang/v0.5.14/spec-capture-v2-to-v3.patch"

usage() {
    cat >&2 <<'EOF'
Usage:
  scripts/apply_sglang_spec_capture_patch.sh --python /path/to/python [--apply]
  scripts/apply_sglang_spec_capture_patch.sh --python /path/to/python --check
  scripts/apply_sglang_spec_capture_patch.sh --python /path/to/python --reverse

Modes are idempotent. --apply upgrades exact supported prior capture-patch
revisions when needed. --check performs no writes and succeeds only when the
exact current v0.5.14 patch is applied and its imports and launch-server flags
work.
EOF
}

die() {
    echo "spec-capture patch error: $*" >&2
    exit 2
}

mode="apply"
python_executable=""
mode_seen=0
while (($#)); do
    case "$1" in
        --python)
            (($# >= 2)) || die "--python requires an executable"
            python_executable="$2"
            shift 2
            ;;
        --apply|--check|--reverse)
            ((mode_seen == 0)) || die "choose exactly one mode"
            mode="${1#--}"
            mode_seen=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            die "unknown argument $1"
            ;;
    esac
done

[[ -n "$python_executable" ]] || {
    usage
    die "--python is required"
}
if [[ "$python_executable" != /* ]]; then
    python_executable="$(command -v -- "$python_executable" || true)"
fi
[[ -n "$python_executable" && -x "$python_executable" ]] \
    || die "Python executable is unavailable"
[[ -f "$PATCH_PATH" && ! -L "$PATCH_PATH" ]] || die "patch file is unavailable"
for migration_path in "$V1_TO_V2_PATH" "$V2_TO_V3_PATH"; do
    [[ -f "$migration_path" && ! -L "$migration_path" ]] \
        || die "patch migration file is unavailable: $migration_path"
done

actual_patch_sha256="$(sha256sum "$PATCH_PATH" | awk '{print $1}')"
[[ "$actual_patch_sha256" == "$EXPECTED_PATCH_SHA256" ]] || die \
    "patch SHA256 mismatch: expected $EXPECTED_PATCH_SHA256, got $actual_patch_sha256"
actual_v1_to_v2_sha256="$(sha256sum "$V1_TO_V2_PATH" | awk '{print $1}')"
[[ "$actual_v1_to_v2_sha256" == "$EXPECTED_V1_TO_V2_SHA256" ]] || die \
    "v1-to-v2 SHA256 mismatch: expected $EXPECTED_V1_TO_V2_SHA256, got $actual_v1_to_v2_sha256"
actual_v2_to_v3_sha256="$(sha256sum "$V2_TO_V3_PATH" | awk '{print $1}')"
[[ "$actual_v2_to_v3_sha256" == "$EXPECTED_V2_TO_V3_SHA256" ]] || die \
    "v2-to-v3 SHA256 mismatch: expected $EXPECTED_V2_TO_V3_SHA256, got $actual_v2_to_v3_sha256"

mapfile -t package_info < <("$python_executable" - <<'PY'
import importlib.metadata
import importlib.util
import os
import pathlib
import sys
import sysconfig

try:
    version = importlib.metadata.version("sglang")
except importlib.metadata.PackageNotFoundError as exc:
    raise SystemExit(f"sglang distribution is missing: {exc}")
spec = importlib.util.find_spec("sglang")
if spec is None or spec.origin is None:
    raise SystemExit("sglang package is not importable")
root = pathlib.Path(spec.origin).resolve().parent
if root.name != "sglang" or not root.is_dir():
    raise SystemExit(f"unexpected sglang package root: {root}")
site_roots = {
    pathlib.Path(value).resolve()
    for key, value in sysconfig.get_paths().items()
    if key in {"purelib", "platlib"} and value
}
if not any(root == site / "sglang" for site in site_roots):
    raise SystemExit(
        f"refusing to patch non-installed source tree {root}; "
        f"expected one of {[str(site / 'sglang') for site in site_roots]}"
    )
print(version)
print(root)
print(root.parent)
print(pathlib.Path(sys.executable).resolve())
PY
) || die "cannot inspect sglang through $python_executable"

((${#package_info[@]} == 4)) || die "incomplete sglang package inspection"
sglang_version="${package_info[0]}"
sglang_root="${package_info[1]}"
sglang_parent="${package_info[2]}"
python_realpath="${package_info[3]}"
[[ "$sglang_version" == "0.5.14" ]] \
    || die "sglang must be exactly 0.5.14, got $sglang_version"

mapfile -t patch_targets < <(
    sed -n 's#^+++ b/python/##p' "$PATCH_PATH" | LC_ALL=C sort -u
)
((${#patch_targets[@]} == 12)) \
    || die "patch inventory must contain exactly 12 target files"
for target in "${patch_targets[@]}"; do
    [[ "$target" == sglang/srt/* && "$target" != *".."* ]] \
        || die "unsafe patch target $target"
done
mapfile -t v1_to_v2_targets < <(
    sed -n 's#^+++ b/python/##p' "$V1_TO_V2_PATH" | LC_ALL=C sort -u
)
[[ "${v1_to_v2_targets[*]}" == \
    "sglang/srt/managers/scheduler_components/batch_result_processor.py" ]] \
    || die "v1-to-v2 migration has an unexpected target inventory"
mapfile -t v2_to_v3_targets < <(
    sed -n 's#^+++ b/python/##p' "$V2_TO_V3_PATH" | LC_ALL=C sort -u
)
[[ "${v2_to_v3_targets[*]}" == "sglang/srt/spec_capture_sink.py" ]] \
    || die "v2-to-v3 migration has an unexpected target inventory"

patch_probe() {
    local direction="$1"
    local output_file="$2"
    local patch_path="${3:-$PATCH_PATH}"
    local target_root="${4:-$sglang_parent}"
    local -a args=(--dry-run --batch --silent -p2 -d "$target_root")
    if [[ "$direction" == "reverse" ]]; then
        args+=(--reverse)
    else
        args+=(--forward)
    fi
    if ! patch "${args[@]}" < "$patch_path" >"$output_file" 2>&1; then
        return 1
    fi
    # GNU patch can return zero in --batch mode after ignoring every hunk in
    # the wrong direction. Treat any skip/ignore diagnostic as a failed probe.
    if grep -Eqi \
        '(^|[[:space:]])(ignoring|skipping)([[:space:]]|$)|FAILED|does not exist|malformed patch' \
        "$output_file"; then
        return 1
    fi
}

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/spec-capture-patch.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

apply_probe_migration() {
    local patch_path="$1"
    local probe_root="$2"
    local output_file="$3"
    if ! patch --batch --forward -p2 -d "$probe_root" \
        < "$patch_path" >"$output_file" 2>&1; then
        return 1
    fi
    ! grep -Eqi \
        '(^|[[:space:]])(ignoring|skipping)([[:space:]]|$)|FAILED|does not exist|malformed patch' \
        "$output_file"
}

probe_legacy_upgrade() {
    local source_version="$1"
    local probe_root="$tmpdir/legacy-v${source_version}"
    local target source destination
    for target in "${patch_targets[@]}"; do
        source="$sglang_parent/$target"
        destination="$probe_root/$target"
        [[ -f "$source" && ! -L "$source" ]] || return 1
        mkdir -p "$(dirname "$destination")"
        cp -- "$source" "$destination"
    done
    if ((source_version == 1)); then
        apply_probe_migration \
            "$V1_TO_V2_PATH" "$probe_root" "$tmpdir/legacy-v1-to-v2.log" \
            || return 1
    fi
    apply_probe_migration \
        "$V2_TO_V3_PATH" "$probe_root" "$tmpdir/legacy-v2-to-v3.log" \
        || return 1
    patch_probe \
        reverse "$tmpdir/legacy-current-v${source_version}.log" \
        "$PATCH_PATH" "$probe_root"
}

forward_ok=0
reverse_ok=0
legacy_version=0
patch_probe forward "$tmpdir/forward.log" && forward_ok=1
patch_probe reverse "$tmpdir/reverse.log" && reverse_ok=1
if ((forward_ok == 0 && reverse_ok == 0)); then
    probe_legacy_upgrade 2 && legacy_version=2
    if ((legacy_version == 0)); then
        probe_legacy_upgrade 1 && legacy_version=1
    fi
fi

if ((forward_ok == reverse_ok && legacy_version == 0)); then
    echo "forward dry-run:" >&2
    sed -n '1,80p' "$tmpdir/forward.log" >&2
    echo "reverse dry-run:" >&2
    sed -n '1,80p' "$tmpdir/reverse.log" >&2
    die "target tree is neither a clean v0.5.14 tree nor the exact patched state"
fi

verify_patched_tree() {
    SGLANG_CAPTURE_ROOT="$sglang_root" "$python_executable" - <<'PY'
import hashlib
import importlib
import os
from pathlib import Path

root = Path(os.environ["SGLANG_CAPTURE_ROOT"])
required = {
    "srt/spec_capture_sink.py": (
        "class SpecCaptureSink",
        "def get_sink",
        "def maybe_init_sink",
    ),
    "srt/server_args.py": (
        "enable_spec_capture",
        "spec_capture_aux_layer_ids",
        "spec_capture_method",
    ),
    "srt/managers/scheduler.py": ("spec_capture_sink.maybe_init_sink",),
    "srt/managers/schedule_batch.py": (
        "self.return_hidden_states = return_hidden_states",
        "req.return_hidden_states or req.spec_capture is not None",
    ),
    "srt/model_executor/model_runner.py": ("spec_capture_method",),
    "srt/models/qwen2.py": ("Capture after the final transformer layer",),
    "srt/managers/scheduler_components/batch_result_processor.py": (
        "_sink_spec_capture",
        "requested_artifacts = req.spec_capture.get",
    ),
}
for relative, markers in required.items():
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or unsafe patched module: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise SystemExit(f"{path} lacks patch markers {missing}")
    print(f"{relative}\t{hashlib.sha256(path.read_bytes()).hexdigest()}")
importlib.import_module("sglang.srt.spec_capture_sink")
importlib.import_module("sglang.srt.server_args")
PY

    CUDA_VISIBLE_DEVICES='' FLASHINFER_DISABLE_VERSION_CHECK=1 \
        "$python_executable" -m sglang.launch_server --help \
        >"$tmpdir/launch-help.txt" 2>"$tmpdir/launch-help.err" || {
            sed -n '1,120p' "$tmpdir/launch-help.err" >&2
            die "patched sglang launch-server CLI is not importable"
        }
    local flag
    for flag in \
        --enable-spec-capture \
        --spec-capture-method \
        --spec-capture-aux-layer-ids \
        --spec-capture-store-id \
        --spec-capture-max-sample-bytes \
        --spec-capture-inventory-db \
        --spec-capture-lifecycle-db \
        --disable-radix-cache \
        --chunked-prefill-size; do
        grep -Fq -- "$flag" "$tmpdir/launch-help.txt" \
            || die "patched launch-server help lacks $flag"
    done
}

upgrade_legacy_tree() {
    local migration_path migration_name migration_log
    local -a migrations=("$V2_TO_V3_PATH")
    if ((legacy_version == 1)); then
        migrations=("$V1_TO_V2_PATH" "${migrations[@]}")
    fi
    for migration_path in "${migrations[@]}"; do
        migration_name="$(basename "$migration_path" .patch)"
        migration_log="$tmpdir/${migration_name}-apply.log"
        if ! patch --batch --forward -p2 -d "$sglang_parent" \
            < "$migration_path" >"$migration_log" 2>&1; then
            sed -n '1,80p' "$migration_log" >&2
            die "$migration_name migration failed"
        fi
    done
    patch_probe reverse "$tmpdir/post-migration.log" \
        || die "legacy migration did not produce the exact current patch state"
    forward_ok=0
    reverse_ok=1
    legacy_version=0
}

case "$mode" in
    check)
        ((legacy_version == 0)) \
            || die "spec-capture patch v$legacy_version is applied; run --apply to upgrade it"
        ((reverse_ok == 1)) || die "spec-capture patch is not applied"
        verify_patched_tree
        echo "verified spec-capture patch $EXPECTED_PATCH_SHA256 in $sglang_root"
        ;;
    apply)
        if ((legacy_version > 0)); then
            upgrade_legacy_tree
        elif ((forward_ok == 1)); then
            patch --batch --forward -p2 -d "$sglang_parent" \
                < "$PATCH_PATH" >"$tmpdir/apply.log"
            patch_probe reverse "$tmpdir/post-apply.log" \
                || die "post-apply reverse dry-run failed"
        fi
        verify_patched_tree
        echo "applied and verified spec-capture patch $EXPECTED_PATCH_SHA256"
        echo "python=$python_realpath sglang=$sglang_version root=$sglang_root"
        ;;
    reverse)
        if ((legacy_version > 0)); then
            upgrade_legacy_tree
        fi
        if ((reverse_ok == 1)); then
            patch --batch --reverse -p2 -d "$sglang_parent" \
                < "$PATCH_PATH" >"$tmpdir/reverse-apply.log"
            patch_probe forward "$tmpdir/post-reverse.log" \
                || die "post-reverse forward dry-run failed"
        fi
        echo "verified clean unpatched sglang $sglang_version at $sglang_root"
        ;;
esac
