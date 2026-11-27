#!/usr/bin/env bash
# Run in the CI venv after all project dependencies have been installed.
set -euo pipefail

: "${SPECFORGE_CI_CACHE:?}"
: "${FLASH_ATTN_VERSION:?}"

check_flash_attention() {
    python - <<'PY'
import os
from importlib.metadata import distributions

installed = list(distributions(name="flash-attn"))
assert len(installed) == 1, "Expected one FlashAttention installation"
assert installed[0].version == os.environ["FLASH_ATTN_VERSION"]
from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

lazy_import_flash_attention("flash_attention_2")
PY
}

if check_flash_attention; then
    echo "Reusing installed FlashAttention; ABI check passed."
    exit 0
fi

# uv does not normally key source-built wheels by torch's ABI. Keep this cache
# separate from ordinary pip installs and from builds for other environments.
build_key=$(python - <<'PY'
import hashlib
import os
import platform
import sys

import torch

parts = (
    sys.implementation.cache_tag, platform.machine(), sys.platform,
    torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI,
    os.environ["FLASH_ATTN_VERSION"],
)
print(hashlib.sha256(repr(parts).encode()).hexdigest())
PY
)
export UV_CACHE_DIR="$SPECFORGE_CI_CACHE/flash-attn/$build_key"

# Uninstall first: --reinstall-package also refreshes uv's package cache.
# Reuse a wheel built here, or compile once against the final torch. Never let
# FA's setup.py download a wheel built against a different torch patch version.
#
# torch 2.14's headers require C++20 (std::strong_ordering in c10), while
# flash-attn 2.8.3's setup.py still appends -std=c++17 after torch's default
# flags, so a plain source build now fails. Patch the flag and cache the built
# wheel per build_key; uv never caches builds from local sources.
#
# Only a wheel that installed and passed the ABI check is published to the
# cache, by an atomic rename, so an interrupted build or a broken wheel cannot
# be reused by later runs. A cached wheel that fails is dropped to force a
# rebuild next time.
wheel_dir="$SPECFORGE_CI_CACHE/flash-attn/$build_key/wheels"
wheel=$(ls "$wheel_dir"/flash_attn-*.whl 2>/dev/null | head -1 || true)
built=
if [ -z "$wheel" ]; then
    src=$(mktemp -d)
    trap 'rm -rf "$src"' EXIT
    sdist=$(curl -sSfL "https://pypi.org/pypi/flash-attn/$FLASH_ATTN_VERSION/json" |
        python -c 'import json, sys; u = next(u for u in json.load(sys.stdin)["urls"] if u["packagetype"] == "sdist"); print(u["url"], u["digests"]["sha256"])')
    sdist_url=${sdist% *}
    sdist_sha256=${sdist#* }
    curl -sSfL "$sdist_url" -o "$src/sdist.tar.gz"
    echo "$sdist_sha256  $src/sdist.tar.gz" | sha256sum -c -
    tar xzf "$src/sdist.tar.gz" -C "$src"
    sed -i 's/-std=c++17/-std=c++20/g' "$src"/flash_attn-*/setup.py
    (cd "$src"/flash_attn-* && FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 python setup.py bdist_wheel -d "$src/dist")
    wheel=$(ls "$src"/dist/flash_attn-*.whl | head -1)
    built=1
fi
uv pip uninstall flash-attn
if ! { uv pip install --no-deps "$wheel" && check_flash_attention; }; then
    [ -n "$built" ] || rm -f "$wheel"
    exit 1
fi
if [ -n "$built" ]; then
    mkdir -p "$wheel_dir"
    staged=$(mktemp "$wheel_dir/.staging.XXXXXX")
    cp "$wheel" "$staged"
    mv "$staged" "$wheel_dir/$(basename "$wheel")"
fi
