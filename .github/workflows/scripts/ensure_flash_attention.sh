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
uv pip uninstall flash-attn
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 uv pip install -v \
    --no-binary flash-attn --no-build-isolation --no-deps \
    "flash-attn==$FLASH_ATTN_VERSION"
check_flash_attention
