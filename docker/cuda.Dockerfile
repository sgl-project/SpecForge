# syntax=docker/dockerfile:1.7
#
# SpecForge on NVIDIA GPUs (CUDA 13).
#
# The base is the SGLang release whose Torch and Transformers pins equal the
# ones in pyproject.toml, so its virtual environment (/opt/sglang) is reused
# as is. No released image carries the SGLang revision the capture plugin needs,
# so the image's SGLang checkout moves to that revision and is rebuilt in place.
# SpecForge is cloned from GitHub and installs next to it with the capture plugin
# and Mooncake.
#
#   docker build -f docker/cuda.Dockerfile -t specforge:cuda .
#   docker run --rm -it --gpus all --ipc=host specforge:cuda
#
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.23-cu130
FROM ${BASE_IMAGE}

# SGLang revision with the capture extension APIs. It follows SGLANG_COMMIT in
# .github/workflows/test.yaml; tests/test_scripts/test_ci_dependencies.py fails
# when they drift.
ARG SGLANG_COMMIT=c19e9c7d19a873498c4fee1d789596c1a2b6e6f0
# Channel in that revision's rust/rust-toolchain.toml.
ARG RUST_TOOLCHAIN=1.92
ARG FLASHINFER_INDEX_URL=https://flashinfer.ai/whl
ARG FLASHINFER_CUDA=cu130

# uv installs into VIRTUAL_ENV, which the base image sets; stop early on a base
# whose python is not that environment's.
RUN test "$(python -c 'import sys; print(sys.prefix)')" = "${VIRTUAL_ENV:?}"

# Editable like the base image's install, so /sgl-workspace/sglang is the code
# that runs. --no-build-isolation compiles the Rust extensions against the
# installed Torch; the Cargo target directory is a build cache and stays out of
# the image. FlashInfer's prebuilt kernels then follow the new flashinfer-python.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cache/cargo-target \
    cd /sgl-workspace/sglang \
 && git fetch --depth 1 origin ${SGLANG_COMMIT} \
 && git checkout --detach FETCH_HEAD \
 && git clean -ffdxq python rust \
 && rustup toolchain install ${RUST_TOOLCHAIN} --profile minimal \
 && uv pip install setuptools-rust setuptools-scm \
 && RUSTUP_TOOLCHAIN=${RUST_TOOLCHAIN} CARGO_TARGET_DIR=/root/.cache/cargo-target \
    uv pip install --no-build-isolation --prerelease=allow -e python \
 && flashinfer="$(python -c 'from importlib.metadata import version; print(version("flashinfer-python"))')" \
 && uv pip install "flashinfer-cubin==${flashinfer}" --index-url ${FLASHINFER_INDEX_URL} \
 && uv pip install "flashinfer-jit-cache==${flashinfer}" \
      --index-url ${FLASHINFER_INDEX_URL}/${FLASHINFER_CUDA}

# SpecForge is cloned rather than copied from the build context, so the image
# holds a git checkout and the build needs no local files. SPECFORGE_REF takes a
# branch, tag or commit. Docker caches this layer by its text, so a branch stays
# at the commit it had when the layer was built until a --no-cache build.
ARG SPECFORGE_REPO=https://github.com/sgl-project/SpecForge.git
ARG SPECFORGE_REF=main
RUN git clone ${SPECFORGE_REPO} /sgl-workspace/SpecForge \
 && git -C /sgl-workspace/SpecForge checkout ${SPECFORGE_REF}

WORKDIR /sgl-workspace/SpecForge

# SpecForge is editable too: `git pull` in the container, or a checkout mounted
# over this directory, takes effect without a rebuild. The final command fails
# the build if the installs replaced the CUDA Torch, the capture plugin cannot
# find its SGLang APIs, or the cloned SpecForge pins a different SGLang revision
# than the one built above. Mooncake is only located: importing it needs the
# driver's libcuda, absent during a build.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-build-isolation --prerelease=allow -e . \
      mooncake-transfer-engine-cuda13 \
 && uv pip install --no-deps --no-build-isolation ./plugins/sglang-spec-capture \
 && python -c "import importlib.util, torch, specforge.cli, specforge_sglang_capture; \
from sglang.srt.model_executor.forward_observer import ForwardObserver; \
from sglang.srt.managers.deferred_output import ResponseCompletion; \
from specforge.checkpoint.formats.sglang import SGLANG_COMMIT; \
assert SGLANG_COMMIT == '${SGLANG_COMMIT}', SGLANG_COMMIT; \
assert importlib.util.find_spec('mooncake.store'); \
assert torch.version.cuda, torch.__version__"

CMD ["/bin/bash"]
