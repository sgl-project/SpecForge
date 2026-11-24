# syntax=docker/dockerfile:1.7
#
# SpecForge on Google Cloud TPU (tpu7x) via torch_tpu.
# Layers SpecForge and a CUDA-free Mooncake onto the sglang-torchtpu image, which
# already carries torch_tpu, sglang and the sglang_tpu plugin.
#
ARG BASE_IMAGE=us-central1-docker.pkg.dev/cloud-tpu-inference-test/junyanxu-vllm-tpu/sglang-torchtpu:latest
FROM ${BASE_IMAGE}

ENV SPECFORGE_DEVICE=tpu

# The stock mooncake-transfer-engine wheel links libcuda; the non-CUDA build
# still needs the RDMA userspace libraries even on TCP.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libibverbs1 librdmacm1 libnuma1 \
 && rm -rf /var/lib/apt/lists/* \
 && pip uninstall -y mooncake-transfer-engine
RUN pip install --no-cache-dir mooncake-transfer-engine-non-cuda==0.3.13.post1 openai

WORKDIR /opt/specforge
COPY pyproject.toml version.txt README.md LICENSE MANIFEST.in ./
COPY specforge ./specforge
COPY plugins/specforge-tpu ./plugins/specforge-tpu
COPY examples ./examples
COPY scripts ./scripts
# --no-deps: torch, transformers and sglang come pinned from the base image.
RUN pip install --no-deps --no-build-isolation -e . ./plugins/specforge-tpu \
 && python3 -c "import mooncake.store, specforge_tpu, specforge.cli"

CMD ["/bin/bash"]
