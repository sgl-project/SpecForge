# syntax=docker/dockerfile:1
FROM lmsysorg/sglang:v0.5.18-cu130

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git sudo tmux vim zsh && \
    rm -rf /var/lib/apt/lists/*

# Install uv somewhere the development user can execute it.
RUN curl --proto '=https' --tlsv1.2 -sSf https://astral.sh/uv/install.sh -o /tmp/install-uv.sh && \
    UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh /tmp/install-uv.sh && \
    rm /tmp/install-uv.sh

# Create non-root user with specified UID and GID
# NOTE: Replace with your own UID and GID. This is a workaround from https://github.com/microsoft/vscode-remote-release/issues/49#issuecomment-489060908.
ARG HOST_UID=1003
ARG HOST_GID=1003
RUN groupadd -g "$HOST_GID" devuser && \
    useradd -m -u "$HOST_UID" -g "$HOST_GID" -s /bin/zsh devuser

# Give devuser sudo access
RUN echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    chmod 0440 /etc/sudoers.d/devuser

# Use a minimal shell configuration, without inherited SGLang activation.
RUN <<'SH'
set -eu
cat > /home/devuser/.zshenv <<'EOF'
export VIRTUAL_ENV=/opt/specforge-venv
export PATH="$VIRTUAL_ENV/bin:/home/devuser/.cargo/bin:$PATH"
EOF
cat > /home/devuser/.zshrc <<'EOF'
# Restore the project interpreter after any system-wide shell startup files.
source ~/.zshenv
autoload -Uz compinit
compinit
PROMPT='%n@%m:%~%# '
EOF
chown -R devuser:devuser /home/devuser
SH

# Set workspace directory and ownership
WORKDIR /sgl-workspace/specforge
RUN mkdir -p /opt/specforge-venv && \
    chown devuser:devuser /sgl-workspace/specforge /opt/specforge-venv

ENV HOME=/home/devuser
ENV VIRTUAL_ENV=/opt/specforge-venv
ENV PATH="/opt/specforge-venv/bin:/home/devuser/.cargo/bin:${PATH}"

# Switch to devuser
USER devuser

# The pinned SGLang revision requires Rust 1.92.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/install-rust.sh && \
    sh /tmp/install-rust.sh -y --profile minimal --default-toolchain 1.92 --no-modify-path && \
    rm /tmp/install-rust.sh

# Keep the isolated environment outside the workspace bind mount.
RUN uv venv --seed -p 3.11 "$VIRTUAL_ENV" && \
    uv pip install --python "$VIRTUAL_ENV/bin/python" \
        wheel setuptools setuptools-rust setuptools-scm psutil packaging ninja && \
    uv pip install --python "$VIRTUAL_ENV/bin/python" \
        torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130

# Only dependency metadata invalidates this expensive layer. Register the
# mounted source and capture plugin in postCreateCommand, not in the image.
COPY pyproject.toml /tmp/specforge-build/pyproject.toml
RUN /opt/specforge-venv/bin/python - <<'PY'
import subprocess
import sys
import tomllib

with open("/tmp/specforge-build/pyproject.toml", "rb") as metadata:
    project = tomllib.load(metadata)["project"]

subprocess.run(
    [
        "uv", "pip", "install", "--python", sys.executable,
        "--no-build-isolation", "--prerelease=allow",
        *project["dependencies"],
        *project["optional-dependencies"]["dev"],
        "sglang @ git+https://github.com/sgl-project/sglang.git@c757c4bb7c55d1eb69fdb2bdc6e6fb8507acb7a0#subdirectory=python",
        "mooncake-transfer-engine-cuda13",
    ],
    check=True,
)
PY
