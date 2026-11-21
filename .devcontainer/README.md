# Remote development with VS Code

This container provides a Python 3.11 environment at `/opt/specforge-venv`,
CUDA 13 training dependencies, the pinned SGLang capture revision, Mooncake,
and development tools. The checkout is mounted at `/sgl-workspace/specforge`
and installed editably along with the capture plugin.

## Connect and create the container

1. Prepare a remote Linux NVIDIA host with a CUDA 13-compatible NVIDIA driver,
   Docker, and NVIDIA Container Toolkit configured for Docker. Your SSH user
   must be able to run Docker and write to the checkout. Training also needs
   enough GPU memory for the target capture server and draft trainer. This is
   a Linux GPU workflow; a local macOS Docker VM cannot run the training example.
2. Install VS Code's **Remote - SSH** and **Dev Containers** extensions. Use
   **Remote-SSH: Connect to Host...**, then clone on that host:

   ```bash
   git clone https://github.com/sgl-project/SpecForge.git
   cd SpecForge
   id -u
   id -g
   ```

3. Open the remote checkout in VS Code. Dev Containers automatically adjusts
   `devuser` to the Linux host's UID/GID, and the creation command restores
   environment ownership before installing the checkout. To set matching IDs
   at build time too, optionally add `args` to the `build` object in
   `.devcontainer/devcontainer.json`, using the numeric UID and GID printed
   on the **remote host**. For example, for UID/GID 1000:

   ```json
   "build": {
       "dockerfile": "Dockerfile",
       "context": "..",
       "args": {
           "HOST_UID": "1000",
           "HOST_GID": "1000"
       }
   }
   ```

   The build defaults are 1003/1003. Matching the host IDs also gives `devuser`
   write access when running the image outside Dev Containers. If Docker reports
   an ID already exists in the base image, inspect the conflicting account before
   choosing a host account/ID mapping; do not bypass it by running as root.
4. Run **Dev Containers: Reopen in Container**. The first build downloads and
   builds the training stack, which can take a while. Wait for the
   `postCreateCommand` editable installation to finish successfully before
   using the environment. The VS Code Python interpreter defaults to
   `/opt/specforge-venv/bin/python`.

Source edits are available through the editable installation; reopen the
container or restart the running Python process as needed. Use **Dev
Containers: Rebuild Container** after changing dependency metadata in
`pyproject.toml` or the Dockerfile. Dependency layers are cached separately
from source files. The environment lives outside the workspace mount and
survives reopening; rebuilding recreates it from the Dockerfile. To repeat
the editable registration from the repository root without resolving or
reinstalling dependencies, run the editable install command:

```bash
/opt/specforge-venv/bin/python -m pip install --no-deps --no-build-isolation -e . -e plugins/sglang-spec-capture
```

## Check the environment

Run these checks as `devuser` in the container terminal, from
`/sgl-workspace/specforge`:

```bash
command -v python pip pip3
python -c 'import sys; print(sys.executable); print(sys.prefix)'
python -m pip --version
python -m pip check
specforge --help
python - <<'PY'
import specforge
import specforge_sglang_capture
from sglang.srt.model_executor.forward_observer import ForwardObserver
from sglang.srt.managers.deferred_output import ResponseCompletion

print(specforge.__file__)
print(specforge_sglang_capture.__file__)
PY
```

All three commands must resolve under `/opt/specforge-venv/bin`, `sys.prefix`
must be `/opt/specforge-venv`, and pip must report an installation inside that
environment. The module paths must point into the mounted checkout, with the
capture plugin under `plugins/sglang-spec-capture`. `pip check` should report
no broken requirements.

Check a fresh interactive shell and a noninteractive shell too:

```bash
zsh -ic 'command -v python pip pip3; python -c "import sys; print(sys.prefix)"; python -m pip --version'
zsh -c 'command -v python pip pip3; python -c "import sys; print(sys.prefix)"; python -m pip --version'
```

If paths point elsewhere, remove custom shell activation or PATH overrides,
select `/opt/specforge-venv/bin/python` with **Python: Select Interpreter**,
and open a new terminal. If package imports or `specforge` are missing, inspect
the Dev Containers creation log and rerun the editable command above; a failed
install means setup is incomplete. Dependency conflicts require a rebuild
with the current metadata. For permission errors, compare `id` with
`ls -ld /opt/specforge-venv /sgl-workspace/specforge` and the remote host's
UID/GID. If environment ownership is stale, rerun the full `postCreateCommand`
from `.devcontainer/devcontainer.json` to repair it before installing. For
containers run outside Dev Containers, correct the build arguments and rebuild.
Check host checkout ownership too; rebuilding cannot change ownership of a
bind mount.

## Prepare data and model access

Run from the container's repository root:

```bash
python scripts/prepare_data.py --dataset sharegpt
hf auth login
```

The dataset command writes `cache/dataset/sharegpt_train.jsonl`, which the
Llama recipe already references. Request and receive access to
`meta-llama/Llama-3.1-8B-Instruct` on Hugging Face before training, then log in
with an account/token that has that access. The capture server also needs
access in its own environment. A 401/403 or gated-repository error means
authentication or model approval is still missing. Container home credentials
may need to be entered again after rebuilding.

## Start the first Llama EAGLE3 run

Follow the current [Llama online example](../docs/sections/examples/llama3-eagle3-online.md),
the [capture plugin setup](../plugins/sglang-spec-capture/README.md), and the
[disaggregated training guide](../docs/sections/basic_usage/disaggregated_training.md)
before launching the [external Llama recipe](../examples/configs/online/disaggregated/external/llama3.1-8b-eagle3.yaml).
This recipe requires already-running, reachable **SGLang capture and Mooncake
services**. `specforge train` starts the producer and trainer; it does not
start those external services.

The capture server must use the compatible SGLang revision and plugin,
the Llama target model, and the EAGLE3 capture convention and auxiliary layer
IDs required by the draft configuration. The plugin guide's DFlash command is
an example for a different capture convention, so adapt its model and capture
flags for this recipe. Allocate server and trainer GPUs explicitly to avoid
competing for the same memory.

Make these `deployment.disaggregated` YAML fields match the services as seen
**from inside the container**:

| Field | Checked-in endpoint |
| --- | --- |
| `server_urls` | `http://127.0.0.1:30000` |
| `mooncake_metadata_server` | `http://127.0.0.1:35880/metadata` |
| `mooncake_master_server_addr` | `127.0.0.1:35551` |
| `mooncake_protocol` | `tcp` |

Loopback addresses refer to this container, not the remote Docker host or
another container. Use reachable service addresses when services run elsewhere,
and configure Mooncake consistently on the capture server and trainer. Confirm
connectivity before training; connection refusals or timeouts mean the service
setup or addresses need attention. Installing the packages alone does not
provide running services.

After those prerequisites are ready, launch from the repository root:

```bash
specforge train --config examples/configs/online/disaggregated/external/llama3.1-8b-eagle3.yaml
```

For a short smoke run instead, use the documented `training.max_steps` override:

```bash
specforge train --config examples/configs/online/disaggregated/external/llama3.1-8b-eagle3.yaml \
  training.max_steps=10
```

Use fresh `run_id`, `output_dir`, `deployment.disaggregated.control_dir`, and
`deployment.disaggregated.consumer_state_dir` values for each new attempt,
as described in the disaggregated guide. The small step count still needs
GPU resources, model access, prepared data, and live capture services.

## Validate image changes

On the Linux GPU host, build from the repository root:

```bash
docker build -f .devcontainer/Dockerfile -t specforge-dev \
  --build-arg HOST_UID="$(id -u)" --build-arg HOST_GID="$(id -g)" .
```

Then reopen through Dev Containers to exercise its actual lifecycle command
and run the environment checks above. Repeat the build to verify dependency
layers are cached. Editing only a Python source file should preserve those
layers and be visible through the editable mount; changing `pyproject.toml`
must rerun dependency installation on rebuild. Repeat the lifecycle command
to verify it succeeds without downloading dependencies. Also check a host
account with non-default IDs and a checkout path containing spaces, confirming
both the mount and `/opt/specforge-venv` remain writable. The GPU/model smoke
run is a separate check from establishing the development environment.
