# EAGLE-Harness serving compatibility patch

This bundle preserves the adapter used for the Qwen3.8-27B stage-2 serving
comparison, including the dense, SVD, published low-rank, and trained low-rank
heads. It patches **EAGLE-Harness**, not the SGLang source tree.

Apply only to [eagle-hq/EAGLE-4](https://github.com/eagle-hq/EAGLE-4) revision
`3e20798bcce810486cd810ecf087936488106ae2`. The tested engine was
`sglang==0.5.23.dev252+eagle.4.rocm`; other package versions and the patched
source SHA-256 hashes are recorded in [manifest.json](manifest.json).

## Changes

- `eagle_harness/models/eagle4_config.py`: opt-in validation for dense heads.
- `eagle_harness/sglang/models/llama_eagle4.py`: full-vocabulary dense-head
  serving at TP1, identity vocabulary-buffer loading, packed expert loading,
  separate draft expert placement, and ROCm ReLU2 compatibility. Low-rank
  heads retain the upstream candidate-selection and full-head scoring path.
- `eagle_harness/sglang/worker.py`: initialize MoE runtime configuration before
  constructing the draft worker, including when the target is dense.

The official attention residual is preserved. The historical
`specforge_normalized_fusion_residual` key remains accepted by config validation
to reproduce the tested files exactly, but this adapter does not read it or
override the residual. Earlier scratch patches with that override are superseded.
Untimed diagnostic tracing is not included.

## Apply and verify

From an existing SpecForge checkout, in a Linux shell:

```bash
bundle="$(pwd)/patches/eagle-harness/3e20798"
git clone https://github.com/eagle-hq/EAGLE-4.git EAGLE-4-serving
cd EAGLE-4-serving
git checkout --detach 3e20798bcce810486cd810ecf087936488106ae2
git apply --check "$bundle/serving-compat.patch"
git apply "$bundle/serving-compat.patch"
uv run --no-project python - "$bundle/manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
for name, expected in manifest["sha256_lf"].items():
    source = pathlib.Path(name).read_text(encoding="utf-8")
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != expected:
        raise SystemExit(f"Hash mismatch: {name}")
print("All patched source hashes match.")
PY
```

Use the pinned checkout's `docs/install.md`, `docs/release.env`, and
`docker/rocm/requirements.lock` for its ROCm runtime dependencies. Install the
patched checkout into that serving environment with
`uv pip install --python /path/to/serving-venv/bin/python --no-deps -e .`.
Check `eagle_harness.__file__` with that interpreter to ensure the patched source
is loaded. Use `--speculative-moe-runner-backend triton` on ROCm.
The dense path supports TP1 only; the recorded comparison also used TP1 for
all low-rank heads. This bundle does not establish compatibility with newer
engine releases or other serving configurations.

Validation when archived: the patch applied cleanly to the pinned source, all
three modified Python files parsed, and their LF-normalized hashes matched
both the baseline and post-training evaluation manifests. No new GPU run was
performed to archive the bundle.

## Attribution and modification notice

This is an independent modification of Eagle 4 technology developed by Eagle AI
Corp, not an official release or an affiliated, certified, sponsored, or endorsed
Eagle AI Corp offering. References identify origin and compatibility only.
Modification notice: Nan Qin (`yudigege86`), archived November 16, 2026;
the three materially modified files and their changes are identified above.

The Eagle-derived patch is subject to the accompanying Eagle 4 Source-Available
[LICENSE](LICENSE), which is **not an OSI-approved open-source license**, and
incorporated [Trademark Policy](TRADEMARKS.md). Downloading, cloning, forking,
copying, accessing, modifying, using, hosting, distributing, or otherwise
exercising licensed rights constitutes acceptance of that license and policy.
Preserved upstream notices are in [NOTICE](NOTICE),
[SOURCE_NOTICES.txt](SOURCE_NOTICES.txt), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), with license texts in
`LICENSES/`. The surrounding SpecForge license does not relicense these materials.
