# Domino disaggregated overfit and DFLASH serving gate

This gate checks the smallest end-to-end Domino disaggregated training path:
one patched SGLang capture server captures target features for one ShareGPT
row, and a separate DP trainer repeatedly consumes that row. A companion
post-training gate exports the resulting checkpoint and loads it in real
SGLang DFLASH speculative serving. Every target model uses these same entry
points; model-specific paths, templates, tensor keys, and reasoning behavior
are inputs rather than separate test implementations.

Domino is a DFLASH draft architecture with its own GRU logit correction, not a
second serving algorithm. Training therefore needs a Domino architecture
config, while SGLang serves the resulting draft through the common `DFLASH`
algorithm.

| Example | Dataset contract | `CHAT_TEMPLATE` | Reasoning inputs |
| --- | --- | --- | --- |
| Qwen3-8B | non-reasoning | `qwen` | `REASONING_POLICY=forbidden`, `ENABLE_THINKING=0` |
| Qwen3.6-27B | structured reasoning | `qwen3.5` | `REASONING_POLICY=required`, `ENABLE_THINKING=1` |

Mask token, capture layers, and block size are read from the selected draft
config so the launch command does not duplicate model metadata.

`SOURCE_DATA_PATH` is the only data input. It must point to validated,
ShareGPT-compatible JSONL; data regeneration is intentionally outside this
gate. See [Data Preparation](../basic_usage/data_preparation.md) for the general
regeneration workflow. Before starting a server or trainer, this gate enforces
the requested reasoning contract and the real post-tokenization loss-mask
threshold.

`DRAFT_CONFIG_PATH` describes the Domino draft architecture to initialize and
train, including its block size, mask token, and captured target layers. The
overfit stage does not require an existing draft checkpoint. The real serving
stage necessarily consumes the checkpoint produced by that overfit run because
DFLASH serving verifies proposals from the trained draft against the target.

Run from the repository root on an eight-GPU host with Mooncake and a patched
SGLang capture server installed:

```bash
SOURCE_DATA_PATH=/path/to/validated_sharegpt.jsonl \
TARGET_MODEL_PATH=Qwen/Qwen3-8B \
DRAFT_CONFIG_PATH=configs/qwen3-8b-domino.json \
CHAT_TEMPLATE=qwen \
EMBEDDING_KEY=model.embed_tokens.weight \
REASONING_POLICY=forbidden ENABLE_THINKING=0 MAX_LENGTH=512 \
SERVER_GPUS=0 \
CONSUMER_GPUS=1,2,3,4,5,6,7 \
TRAIN_DP=7 \
bash examples/disagg/run_domino_disagg_overfit_gate.sh
```

For the validated Qwen3.6-27B reasoning dataset and full-attention Domino config:

```bash
SOURCE_DATA_PATH=/path/to/validated_qwen36_reasoning.jsonl \
TARGET_MODEL_PATH=Qwen/Qwen3.6-27B \
DRAFT_CONFIG_PATH=configs/qwen3.6-27b-domino-full-attention.json \
CHAT_TEMPLATE=qwen3.5 \
EMBEDDING_KEY=model.language_model.embed_tokens.weight \
REASONING_POLICY=required ENABLE_THINKING=1 MAX_LENGTH=2048 \
SERVER_GPUS=0 SERVER_TP=1 \
CONSUMER_GPUS=1,2,3,4,5,6,7 TRAIN_DP=7 \
bash examples/disagg/run_domino_disagg_overfit_gate.sh
```

The launcher intentionally refuses an existing `WORK_DIR`; choose a new path
for every run. It selects one clean single-turn row into `single_sample.jsonl`
and runs the selected target tokenizer and SpecForge preprocessing used by the
producer. Selection succeeds only when the resulting loss mask has at least 32
tokens, matching the producer's `2 * block_size` filter, and the rendered row
fits without truncation. Selection also writes `prompt_artifact.json`, which
records the exact rendered prompt/target suffix.
The serving client uses that artifact instead of reconstructing a target from
visible content; this preserves structured reasoning while stripping hidden
reasoning fields from request history.

`DISAGG_MAX_PROMPTS=1` limits the producer input to that one source row.
`NUM_EPOCHS` is then set to `DISAGG_MAX_STEPS * TRAIN_DP`, so replay produces
exactly enough sample references for every DP rank at every bounded optimizer
step. The default gate stops at `DISAGG_MAX_STEPS=400`.

Useful bounded overrides are:

```bash
WORK_DIR=outputs/domino-overfit-$(date +%Y%m%dT%H%M%S) \
DISAGG_MAX_STEPS=600 \
bash examples/disagg/run_domino_disagg_overfit_gate.sh
```

The command exits successfully only when all three conditions hold:

- the final logged loss is at most `0.0001` (displayed as approximately
  `0.0000`);
- final token accuracy is exactly `1.0000` by default;
- a `training_state.pt` checkpoint exists under the fresh consumer output.

The final JSON emitted by `scripts/gates/check_overfit_metrics.py` records the step,
loss, token accuracy, and checkpoint path. A nonzero exit means this gate did
not pass even if the trainer process itself exited cleanly.

## Real DFLASH serving gate

Run the strict serving check against the fresh overfit checkpoint and its
`prompt_artifact.json`:

```bash
CHECKPOINT_PATH=/path/to/consumer/<run-id>-step400 \
PROMPT_ARTIFACT_PATH=/path/to/run/prompt_artifact.json \
TARGET_MODEL_PATH=Qwen/Qwen3-8B \
DRAFT_CONFIG_PATH=configs/qwen3-8b-domino.json \
EMBEDDING_KEY=model.embed_tokens.weight \
SERVED_MODEL=qwen3-8b \
SGLANG_ROOT=/path/to/domino-capable/sglang \
SGLANG_PYTHON=/path/to/sglang-env/bin/python \
bash examples/disagg/run_domino_dflash_serving_gate.sh
```

Alternatively, set `RUN_REAL_SERVING_GATE=1` on the training wrapper and set
`SERVING_SGLANG_ROOT` / `SERVING_PYTHON` when serving uses a separate SGLang
environment. The wrapper releases the capture server and trainer GPUs before
starting the real serving process.

The runner exports the runtime checkpoint with the target embedding, changes
the exported architecture to SGLang's `DFlashDraftModel`, and launches a real
server with `--speculative-algorithm DFLASH` and block size 16. Its client only
uses `/v1/chat/completions`, removes `reasoning_content` from request history,
and takes `enable_thinking` from the prompt artifact. It combines response
`reasoning_content + content` and compares that text with the artifact's
rendered target suffix. Reasoning parsers, TP, and serving context are ordinary
launcher inputs (`REASONING_PARSER`, `SERVING_TP`, and
`SERVING_CONTEXT_LENGTH`).

This is a strict per-request gate. The SGLang checkout must support
`return_meta_info=true` on the OpenAI chat API and expose
`choices[0].meta_info.spec_accept_length`; aggregate `/server_info` acceptance
statistics are not accepted as a substitute. The result passes only when:

- `/server_info.speculative_algorithm` is `DFLASH`;
- the request contains no `reasoning_content`;
- the choice reports `spec_accept_length >= 16`;
- the generated output matches at least the first 16 target tokens.

The auditable result is written to `serving_gate.json`. An older SGLang checkout
that omits choice metadata fails explicitly instead of being reported as real
per-request serving success.
