# DMTP training through the capture plugin

DMTP freezes the target and its native MTP, fuses the MTP predictor state with
the first candidate's embedding, and trains DFlash to predict the remaining
candidates. Training uses the first candidate from the dataset. The example
uses seven candidate slots: `[T1, MASK, MASK, MASK, MASK, MASK, MASK]`; T1 does
not receive draft prediction loss.

## Capture contract

The optional `mtp_hidden` artifact has shape `[1, sequence_length, hidden_size]`.
Its `anchor_post_norm_v1` alignment is:

```text
mtp_hidden[a] = MTP(input_ids[a], target_last_hidden[a-1], causal MTP history)
T1            = input_ids[a+1]
```

The captured tensor is after MTP final normalization, before its LM head. Row
zero is padding and is never selected for training. The target context visible
to a block ends before A (`a`), even though the offline target prefill computed
the entire sequence. MTP processing uses shifted target/token pairs and retains
the complete causal history of each request. Requests must contain one document;
packed documents and multimodal capture are not supported.

The plugin executes the frozen MTP after the normal target prefill, through the
existing forward observer. It does **not** enable speculative decoding or modify
SGLang source files. It uses SpecForge's PyTorch native-MTP implementation, not
the SGLang draft worker. There is no MTP token sampling or vocabulary projection
during capture. Only the writer rank holds the replicated MTP; reserve additional
GPU memory for its embedding, weights, and activations.

The tested SGLang source revision is `4517a5ff1b` in the `.sgl` environment.

The initial implementation supports unquantized, single-layer, dense
Qwen3.5/Qwen3.6 native MTP checkpoints with default RoPE. All required weights
are loaded strictly. Unsupported families, MoE MTP, or missing weights fail
rather than produce random teacher features.

## Server and online training

Use an SGLang environment with the forward-observer and deferred-response plugin
APIs. The standard SGLang version pinned by this main branch is insufficient;
this feature is developed against the user-provided SGLang main environment.
Install this SpecForge checkout and its capture plugin into that environment
without replacing its existing Torch/Transformers/SGLang stack:

```bash
python -m pip install --no-deps -e .
python -m pip install --no-deps -e plugins/sglang-spec-capture
```

For source-tree development, put the checkout root and
`plugins/sglang-spec-capture` on `PYTHONPATH` instead. The additional SpecForge
dependency is needed only when enabling MTP features; ordinary capture remains
independent of SpecForge model imports.

Configure a Mooncake master and use the same `MOONCAKE_*` connection settings in
the capture server and producer. For example, with an already running master:

```bash
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_PROTOCOL=tcp

export MODEL=/path/to/Qwen3.6-27B
export SPECFORGE_SPEC_CAPTURE=1
export SPECFORGE_MTP_MODEL_PATH="$MODEL"
export SGLANG_PLUGINS=specforge_spec_capture
export SGLANG_SPEC_CAPTURE_GPU_PUT=0

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --speculative-algorithm DFLASH \
  --enable-return-aux-hidden-states \
  --return-aux-hidden-state-layer-ids 1 16 31 46 61 \
  --return-hidden-states-mode full \
  --chunked-prefill-size -1 \
  --mem-fraction-static 0.8 \
  --port 30000
```

`SPECFORGE_MTP_MODEL_PATH` must identify the same checkpoint as `--model-path`.
The native aux-capture flag uses DFLASH as the target-feature convention; it does
not request an autoregressive MTP rollout. Requesting `mtp_hidden` without an
enabled MTP provider fails explicitly. The response includes
`mtp_hidden_alignment: anchor_post_norm_v1` alongside the feature references.

Use `examples/configs/online/disaggregated/external/qwen3.6-27b-dmtp.yaml`, update
the target/data paths and server URL, and launch through the normal SpecForge
CLI. The trainer strategy is `dmtp`, and the draft architecture is
`DMTPDraftModel`. Its fusion parameters are part of the draft state and survive
checkpoint/resume. This change does not add a DMTP speculative inference engine
or make standard DFlash serving understand the new architecture.

## Offline features

The existing local-ModelRunner preprocessing command does not implement DMTP.
Use the plugin client with tokenized JSONL instead. Each input line contains
`input_ids` and a same-length `loss_mask`.

```bash
python scripts/prepare_dmtp_hidden_states.py \
  --input-path tokenized.jsonl \
  --output-path cache/dmtp-features \
  --server-url http://127.0.0.1:30000 \
  --target-model-path "$MODEL" \
  --target-layer-ids 1 16 31 46 61
```

The output directory must be empty. Each `.ckpt` contains input IDs, loss mask,
target features, `mtp_hidden`, and its alignment metadata. Offline training uses
`training.strategy: dmtp` and `data.hidden_states_path: cache/dmtp-features` with
the same draft config. No target or MTP model is loaded by the offline trainer.

## Validation

Unit tests cover token alignment, prefix equivalence, context visibility,
frozen features, fusion gradients, and checkpoint state. The optional live test
starts isolated SGLang/Mooncake processes, captures variable-length requests,
checks real native-weight predictor states, writes and reads offline features,
and runs both a DMTP optimizer step and two steps through the training CLI:

```bash
SPECFORGE_DMTP_MODEL="$MODEL" \
  python -m unittest -v tests.test_runtime.test_dmtp_capture_live
```

The live test defaults to server GPU 1 and trainer GPU 2; override with
`SPECFORGE_DMTP_SERVER_GPU` and `SPECFORGE_DMTP_TRAIN_GPU`.
