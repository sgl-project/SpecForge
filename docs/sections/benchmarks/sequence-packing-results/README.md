# Sequence-packing benchmark evidence

These files support the [EAGLE3](../eagle3-sequence-packing.md),
[DFlash/DFlash2](../dflash-sequence-packing.md),
[32-step online](../online-sequence-packing.md) and
[256-step full-model](../full-model-sequence-packing.md) reports measured on
2026-10-02. The implementation was based on upstream
`53398a8f01ae47175bee8459c5b5cca3848c8a7e` plus the sequence-packing changes.
Each report defines its workload and timing boundary; the results are not
interchangeable estimates of end-to-end gains.

## Committed files

| File | Evidence |
| --- | --- |
| [full-model-analysis.json](full-model-analysis.json) | Every measured run's timing values and medians for the 256-step comparison. |
| [full-model-audit.json](full-model-audit.json) | Independent checks for all 16 measured runs and four warmups: model dimensions, optimizer participation, updates, finiteness, compilation and pipeline ordering. |
| [target-parameter-inventory.json](target-parameter-inventory.json) | Qwen3-4B checkpoint tensor names, shapes and counts; verifies all 36 target layers and 4,022,468,096 saved parameters. |
| [online-32step.json](online-32step.json) | Derived summary preserving every run's timing samples, settings and lifecycle checks from the shorter online experiment. |
| [eagle3-medium-bf16.json](eagle3-medium-bf16.json) | Final-source medium EAGLE3 compute timings, memory, numerical checks and source hashes. |
| [eagle3-large-bf16.json](eagle3-large-bf16.json) | Larger EAGLE3 compute case, measured before the final FSDP metadata conversion. |
| [eagle3-tiny-fp32.json](eagle3-tiny-fp32.json) | Derived FP32 correctness summary, including the number of gradient tensors and maximum differences. |
| [dflash-family-128anchors-bf16.json](dflash-family-128anchors-bf16.json) | Final optimized DFlash/DFlash2 compute measurements with 128 anchors per document. |
| [dflash-family-512anchors-bf16.json](dflash-family-512anchors-bf16.json) | Final optimized compute measurements with 512 anchors per document. |
| [dflash-family-tiny-fp32.json](dflash-family-tiny-fp32.json) | Derived FP32 correctness summary for DFlash/DFlash2, including exact sampled-anchor checks. |

The two tiny FP32 summaries replace per-parameter gradient lists with tensor
counts, the all-pass flag and maximum absolute/relative differences. Their other
fields are unchanged, and each includes the original report SHA256. The 32-step
summary removes detailed per-capture event lists while preserving individual
run timings, including warmups, and records the original report SHA256. All
other JSON files preserve their corresponding local evidence; only the target
inventory's missing final newline was normalized by the repository hooks.

## Configuration and interpretation

The online benchmark's raw CLI settings retain `draft_layers=2`, a default used
only when no draft config is supplied. Both online experiments supplied
`configs/qwen3-4b-dflash.json`; the resolved config and runtime model inventory
confirm **five draft layers**. Likewise, supplying `prompts_path` bypasses the
synthetic `lengths` and `prompt_fraction` defaults. These experiments use real
ShareGPT tokens and their actual assistant supervision masks.

The target checkpoint has 36 layers and is frozen. All trainable draft parameters
are updated: 537,427,200 parameters for DFlash and 558,918,912 for DFlash2. The
256-step evidence checks each trainable tensor's optimizer step count and scans
all trainable elements for finiteness. Layer-update hashes sample weight values;
they do not assert that every element changed or that packed and padded training
trajectories are numerically identical.

The full-model training-step diagnostics cover the first 250 of 256 steps and
include periodic metrics. Pipeline timing includes the step-128 checkpoint;
completion timing includes both step-128 and step-256 checkpoints. Startup,
data preparation and separate corpus warmup are excluded. The tests do not
establish convergence, final model quality or speculative-serving throughput.

## Evidence retained locally

The original 256-step raw report, about 9.65 MB, is retained locally as
`artifacts/sequence-packing/full-model/long_v2.json`; it is **not committed**.
Its SHA256 is
`f8895e415d1baaef7ddaa74a7f8e322d83d13e008d2f7606b0378fb0b4861179`.
The original 32-step raw report is retained locally as
`artifacts/sequence-packing/e2e/full_v1.json`; its SHA256 is
`d0284b6ab87ad26bba3c6624c4d4b203cb4f18fef08609476729d650b55f9be4`.

Detailed source snapshots, source-verification records, dataset manifests,
preparation and service-launch helpers, service logs, ad hoc distributed probes,
reservation metadata and cleanup receipts also remain local. They are not part
of this directory or promised as repository downloads. Original conversation
text was not copied into these artifacts. The public dataset revision is
`anon8231489123/ShareGPT_Vicuna_unfiltered@192ab2185289094fc556ec8ce5ce1e8e587154ca`;
the 1,024-prompt tokenized file SHA256 is
`f90478e0d3e32f0d3c45c515381b9bc42951cdd6931830ac1a7380f4459f24b1`.

The full-model measured source manifest matches all 17 changed/new production
files in the implementation at packaging time. Benchmark documentation and
public evidence packaging were finalized after the measurements.
