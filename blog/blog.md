# SpecForge v0.3.0: a Unified Disaggregated and Colocated Speculative Decoding Stack, and New Open SpecBundle Draft Models

**The SpecForge Team · August 2026**

When we first released [SpecForge](https://www.lmsys.org/blog/2025-07-25-spec-forge/), a training job owned both the frozen target model and the draft model being optimized. This made EAGLE3 draft-model training practical and directly compatible with SGLang, but it also tied two very different workloads to the same process lifecycle and resource topology.

Today, we are introducing a major update to SpecForge. The new runtime separates target-model inference from draft-model training, supports a broader family of speculative decoding algorithms, and unifies online, offline, and disaggregated workflows behind one typed training entry point. Alongside the release, we are publishing more draft models, covering different speculative decoding methods and different target models.

## What's New

- **Online training is now fully disaggregated.** Patched SGLang servers capture target-model features, Mooncake transports the tensors, and trainer workers consume lightweight references through a separate control plane.
- **Inference and training can scale independently.** On our 8xH20 testbed, a topology with **3 SGLang servers and 5 trainer workers** improves end-to-end training throughput by approximately **10%** over our previous colocated implementation.
- **One runtime now supports multiple drafting families:** [EAGLE3](https://arxiv.org/abs/2503.01840), **EAGLE3.1**, [P-EAGLE](https://arxiv.org/abs/2602.01469), [DFlash](https://arxiv.org/abs/2602.06036), [Domino](https://arxiv.org/abs/2605.29707), and [DSpark](https://arxiv.org/abs/2607.05147), together with the optional [D-PACE](https://arxiv.org/abs/2605.18810) objective for DFlash.
- **More draft models are being released, most of them contributed by the community.** Partners and individual contributors have trained and published draft models with this runtime across different speculative decoding methods and different target models, all trained only on open data.

This release also ships a training-serving consistency gate that verifies capture, training, export, and SGLang serving agree on a controlled example — a fast correctness check before investing in a full training run.

## From a Coupled Trainer to a Training Pipeline

Online draft-model training contains two distinct workloads:

- The **target side** runs a large, frozen model over training conversations to capture hidden states. It is inference-heavy, often uses tensor parallelism, and benefits from a production inference engine.
- The **draft side** trains a much smaller model with forward and backward passes. It scales through data or sequence parallelism and has a different memory and compute profile.

In the previous colocated design, both sides shared one lifecycle and one fixed resource layout. This created three practical limitations:

1. **A fixed inference-to-training ratio.** Scaling trainer workers also affected target-model placement, even when only one side was the bottleneck.
2. **Resource interference.** Target capture and draft optimization competed inside the same tightly coupled job.
3. **A shared failure boundary.** Slow or failed feature generation could stall training, while excess production could create unbounded memory pressure.

The key observation is simple: **trainers do not need to own the target model**. They only need the token sequences, masks, and target features required by the selected training objective. Making that boundary explicit changes SpecForge from a trainer process containing a target model into a coordinated training pipeline.

## One Boundary, Three Contracts

The new online runtime has a producer pool and a consumer pool. Producers schedule prompts across patched SGLang capture servers. SGLang writes feature tensors to Mooncake, while SpecForge sends only lightweight `SampleRef` metadata through the control plane. Trainer ranks resolve those references when they are ready to build a batch.

![SpecForge online-disaggregated training architecture](./specforge-disaggregated-architecture.svg)

*Figure 1. The online-disaggregated training flow. Large tensors remain in the data plane; references and lifecycle state travel through the control plane.*

### 1. The capture contract

Every URL in `deployment.disaggregated.server_urls` creates a rollout worker connected to a patched SGLang server. Workers lease disjoint prompts from a shared controller, so capture capacity can change without changing the trainer topology.

The capture support is a small patch on top of the pinned `sglang==0.5.14`: [`patches/sglang/v0.5.14/spec-capture.patch`](https://github.com/sgl-project/SpecForge/blob/main/patches/sglang/v0.5.14/spec-capture.patch) adds an `--enable-spec-capture` flag and a server-side sink that writes captured tensors directly into Mooncake using the feature store's key layout. A capture server is a stock SGLang server with this patch applied.

This creates a clear ownership boundary: SGLang owns target-model parallelism and feature capture, while SpecForge owns prompt scheduling, reference publication, and draft-model optimization.

### 2. The delivery contract

Captured hidden states can be large, so forwarding them through a Python queue or control database would quickly become a bottleneck. SpecForge separates tensor storage from sample coordination:

- SGLang writes feature tensors to Mooncake.
- The producer publishes tensor-free `SampleRef` records.
- `FeatureDataLoader` resolves references into tensor-carrying training batches.
- Trainers release feature objects after an optimizer-step acknowledgement.

The training loop depends on the `FeatureStore` contract rather than a specific transport. The same consumer path can therefore use local features, a shared directory, or Mooncake-backed online capture.

### 3. The lifecycle contract

Distributed ranks must advance together. The consumer releases references in complete optimizer-step quanta, so every rank receives the samples it needs for one synchronized update. High and low in-flight watermarks pause and resume capture, preventing the producer from running arbitrarily far ahead of training.

At optimizer boundaries, consumer rank 0 records trained sample IDs in a retained SQLite ledger before acknowledgements reduce the in-flight depth and release feature objects. After an interruption, the consumer can use that ledger to skip completed sample IDs and replay the remaining references.

Failures are explicit on the producer side as well. A failed capture worker returns its leased prompts to the shared controller, allowing healthy workers to continue. The run fails loudly if all capture servers are unavailable or a prompt exhausts its retry budget.

Together, these contracts keep the trainer independent of feature transport without weakening distributed-step alignment, bounded buffering, or recovery semantics.

## What This Separation Unlocks

The new boundary produces two user-visible benefits: infrastructure can be balanced around the workload, and algorithm implementations can share one training runtime.

### Independent inference and training pools

Target-model tensor and expert parallelism now belong to SGLang; draft-model data and sequence parallelism belong to SpecForge. The two pools can run under one local supervisor or as separate scheduler-managed jobs.

When both pools share a fixed GPU budget, their ratio remains a resource trade-off. The difference is that the trade-off is now explicit and tunable instead of hard-wired into the trainer. When more resources are available, either pool can be expanded without forcing the other to adopt the same topology.

#### Preliminary system result: 3 capture servers + 5 trainers

On an 8×H20 testbed, we evaluated Qwen3-8B Domino training with a 3K-token context length. Reallocating the workload to three SGLang capture servers and five trainer workers improved measured end-to-end training throughput by approximately 10% over the previous colocated implementation.

| Runtime | Target capture | Draft training | Relative end-to-end training throughput |
| --- | --- | --- | ---: |
| Previous colocated version | Coupled with the training job | Fixed colocated layout | 1.00× |
| New disaggregated runtime | 3 SGLang servers | 5 trainer workers | **1.10×** |

The profiles below show representative training windows for the two runtime topologies.

**(a) Colocated baseline**

![Qwen3-8B Domino 3k-context colocated training profile](./colocated.png)

**(b) Disaggregated: 3 SGLang servers + 5 trainer workers**

![Qwen3-8B Domino 3k-context disaggregated training profile](./disagg.png)

*Figure 2. Representative Qwen3-8B Domino training windows at a 3K-token context length. The throughput table reports the end-to-end comparison; the traces provide a qualitative view of the two execution patterns.*

In this workload, profiling indicates that the gain came from better matching feature-production supply to trainer demand. The broader benefit is configurability: different target sizes, sequence lengths, and drafting algorithms can use different capture-to-training ratios without requiring another trainer implementation.

### One runtime for multiple drafting families

SpecForge began with a strong focus on EAGLE3. The new runtime separates common systems concerns from strategy-specific modeling code.

Every strategy reuses prompt scheduling, feature transport, distributed execution, checkpointing, and process supervision. A strategy defines only the target features it needs, how those features become a training batch, its draft-model architecture, and its objective.

| Method | Strategy-specific idea | SpecForge support |
| --- | --- | --- |
| [EAGLE3](https://arxiv.org/abs/2503.01840) | Direct token prediction with training-time test and multi-layer target-feature fusion | Online-disaggregated, local offline, and disaggregated offline; optional [LK loss](https://arxiv.org/pdf/2602.23881) objective|
| [P-EAGLE](https://arxiv.org/abs/2602.01469) | Parallel multi-token prediction through a shared hidden state | Online-disaggregated |
| EAGLE3.1 | An EAGLE3 configuration variant with per-layer normalization and attention-drift settings | Online-disaggregated through the `eagle3` strategy |
| [DFlash](https://arxiv.org/abs/2602.06036) | Block-diffusion drafting that predicts a token block in parallel | Online-disaggregated, local offline, and disaggregated offline; optional [D-PACE](https://arxiv.org/abs/2605.18810) objective |
| [Domino](https://arxiv.org/abs/2605.29707) | A parallel draft backbone followed by a lightweight causal correction head | Online-disaggregated, local offline, and disaggregated offline |
| [DSpark](https://arxiv.org/abs/2607.05147) | Semi-autoregressive drafting with confidence modeling for adaptive verification | Online-disaggregated, local offline, and disaggregated offline |

Here, *unified* means one configuration schema, launcher, dataflow contract, trainer lifecycle, and checkpoint surface. It does not mean that every strategy supports every data source and topology combination.

## Data Source and Deployment Are Separate Choices

Online and offline describe where target features come from. Local and disaggregated describe how the training workflow is deployed. Keeping those concepts separate makes the supported combinations easier to understand:

| Feature source | Local/dataflow deployment | Producer/consumer deployment |
| --- | --- | --- |
| Online SGLang capture | No | Yes, with Mooncake |
| Offline feature checkpoints | Yes | Yes, with a shared feature store |

Every online run uses the producer/consumer topology; the trainer never initializes a colocated target model. Offline EAGLE3, DFlash, Domino, and DSpark training can run locally or with separate ingestion and consumer pools. P-EAGLE currently supports online training only.

### Feature source is not data policy

One further distinction matters in practice. Capture servers execute a full prefill over the conversations you provide and never generate the training responses, so online capture does not choose whose text the draft learns from — the dataset does. This choice matters more than any topology decision: when dataset responses were written by humans or by a different model, the draft learns to continue text the target itself would rarely produce, and acceptance saturates well below what the same draft reaches on target-generated data. In our training runs, regenerating dataset responses with the target model — greedily, in the reasoning mode that will be served — has been the single largest lever on final acceptance. We recommend target-generated data for every strategy. The consistency gate described below rests on the same property: its serving stage can only pass when the trained sample is text the target reproduces at temperature 0.

See the [training guide](https://github.com/sgl-project/SpecForge/blob/main/docs/basic_usage/training.md) for the complete strategy matrix and the [disaggregated training guide](https://github.com/sgl-project/SpecForge/blob/main/docs/basic_usage/disaggregated_training.md) for deployment details.

## One Configuration, One Entry Point

The topology lives in the same typed YAML document as the model, data, algorithm, and optimizer settings. The following excerpt illustrates the three-server/five-trainer shape used above:

```yaml
training:
  strategy: domino

deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 5
  disaggregated:
    control_dir: outputs/qwen3-8b-domino/control
    consumer_state_dir: outputs/qwen3-8b-domino/consumer-state
    backend: mooncake
    server_urls:
      - http://capture-0:30000
      - http://capture-1:30000
      - http://capture-2:30000
```

The same command resolves and launches the selected topology:

```bash
specforge train --config run.yaml
```

On a single node, the launcher can supervise both SpecForge roles. Under an external scheduler, the pools can be launched independently with the same configuration:

```bash
# Inference and ingestion pool
specforge train --config run.yaml --role producer

# Draft-model training pool
specforge train --config run.yaml --role consumer
```

There are no method-specific Python training entry points. Full, runnable configurations are available under [`examples/configs`](https://github.com/sgl-project/SpecForge/tree/main/examples/configs).

### Training Example: Qwen3.6-27B
Start training and inference on a single node using a single command:
```
specforge train --config examples/configs/qwen3.6-27b-dspark-disaggregated.yaml
```


## Draft-Model Serving Performance

Training-system throughput and draft-model serving speedup answer different questions. The H20 result above measures the efficiency of the training pipeline. The following evaluation measures the end-to-end serving speedup of a draft checkpoint trained with SpecForge; it is not used as evidence for the 10% training-runtime result.

We evaluated [Qwen3.6-27B-Domino](https://huggingface.co/Huang2020/qwen3.6-27B-domino) on 2×A100 GPUs. All values are relative to target-only autoregressive decoding (`AR = 1.00×`). `B8` and `B16` denote draft block sizes of 8 and 16.

### Concurrency = 1

| Dataset | AR | MTP-S3 | MTP-S7 | DFlash-B8 | DFlash-B16 | Domino-B8 | Domino-B16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 1.00× | 2.68× | 3.22× | 3.79× | 4.25× | 4.36× | **5.25×** |
| MATH500 | 1.00× | 2.80× | 3.55× | 4.29× | 5.07× | 4.60× | **5.72×** |
| HumanEval | 1.00× | 2.65× | 3.18× | 3.98× | 4.47× | 4.20× | **4.98×** |
| MBPP | 1.00× | 2.57× | 2.98× | 3.73× | 3.91× | 3.97× | **4.49×** |
| MT-Bench | 1.00× | 2.44× | 2.65× | 3.00× | 3.05× | 3.31× | **3.44×** |
| Alpaca | 1.00× | 2.38× | 2.54× | 2.87× | 2.84× | 3.18× | **3.34×** |

At concurrency 1, Domino-B16 provides the highest speedup on all six datasets, ranging from 3.34× on Alpaca to 5.72× on MATH500.

## More Draft Models Across Methods and Targets

A training stack is only useful if it produces checkpoints people can actually deploy. Speculative decoding offers strong theoretical guarantees and consistent gains in token acceptance rate and end-to-end speed, but adoption in the open-source community has been limited by a lack of production-ready training tooling, a scarcity of high-quality draft checkpoints, and the small scale of the data those drafts were trained on.

So alongside the runtime, a much larger set of open draft models is now available — and the striking part is how few of them we trained ourselves. Teams running SpecForge in production have trained drafters for the targets they actually serve and contributed the weights back, all trained **only on open data**. Nine of the eleven checkpoints listed below came in this way, from **Ant Group AQ**, **RadixArk**, **China Merchants Bank**, the Domino authors, and individual community members.

That inflow is what broadened the catalog along two axes:

1. **Wider target coverage** of the open-source models the community actually deploys, extending from instruct-tuned models into reasoning models and the current frontier of open-weight releases.
2. **Wider method coverage.** Following the algorithms described in the previous section, the released checkpoints now span **EAGLE3**, **DFlash**, **Domino**, and **DSpark**. Several targets in scope also ship a **native MTP** head, giving the community a chance to compare these drafters against native MTP on the same target.

If you have trained a draft model with SpecForge, we would like to host it alongside these — contributions of new targets and new algorithms are both welcome.

### Released models and performance

All checkpoints are published in the [SpecBundle collection](https://huggingface.co/collections/lmsys/specbundle) on Hugging Face:

| Target model | Draft model | Algorithm | Provider |
| --- | --- | --- | --- |
| GLM-5.1 | [🤗](https://huggingface.co/AQ-MedAI/GLM-5.1-eagle3) | EAGLE3 | Ant Group AQ |
| Kimi-K2.5 | [🤗](https://huggingface.co/AQ-MedAI/Kimi-K25-eagle3) | EAGLE3 | Ant Group AQ |
| Kimi-K2.6 | [🤗](https://huggingface.co/AQ-MedAI/Kimi-K26-eagle3) | EAGLE3 | Ant Group AQ |
| Kimi-K2.7-Code | [🤗](https://huggingface.co/AQ-MedAI/Kimi-K2.7-Code-eagle3) | EAGLE3 | Ant Group AQ |
| Qwen3-32B | [🤗](https://huggingface.co/CMBTech/CMB-Qwen3-32B-Eagle3) | EAGLE3 | China Merchants Bank |
| Qwen3.5-35B-A3B | [🤗](https://huggingface.co/jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge) | EAGLE3 | SpecForge |
| Step-3.5-Flash | [🤗](https://huggingface.co/lmsys/SGLang-EAGLE3-Step-3.5-Flash-SpecForge-RadixArk) | EAGLE3 | RadixArk |
| Qwen3.5-397B-A17B | [🤗](https://huggingface.co/lmsys/Qwen3.5-397B-A17B-DFlash) | DFlash | LMSYS |
| Qwen3.6-27B | [🤗](https://huggingface.co/Huang2020/qwen3.6-27B-domino) | Domino | Domino Team |
| Inkling-Small | [🤗](https://huggingface.co/RadixArk/Inkling-Small-DSpark-Preview) | DSpark | RadixArk |
| Kimi-K3 | [🤗](https://huggingface.co/RadixArk/Kimi-K3-DSpark) | DSpark | RadixArk |

Results for a subset of these models are shown below, grouped by algorithm.

#### EAGLE3

![EAGLE3 draft models: output throughput vs. baseline](./eagle3-speedup.svg)

*Figure 3. Three EAGLE3 draft models at the same drafting configuration (3 steps, top-k 1, 4 draft tokens): output throughput against the autoregressive baseline, with the speedup labelled above each bar. Step-3.5-Flash and Qwen3-32B were measured on 4 × H200 at concurrency 16; the Kimi-K2.7-Code numbers are the 8 × H200, concurrency-8 results published on its [model card](https://huggingface.co/AQ-MedAI/Kimi-K2.7-Code-eagle3).*

#### DFlash

![Qwen3.5-397B-A17B DFlash output throughput vs. baseline](./dflash-speedup.svg)

*Figure 4. Qwen3.5-397B-A17B on 8 × B200 (TP8, bfloat16, thinking enabled, greedy decoding, 4096 max output tokens): output throughput of DFlash at block size 8 against the autoregressive baseline. Block size 16 reaches higher still at concurrency 1 — up to 4.31× on HumanEval — while block size 8 is the stronger choice under load. Full numbers, including the MTP comparison, are on the [model card](https://huggingface.co/lmsys/Qwen3.5-397B-A17B-DFlash).*

#### Domino

![Qwen3.6-27B Domino end-to-end speedup](./domino-speedup.svg)

*Figure 5. Qwen3.6-27B on 2 × A100 (TP2, BF16, thinking enabled, greedy decoding): output throughput of Domino at block size 8 against the autoregressive baseline, with the speedup labelled above each bar. The gain is largest at concurrency 1 — up to 4.60× on MATH500 — and narrows to 1.48–2.11× at concurrency 32, where the target model is already better utilized.*

The full per-workload numbers, including the other block sizes and the MTP and DFlash comparisons, are on the [model card](https://huggingface.co/Huang2020/qwen3.6-27B-domino).

#### DSpark

![Kimi-K3 DSpark output throughput vs. baseline](./dspark-speedup.svg)

*Figure 6. Kimi-K3 on 8 × B300: output throughput of DSpark against the autoregressive baseline across five workloads, with the speedup labelled above each bar. The gain is largest at concurrency 1 — up to 3.14× on GSM8K — and narrows to 1.37–2.36× at concurrency 16, where the target model is already better utilized. MT-Bench, the most open-ended of the five, benefits least at every concurrency level. Full numbers are on the [model card](https://huggingface.co/RadixArk/Kimi-K3-DSpark).*

## Verifying Training-Serving Consistency

Model-quality benchmarks and correctness gates serve different purposes. A benchmark measures generalization; a gate checks that training, export, and serving implement the same algorithmic contract.

SpecForge provides an end-to-end [training and serving gate](https://github.com/sgl-project/SpecForge/blob/main/scripts/gates/README.md) for DFlash-family models, including Domino. It performs three stages:

1. **Select a valid sample.** The gate checks the target chat template, reasoning mode, tokenizer behavior, sequence length, and minimum trainable suffix before producing an auditable prompt artifact.
2. **Overfit through the public training path.** It repeats that sample for a bounded run launched through `specforge train`, then requires the configured loss and token-accuracy thresholds and the exact final checkpoint.
3. **Export and serve the checkpoint.** It exports through `specforge export`, launches SGLang with DFlash speculative decoding, and verifies per-request acceptance metadata and agreement with the target-token prefix.

This gate is intentionally strict and narrow. Passing it shows that capture, training, export, and serving agree on one controlled example; it does not replace held-out model-quality or serving-performance evaluation.

## What's Next

This release changes the unit of scaling in SpecForge. A run is no longer a trainer process that happens to contain a target model; it is a coordinated pipeline whose inference capacity, storage, and optimization capacity can be sized independently.

Our next steps are to finish releasing the draft models for the remaining target models above, and to continue expanding the algorithm and model catalog. We will also conduct testing and adaptation across different modalities such as VLM and additional hardware platforms, including but not limited to AMD and Ascend.

## Acknowledgements

We thank the SGLang and SpecForge communities, the authors of the supported speculative decoding methods, and all contributors who helped test the new runtime and algorithm integrations.

**SpecForge Team**: Jiaping Wang, Shenggui Li, Xiaoming Dong, Chao Wang

**RadixArk Team**: Cheng Mao, Yi Sun, and Kan Wu

**Domino Team**: Jianuo Huang

**Ant Group AQ Team**: Yefei Chen

**China Merchants Bank Team**: Peixiang Tan

**Meta/Pytorch:** Richard Zou

**Modal Team**
