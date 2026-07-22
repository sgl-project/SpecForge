# SpecForge Evolves: Disaggregated Training and a Unified Stack for Modern Speculative Decoding

**The SpecForge Team · July 2026**

When we first released [SpecForge](https://www.lmsys.org/blog/2025-07-25-spec-forge/), our goal was to make EAGLE3 draft-model training practical and directly compatible with SGLang. Since then, speculative decoding has moved quickly: target models have grown, parallel drafting has become increasingly important, and a single training recipe must often span several inference servers and several trainer workers.

Today, we are introducing a major update to SpecForge. The new runtime separates target-model inference from draft-model training, supports a broader family of speculative decoding algorithms, and unifies online, offline, and disaggregated workflows behind one typed training entry point.

## New Feature

- **Online training is now fully disaggregated.** Patched SGLang servers capture target-model features, Mooncake transports the tensors, and trainer workers consume lightweight references through a separate control plane.
- **Inference and training can scale independently.** On our 8xH20 testbed, a topology with **3 SGLang servers and 5 trainer workers** improves end-to-end training throughput by approximately **10%** over our previous colocated implementation.
- **One runtime now supports multiple drafting families:** [EAGLE3](https://arxiv.org/abs/2503.01840), **EAGLE3.1**, [P-EAGLE](https://arxiv.org/abs/2602.01469), [DFlash](https://arxiv.org/abs/2602.06036), [Domino](https://arxiv.org/abs/2605.29707), and [DSpark](https://arxiv.org/abs/2607.05147), together with the optional [D-PACE](https://arxiv.org/abs/2605.18810) objective for DFlash.

## Why Disaggregate Draft-Model Training?

Online draft-model training contains two very different workloads.

The **target side** runs a large, frozen model to generate tokens and capture hidden states. It is inference-heavy, often requires tensor parallelism, and benefits from a production inference engine. The **draft side** trains a much smaller model, performs forward and backward passes, and scales through data or sequence parallelism.

In the previous colocated design, these two workloads shared one lifecycle and one fixed resource layout. This introduced three practical limitations:

1. **A fixed inference-to-training ratio.** Adding trainer workers also affected how the target model was placed, even when only one side was the bottleneck.
2. **Resource interference.** Target-model capture and draft-model optimization competed for memory and compute within the same tightly coupled job.
3. **Synchronized stalls.** A slowdown or failure in feature generation could directly block the training process, while excess production could create unbounded memory pressure.

The key observation is simple: trainers do not need to own the target model. They only need the token sequences, masks, and target features required by the selected training objective. Once that boundary is explicit, inference and training can become independently scheduled pools.

## The New Disaggregated Architecture

The new online runtime has a producer pool and a consumer pool. The producer schedules prompts across one or more patched SGLang capture servers. SGLang writes captured tensors to Mooncake, while the producer publishes only lightweight `SampleRef` metadata. Consumer rank 0 durably records and distributes those references; each trainer then fetches the corresponding tensors directly from Mooncake.

![SpecForge online-disaggregated training architecture](./specforge-disaggregated-architecture.svg)

*Figure 1. SpecForge's online-disaggregated training flow. Large tensors stay in the data plane; the control plane carries references and lifecycle state only.*

### 1. Server-only target capture

The online trainer no longer initializes a colocated target model. Instead, every URL in `deployment.disaggregated.server_urls` creates one rollout worker connected to a patched SGLang server. The workers lease disjoint prompts from a shared controller, so adding a server increases capture capacity without changing the trainer topology.

This also establishes a clean ownership boundary: SGLang owns target-model parallelism and feature capture, while SpecForge owns prompt scheduling, reference publication, and draft-model optimization.

### 2. Tensors and metadata take different paths

Captured hidden states can be large, so forwarding them through a Python queue or a control database would quickly become a bottleneck. SpecForge instead uses a `FeatureStore` abstraction:

- SGLang writes feature tensors to Mooncake.
- The producer publishes tensor-free `SampleRef` records.
- `FeatureDataLoader` is the only component that resolves references into tensor-carrying training batches.
- Trainers release feature objects after a durable optimizer-step acknowledgement.

This split keeps the control plane small and makes the training loop independent of the storage transport. The same trainer path can consume local files, a shared directory, or Mooncake-backed features.

### 3. Optimizer-aware flow control

Distributed training ranks must advance in lockstep. Before capture starts, consumer rank 0 publishes a global dispatch quantum:

```text
quantum = world_size × per_rank_batch_size × gradient_accumulation_steps
```

References are released only in complete quantum-sized windows. Each rank therefore receives exactly the samples needed for one optimizer step, and an incomplete tail at end-of-stream is never dispatched as a partial distributed step.

The producer also observes high and low in-flight watermarks. When the number of committed but unacknowledged samples reaches the high watermark, capture pauses; it resumes only after consumption falls below the low watermark. Optional resident-byte limits provide an additional guard against feature-store pressure.

### 4. Durable recovery and failure handling

Consumer rank 0 is the only writer to the retaining SQLite ledger. At every optimizer boundary, SpecForge commits trained sample IDs, removes their feature objects, and then advances the channel counters. A crash before the transaction replays the untrained references; a crash after it skips the already committed samples.

On the producer side, a failed server returns its leased prompts to the controller, and the remaining workers continue. The run fails loudly when every capture server is unavailable or retry limits are exhausted. This makes server failures explicit without allowing them to silently corrupt the training stream.

### 5. One configuration and one training entry point

The topology is described in the same typed YAML as the model, algorithm, data, and optimizer settings:

```yaml
model:
  target_model_path: zai-org/GLM-5.2-FP8
  draft_model_config: configs/glm-5.2-dspark.json
  target_backend: sglang
  trust_remote_code: true
data:
  train_data_path: ./cache/dataset/glm52_dspark_train.jsonl
  max_length: 4096
  chat_template: glm-5.2
  cache_dir: cache
  build_dataset_num_proc: 64
training:
  strategy: dspark
  num_epochs: 10
  batch_size: 1
  accumulation_steps: 512
  learning_rate: 0.0006
  warmup_ratio: 0.04
  max_grad_norm: 1.0
  num_anchors: 512
  loss_decay_gamma: 4.0
  objective_chunk_blocks: 128
  # Optimizer-step equivalent of about 500 source microsteps at microbatch 16.
  save_interval: 16
  dist_timeout: 30
  seed: 42
run_id: glm-5.2-dspark-disaggregated
output_dir: outputs/glm-5.2-dspark-disaggregated
deployment:
  mode: disaggregated
  trainer:
    nnodes: 1
    nproc_per_node: 1
  disaggregated:
    control_dir: outputs/glm-5.2-dspark-disaggregated/control
    backend: mooncake
    server_urls: [http://127.0.0.1:30000]
    mooncake_metadata_server: http://127.0.0.1:35880/metadata
    mooncake_master_server_addr: 127.0.0.1:35551
    mooncake_protocol: rdma

```


The same public command resolves and launches the selected topology:

```bash
specforge train --config run.yaml
```

For scheduler-managed multi-node deployments, the two pools can be launched explicitly without introducing a second trainer implementation:

```bash
# Inference and ingestion pool
specforge train --config run.yaml --role producer

# Draft-model training pool
specforge train --config run.yaml --role consumer
```

## Preliminary System Result: 3 Servers + 5 Trainers on H20

We evaluated the new runtime on our H20 testbed. Reallocating the workload of Qwen3-8B Domino with 3k context length into three SGLang capture servers and five trainer workers increased end-to-end training throughput by approximately 10% compared with the previous colocated version.

| Runtime | Target capture | Draft training | Relative end-to-end training throughput |
| --- | --- | --- | ---: |
| Previous colocated version | Coupled with the training job | Fixed colocated layout | 1.00× |
| New disaggregated runtime | 3 SGLang servers | 5 trainer workers | **1.10×** |

The following GPU profiles show representative training windows for Qwen3-8B Domino at a 3k context length under the two runtime topologies.

**(a) Colocated baseline**

![Qwen3-8B Domino 3k-context colocated training profile](./colocated.png)

**(b) Disaggregated: 3 SGLang servers + 5 trainer workers**

![Qwen3-8B Domino 3k-context disaggregated training profile](./disagg.png)

*Figure 2. Qwen3-8B Domino training profiles at a 3k context length. The colocated baseline is shown above and the 3-server/5-trainer disaggregated run is shown below.*

The improvement comes from matching resources to the actual pipeline balance: target feature generation and draft optimization can progress concurrently, and neither pool needs to inherit the other's parallelism layout. Just as importantly, the new topology gives us a practical way to tune this balance for different target-model sizes, sequence lengths, and algorithms.


## From EAGLE3 to a Multi-Algorithm Training Stack

SpecForge began with a strong focus on EAGLE3. The current release turns the runtime into a common training substrate for autoregressive, parallel, block-diffusion, and semi-autoregressive drafters.

| Method | Core idea | SpecForge support |
| --- | --- | --- |
| [EAGLE3](https://arxiv.org/abs/2503.01840) | Direct token prediction with training-time test and multi-layer target-feature fusion | Online-disaggregated, colocated offline, and disaggregated offline training |
| [P-EAGLE](https://arxiv.org/abs/2602.01469) | Parallel multi-token prediction through a shared hidden state, with techniques for scalable long-context training | Online-disaggregated training |
| EAGLE3.1 | An EAGLE3 configuration variant with per-layer normalization and attention-drift settings | Online-disaggregated training through the `eagle3` strategy |
| [DFlash](https://arxiv.org/abs/2602.06036) | A lightweight block-diffusion drafter that predicts a token block in parallel while conditioning on target features | Online-disaggregated, colocated offline, and disaggregated offline training |
| [Domino](https://arxiv.org/abs/2605.29707) | A parallel draft backbone followed by a lightweight causal correction head | Online-disaggregated, colocated offline, and disaggregated offline training |
| [DSpark](https://arxiv.org/abs/2607.05147) | A semi-autoregressive drafter with confidence modeling for adaptive verification | Online-disaggregated, colocated offline, and disaggregated offline training |


## Draft Model Performance

> **This section is intentionally left as a publication-ready template.** We will replace the placeholders below after the final trained-model measurements are provided. No model-quality or serving numbers are inferred from the preliminary 10% training-throughput result above.

### Evaluation setup

| Item | Value |
| --- | --- |
| Target model(s) | **[TBD]** |
| Draft method and checkpoint(s) | **[TBD]** |
| Training dataset and sample count | **[TBD]** |
| Evaluation datasets | **[TBD]** |
| Hardware and GPU count | **[TBD]** |
| SGLang and SpecForge commits | **[TBD]** |
| Precision / quantization | **[TBD]** |
| Batch size / concurrency | **[TBD]** |
| Speculative decoding parameters | **[TBD]** |

### Draft-model quality

| Target model | Draft method | Benchmark | Average accepted length |
| --- | --- | --- | ---: |
| **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

### End-to-end serving performance

| Benchmark | Batch | Target-only | DFlash | **[Method]** | **[Method] / DFlash** |
| --- | ---: | ---: | ---: | ---: | ---: |
| **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |

For the final comparison, we will report both acceptance metrics and end-to-end serving throughput. Acceptance alone does not include draft overhead, while throughput alone can hide differences in concurrency, sequence lengths, and speculative parameters. Reporting both makes the trade-off visible and the result reproducible.


## What's next

This release changes the unit of scaling in SpecForge. A run is no longer a trainer process that happens to contain a target model; it is a coordinated pipeline whose inference capacity, storage, and optimization capacity can be sized independently.

Our next steps are to publish the full checkpoint evaluation above, add more reproducible topology studies, and continue expanding the algorithm and model catalog. We are especially interested in how different drafting families shift the optimal server-to-trainer ratio—and how the runtime can adapt that ratio automatically as workloads change.

## Acknowledgements

We thank the SGLang and SpecForge communities, the authors of the supported speculative decoding methods, and all contributors who helped test the new runtime and algorithm integrations. **[Add release-specific contributors, partner teams, and infrastructure acknowledgements before publication.]**

## References

1. SpecForge Team. [SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding](https://arxiv.org/abs/2603.18567), 2026.
2. Yuhui Li et al. [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840), 2025.
3. Mude Hui et al. [P-EAGLE: Parallel-Drafting EAGLE with Scalable Training](https://arxiv.org/abs/2602.01469), 2026.
4. Jian Chen et al. [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036), 2026.
5. Tianyu Wu et al. [D-PACE: Dynamic Position-Aware Cross-Entropy for Parallel Speculative Drafting](https://arxiv.org/abs/2605.18810), 2026.
6. Jianuo Huang et al. [Domino: Decoupling Causal Modeling from Autoregressive Drafting in Speculative Decoding](https://arxiv.org/abs/2605.29707), 2026.
7. Xin Cheng et al. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147), 2026.
