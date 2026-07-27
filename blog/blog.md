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
We have use lateset Specforge to train these draft models: [Qwen3.6-27B-Domino](https://huggingface.co/Huang2020/qwen3.6-27B-domino)、[GLM-5.2-Dspark]()、[Kimi-K3-Dspark]() and so on. The partial models' performance are listed Below.

### Qwen3.6-27B-Domino（2 x A100）:
#### 1. Concurrency = 1
| Dataset | AR | MTP-S3 | MTP-S7 | DFlash-B8 | DFlash-B16 | Domino-B8 | Domino-B16 |
  |:--|--:|--:|--:|--:|--:|--:|--:|
  | GSM8K | 1.00× | 2.68× | 3.22× | 3.79× | 4.25× | 4.36× | <font color="red"><b>5.25×</b></font> |
  | MATH500 | 1.00× | 2.80× | 3.55× | 4.29× | 5.07× | 4.60× | <font color="red"><b>5.72×</b></font> |
  | HumanEval | 1.00× | 2.65× | 3.18× | 3.98× | 4.47× | 4.20× | <font color="red"><b>4.98×</b></font> |
  | MBPP | 1.00× | 2.57× | 2.98× | 3.73× | 3.91× | 3.97× | <font color="red"><b>4.49×</b></font> |
  | MT-Bench | 1.00× | 2.44× | 2.65× | 3.00× | 3.05× | 3.31× | <font color="red"><b>3.44×</b></font> |
  | Alpaca | 1.00× | 2.38× | 2.54× | 2.87× | 2.84× | 3.18× | <font color="red"><b>3.34×</b></font> |

#### 2. Concurrency = 32
  | Dataset | AR | MTP-S3 | MTP-S7 | DFlash-B8 | DFlash-B16 | Domino-B8 | Domino-B16 |
  |:--|--:|--:|--:|--:|--:|--:|--:|
  | GSM8K | 1.00× | 1.80× | 1.73× | 1.86× | 1.41× | <font color="red"><b>2.11×</b></font> | 1.82× |
  | MATH500 | 1.00× | 1.90× | 1.86× | 1.99× | 1.54× | <font color="red"><b>2.10×</b></font> | 1.78× |
  | HumanEval | 1.00× | 1.78× | 1.69× | 1.88× | 1.42× | <font color="red"><b>1.97×</b></font> | 1.62× |
  | MBPP | 1.00× | 1.72× | 1.57× | <font color="red"><b>1.73×</b></font> | 1.33× | 1.68× | 1.46× |
  | MT-Bench | 1.00× | <font color="red"><b>1.56×</b></font> | 1.32× | 1.35× | 0.94× | 1.48× | 1.06× |
  | Alpaca | 1.00× | <font color="red"><b>1.76×</b></font> | 1.43× | 1.43× | 1.00× | 1.67× | 1.13× |


### GLM5.2（）:


## How to verify your draft training is true?
We provide a test to verify the consistency of training and inference, allowing users to easily verify the correctness of algorithms and try out new algorithms. We offer detailed sample [documentation](https://github.com/sgl-project/SpecForge/blob/main/scripts/gates/README.md) for Qwen3.6-27B-Dspark. Its workflow includes the following steps:

1. Prepare a dataset generated using the target model and its corresponding thinking mode. If it's a `thinking mode`, it includes `reasoning_content`; otherwise, it doesn't.

2. Train this dataset online using `specforge` until the `accuracy` equals 1.0. Since dspark's loss consists of both `CE loss` and `L1 loss`, it cannot reach 100%. However, it should ideally reach `0.99`.

3. Deploy the trained draft model using sglang and use the trained data at `temperature=0` and the same `thinking mode` to make requests to determine if the accuracy of the first draft block is 100%.


## What's next

This release changes the unit of scaling in SpecForge. A run is no longer a trainer process that happens to contain a target model; it is a coordinated pipeline whose inference capacity, storage, and optimization capacity can be sized independently.

Our next steps are to publish the more draft models of popular models above, and continue expanding the algorithm and model catalog. Besides, we will conduct testing and adaptation across various hardware platforms, including but not limited to AMD and Ascend.

## Acknowledgements

We thank the SGLang and SpecForge communities, the authors of the supported speculative decoding methods, and all contributors who helped test the new runtime and algorithm integrations. 

**SpecForge Team**: Jiaping Wang, Shenggui Li, Xiaoming Dong,

**Radixark Team**: Cheng Mao,

**Domino**: Jianuo, Huang