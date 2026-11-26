# Multi-drafter training: design

Training two drafters on the same target used to take two capture stacks, so
the target ran twice for every prompt. With a `drafters` list in the recipe,
one capture stack serves all of them: the target runs once per prompt and each
drafter gets its own sample and its own trainer.

This page explains how the feature is built: the overall architecture first,
then how the new pieces relate to each other. For the recipe fields see the
[recipe reference](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/README.md),
and for producers, consumers and capture servers see
[disaggregated training](../basic_usage/disaggregated_training.md).

## Before and after

```text
before: one stack per drafter            after: one stack for every drafter

recipe A         recipe B                recipe with drafters: [a, b]
   |                |                              |
producer A       producer B                     producer
   |                |                              |   one request per prompt
capture          capture                     capture servers
servers A        servers B                         |   target runs once,
   |                |                              |   one sample per drafter
trainer A        trainer B                   +-----+-----+
                                             |           |
2 target passes per prompt               trainer a   trainer b
```

Trainers stay independent: each has its own GPUs, optimizer and checkpoints.
Only the capture is shared.

## Architecture

### Design rules

1. A drafter is a plain single-draft run. The recipe is turned into one
   ordinary `Config` per drafter, so trainer, checkpoint and export code do not
   change.
2. A single-draft recipe is a list of one run. It is converted once, at the
   entry, and the code below never asks "one drafter or several".
3. Only four places know there are several drafters: the recipe, the launcher,
   the capture request, and the ref channels.
4. One target pass per prompt, one stored sample per drafter. A trainer frees
   its own samples, so drafters never coordinate a delete.

### Layers

Every layer works on a list of runs; with one drafter the list has one entry.
The pull requests land bottom-up, so the `drafters` field arrives last and the
recipe never describes a run the code cannot start.

```text
+--------------------------------------------------------------------+
| Recipe and entry                                              PR 4 |
|   DrafterConfig, flatten_drafters, resolve_runs, --drafter         |
|   turns multi.yaml into runs: one single-draft Config per drafter  |
+--------------------------------------------------------------------+
| Launcher                                                      PR 3 |
|   build_launch_plan(cfg, runs), combine_capture_contracts          |
|   starts the capture servers, one producer, one trainer per run    |
+--------------------------------------------------------------------+
| Producer                                                      PR 2 |
|   DrafterChannels, _build_online_producer(runs)                    |
|   one capture output and one ref channel per run                   |
+--------------------------------------------------------------------+
| Capture protocol                                              PR 1 |
|   CaptureOutput, SGLangServerCaptureAdapter(outputs), plugin sink  |
|   one request per prompt writes one sample per output              |
+--------------------------------------------------------------------+
| Trainer                                                  unchanged |
|   reads its refs, fetches and frees its own samples                |
+--------------------------------------------------------------------+
```

### The path of one prompt

The same eight stages exist with one drafter or several.

| # | Stage | One drafter | Several drafters | PR |
| --- | --- | --- | --- | --- |
| 1 | Recipe | top-level `draft` | `drafters` list; `model`, `data`, `runtime` and capture servers shared | 4 |
| 2 | Resolve | one run | `flatten_drafters` gives one `Config` per drafter, `resolve_runs` resolves each | 4 |
| 3 | Launch | producer + trainer | producer + one trainer per run; servers capture the union of the layers | 3 |
| 4 | Assemble the producer | one output, one channel | one `CaptureOutput` and one ref channel per run | 2 |
| 5 | Capture | one sample | one request lists `outputs`; the plugin cuts each output's layers and writes all in one batch put | 1 |
| 6 | Return refs | one ref | one verified ref per output; one bad output frees them all | 1 |
| 7 | Deliver refs, set the pace | one channel | `DrafterChannels` routes each ref to its run and combines the trainers' progress | 2 |
| 8 | Train | trainer | unchanged, one per drafter | - |

## How the new pieces relate

The four parts below follow one recipe from the YAML down to a stored sample.
They use this example:

```yaml
# model and data as usual
run_id: qwen3
output_dir: ./out
deployment:
  mode: disaggregated
  disaggregated:
    control_dir: /ctl
    backend: mooncake
    managed_local: {capture_servers: [{port: 30000, cuda_visible_devices: ["0"]}]}
drafters:
  - name: dflash
    draft: specforge/config/draft/presets/qwen3-8b-dflash.yaml
    training: {learning_rate: 5.0e-4}
    trainer: {cuda_visible_devices: ["1", "2"]}
  - name: dspark
    draft: specforge/config/draft/presets/qwen3-8b-dspark.yaml
    trainer: {cuda_visible_devices: ["3"]}
```

### From the recipe to runs (PR 4)

```text
multi.yaml
   |  load_config
   v
Config
  drafters: [DrafterConfig dflash, DrafterConfig dspark]
  model, data, runtime, deployment          shared by every drafter
   |
   |  resolve_runs(cfg)  calls  flatten_drafters(cfg)
   v
for each DrafterConfig:
     the shared recipe
   + its own draft, initialization, vocab_mapping_path and training keys
   + derived names    run_id qwen3-<name>   output_dir ./out/<name>
                      control_dir /ctl/<name>   store id qwen3 (shared)
   = one plain single-draft Config, validated on its own
   |
   v
runs = (ResolvedRun dflash, ResolvedRun dspark)     each: config, algorithm, name
```

The recipe is checked twice. Loading it checks what drafters share or must
keep apart, such as names and GPUs. Flattening validates each drafter's own
config, so an error there names its drafter. A recipe without `drafters` goes
through the same calls and gives one unnamed run.

Every process loads the same recipe and asks for the runs it needs:

```text
specforge train -c multi.yaml                             every run -> launch plan
specforge train -c multi.yaml --role producer             every run -> producer
specforge train -c multi.yaml --role consumer --drafter dspark    one run -> trainer
```

### From runs to processes (PR 3)

```text
build_launch_plan(cfg, runs)

cfg, the recipe as loaded              runs, one per drafter
        |                                     |
        v                                     v
the shared stack                       one trainer command each
  Mooncake                               its own ref channel and database
  capture servers <----+                 its own GPUs
  one producer command |                 --role consumer --drafter NAME
                       |
       combine_capture_contracts(the contract of each run)
         dflash: layers (1, 9)  \
                                 +-->  servers capture layers (1, 5, 9),
         dspark: layers (1, 5)  /      with the first run's capture method
```

On a `managed_local` stack one command starts all of it. Elsewhere the roles
are started by hand with the three commands above, against capture servers
that capture every drafter's layers.

### Inside the producer (PR 2)

```text
_build_online_producer(runs)

for each run:
  ref_channel_path(run.config)   --> StreamingRefChannel --+
                                                           +--> DrafterChannels
  capture_output_for(run.config) --> CaptureOutput --+          {run_id: channel}
                                                     |
                                                     +--> SGLangServerCaptureAdapter
                                                          (outputs), one per server

drive loop, the same code as with one drafter:
  adapter.produce_refs(prompts)   -->  one ref per output
  channels.begin_publish(refs)    -->  each ref goes to the channel of its run_id
```

The drive loop still talks to one channel. `DrafterChannels` spreads each call
over the run's channels and gives back one answer:

```text
the drive loop asks      DrafterChannels                     each run's channel

consumer_quantum()  -->  wait for all, take the largest  <--  its trainer's quantum
in_flight_remote()  -->  the fullest running channel     <--  refs not read yet
begin_publish(refs) -->  group the refs by run_id        -->  one transaction each
consumer_stopped()  -->  all stopped, or one failed      <--  done or failed
close()             -->  refuse if a trainer failed      -->  close
```

Capture pauses on the fullest channel. For example, after 30 prompts with
`dflash` having read 20 and `dspark` 27, the unread count is 10, not 3. All
trainers are therefore fed at the pace of the slowest one. A failed publish
frees only the refs that were not handed over, and after a trainer fails no
healthy trainer is told its data is complete.

### One capture request (PR 1)

```text
producer: SGLangServerCaptureAdapter           capture server: plugin

produce_refs(tasks)
  _request_spec(task)
    one CaptureOutput: the flat spec, as before
    several: {"outputs": [one per CaptureOutput]}
                        ---- POST /generate ---->
                                               request_spec     checks `outputs`
                                               target forward   once per prompt
                                               SpecCaptureSink.put_samples
                                                 for each of spec_outputs(spec):
                                                   cut its layers from the capture
                                                   stage it under its sample id
                                                 one batch put, all or nothing
                        <------- result ---------
  _result_refs(task, result)
    one ref per output, checked against
    that output's own CaptureConfig
    one bad output: free every sample of the prompt
```

The request for prompt `t42`:

```text
{"store_id": "qwen3", "gen": 1, "replace": false,
 "outputs": [
   {"sample_id": "qwen3-dflash:t42", "aux_layer_ids": [1, 9], "features": {...}},
   {"sample_id": "qwen3-dspark:t42", "aux_layer_ids": [1, 5], "features": {...}}]}
```

The plugin cuts each output's columns from the captured layers. Both lists
must ascend, because the model concatenates captured layers in that order.

```text
servers capture layers   1   5   9      stored as
dflash wants (1, 9)      x       x      qwen3/qwen3-dflash:t42/...
dspark wants (1, 5)      x   x          qwen3/qwen3-dspark:t42/...
```

Each trainer then reads its refs from `/ctl/<name>/refs.jsonl`, fetches its
samples from the store and frees them, exactly as a single-draft trainer does.

## How it was tested

Besides the CPU tests in each PR, the whole stack ran once on one node with
four B200 GPUs: Qwen3-8B as the target, SGLang at the CI pin, and three
drafters from the checked-in presets. They want different layers, so the
servers capture the union:

```text
eagle3   (1,    17,     32    )   its own vocab_mapping_path
dflash   (1, 9, 17, 25,     33)
dspark   (1, 9, 17, 25,     33)
servers  (1, 9, 17, 25, 32, 33)
```

Capture. The same four prompts went through the producer's own adapter to a
shared server and to one server per drafter set up as that drafter alone:

```text
GPU 0  shared   union layers, run once with each method: EAGLE3, DFLASH, DSPARK
GPU 1  eagle3   its own layers and method        \
GPU 2  dflash   its own layers and method         >  the reference
GPU 3  dspark   its own layers and method        /
```

With every method, each drafter's stored tensors were bitwise equal to its
reference (132 tensors). On Qwen3 the server's capture method does not change
the data, and the plugin cuts the right columns.

Training. One `specforge train -c` on the three-drafter recipe started
Mooncake, one capture server (GPU 0, method EAGLE3 from the first drafter), the
producer and one trainer per drafter (GPUs 1 to 3). Each drafter trained 100
steps at batch size 1 and saved its checkpoint; the producer captured 360
prompts once each and published 1,080 samples with no failure, and every
process exited. Cross-entropy fell for all three between steps 5 and 100:
EAGLE3 27.2 to 6.6, DFlash 20.4 to 6.1, DSpark 35.1 to 16.2. With three
drafters the store held about 30 GB at the default watermarks.

Not tested yet: acceptance length after a full run, a long run against solo
runs, targets other than Qwen3, and drafters whose trainers use several GPUs.

## Capture speed

Several drafters share one target forward but add bytes to every write. This
section says when that slows capture down. It only matters when capture is the
bottleneck, that is when every drafter alone could train faster than capture
produces samples. Otherwise the slowest trainer sets the pace and capture idles
between refills.

Per capture server:

| Name | Meaning |
| --- | --- |
| `b` | samples per request, `runtime.producer_lease` (default 8) |
| `c` | requests the producer keeps in flight, `runtime.producer_concurrency` (default 1) |
| `F` | GPU time of one target forward over one request |
| `W` | time the plugin's single writer thread takes to store one request's batch: staging plus `batch_put_from` (`total_ms` with `SGLANG_SPEC_CAPTURE_TIMING=1`) |

A second drafter leaves `F` alone, since the target runs once and computes every
layer anyway. It raises `W`, since each prompt now stores every drafter's
layers.

```text
                         too few requests    GPU-bound    writer-bound
samples/s  =  b  x  min(   c / (F + W),        1 / F,       1 / W   )
```

```text
1. Too few requests: c x max(F, W) < F + W, which means c = 1.
   The GPU waits for every write and for the next request.

   GPU     [    F    ]         [    F    ]
   writer             [  W  ]             [  W  ]

2. GPU-bound: c large enough and F >= W.
   Each write ends before the next forward does.

   GPU     [    F    ][    F    ][    F    ][    F    ]
   writer             [  W  ]    [  W  ]    [  W  ]

3. Writer-bound: c large enough and W > F.
   At most two batches wait for the writer and the scheduler stalls on a
   third, so the GPU waits.

   GPU     [  F  ]    [  F  ]    [  F  ]
   writer         [    W    ][    W    ][    W    ]
```

Measured on Qwen3-8B with `b = 8` and about 930 tokens per prompt, as the
capture speed during refills:

```text
samples/s                          c=1     c=2     c=3     c=4     c=8
B200, dspark                       63.2    76.5      -     81.8    83.6
B200, dflash + dspark              52.3    73.0      -     80.1    83.6
H200, dflash                       34.5    42.3      -     43.8      -
H200, dflash + dspark              26.5      -       -     42.0      -
B300, dflash + dspark, memcpy      33.5    55.0    59.7    59.2      -
B300, dflash + dspark, TCP x16     44.3    72.0    81.2    80.4      -

                     F        W, one drafter    W, dflash + dspark
B200                 98 ms    25 ms             45 ms
H200                183 ms    50 ms            110 ms
B300, memcpy           -        -              132 ms
B300, TCP x16       100 ms      -               71 ms
```

`W` is measured; `F` is `b` divided by the speed at `c = 4` where `F > W`.
`memcpy` and `TCP x16` are two Mooncake settings, explained below.

- Case 1 is `c = 1`, the default. Every write adds to every request, so a
  second drafter costs directly: -17% on B200 and -23% on H200.
- Case 2 covers every setup above at `c >= 4` except B300 with `memcpy`.
  `F > W` even with two drafters, so capture stays bound by the target forward
  and a second drafter costs almost nothing: -2% at `c = 4` and 0% at `c = 8`
  on B200.
- Case 3 starts when `W` catches up with `F`. `W` grows with the drafters' total
  captured width and with a slow host; `F` shrinks with a faster GPU or a
  smaller target. The B300 host with `memcpy` is an example: `W = 132 ms`
  against `F = 100 ms`, so capture stops at `b / W`, about 61 samples/s, from
  `c = 3` on, and a larger `c` does not help.

Both `F` and `W` grow with the tokens per prompt, so the case depends on the
target, the drafters and the hardware, but hardly on prompt length.

### `W` depends on the host

When only TCP is available, Mooncake (0.3.14) does not send the local write
over TCP. It copies the batch into the store with `memcpy` on a single thread
(`kDefaultMemcpyWorkers = 1`, not configurable), and one `batch_put_from` is one
copy task. `W` is therefore set by how fast one CPU core copies memory: about
25 GB/s on the B200 host and 7-8 GB/s on the H200 and B300 hosts. More plugin
writer threads would only queue on that same thread.

On a slow host, turning the copy off and giving TCP more threads spreads the
copy over several cores:

```text
B300, dflash + dspark, c = 4                         put         W        samples/s
memcpy: the default                                   7.1 GB/s   132 ms   59.2
MC_STORE_MEMCPY=0, MC_NUM_TCP_IO_THREADS=8           10.6 GB/s   102 ms   75.0
TCP x16: MC_STORE_MEMCPY=0, MC_NUM_TCP_IO_THREADS=16,
         MC_TCP_LANES_PER_PEER=16                    23.7 GB/s    71 ms   80.4
```

This moved the B300 host from case 3 to case 2. The variables were set for the
whole `specforge train` run, and the trainers' fetch time did not change.

### SGLang answers only between forwards

The writer runs in the background, but SGLang sends each held response from its
scheduler thread, and that thread checks held responses only between batches
(`process_batch_result` and `on_idle`). While a forward runs, the scheduler
waits for it. A request whose write ends during another request's forward is
therefore answered when that forward ends, not when its write ends:

```text
c = 2, F > W           _ = the answer travels back and the next request arrives

GPU        [   C   ]_[   D   ]_[   E   ]_
request C  [   C   ][WC]......> answered when D ends, then E is sent
request D           _[   D   ][WD]......> answered when E ends
```

Each request goes around in about `2F`, and the GPU idles for one short return
trip after every forward. With `c >= 3` another request is always queued when a
forward ends, so the gap closes. At `c = 1` the scheduler is idle and polls, so
a request is answered as soon as its write ends.

This was timed on the B300 host with TCP x16, where `F > W`, using millisecond
logs and a test build of SGLang that also releases held responses while it
waits for the GPU:

```text
dflash + dspark            samples/s   write ends during the next forward:
                                       write done -> answered
c = 2, SGLang              72.0        median 36 ms, 93% within 3 ms after that forward
c = 2, released early      77.0        median 0 ms
c = 3, SGLang              81.2
c = 4, SGLang              80.4
```

The gap costs about 7% at `c = 2`, which also explains why `c = 2` ran 6-9%
below `c = 4` on B200. Releasing early recovers most of it, and `c = 3` all of
it.

### Writing straight from the GPU

Over TCP the plugin first copies captures to pinned host memory, and Mooncake
then copies them into the store on one CPU thread. Mooncake can also take GPU
pointers: its copy thread then calls `cudaMemcpy` device to host, and if the
store segment is pinned, the GPU's copy engine writes into it directly.

```text
default:     GPU -> pinned buffer -> memcpy, one CPU thread -> segment
GPU-direct:  GPU -> cudaMemcpy, DMA over PCIe -> pinned segment
```

This was tried on the B300 host with a one-line plugin change that lets
`SGLANG_SPEC_CAPTURE_GPU_PUT=1` run over TCP, plus
`MC_STORE_PIN_MEMORY_MAX_BYTES` above the segment size, so Mooncake pins the
segment with `cudaHostRegister`:

```text
B300, samples/s                  c=2     c=4     W at c=4
dflash + dspark, memcpy          55.0    59.2    132 ms
dflash + dspark, TCP x16         72.0    80.4     71 ms
dflash + dspark, GPU-direct      77.1    84.9     61 ms
dspark, memcpy                   74.7    83.1     75 ms
dspark, GPU-direct               87.4    88.2     41 ms
```

- At `c = 4` capture reaches the forward limit with one or two drafters. The
  copy itself takes about 17 ms of the 61; most of the rest is Mooncake
  registering and unregistering the batch's buffers, since each call sends the
  client's whole buffer list to the metadata server.
- Costs: pinning the 48 GiB segment takes about 15 s at the first write and keeps
  that host memory locked, and captures stay on the GPU until written (at most
  1.4 GiB more). Trainer speed and loss did not change.
- With one drafter, `c = 2` matched `c = 4`. The scheduler checks held responses
  once per forward, about 34 ms after the previous forward ends, right after it
  launches the next prefill. A 29 ms write makes that check and is released
  mid-forward, so the SGLang gap closes. This is a race, not a fix: with two
  drafters the write takes 56 ms and the gap stays.

The recipe refuses `gpu_put` over TCP, so the test set the variables by hand.
Supporting it needs the plugin rule limited to a local segment with `memcpy`,
the config check relaxed, and the launcher setting the pin size.

### Choosing `c`

`c` is `runtime.producer_concurrency`, set per rollout worker. Use `c = 3` or
`4` for a capture-bound run: both cover `W` and the SGLang gap (`c = 3` matched
`c = 4` on B300), and `c = 4` reached 96-98% of `c = 8` on B200. More does not
help once the GPU is busy, because
`max_prefill_tokens` (16384) fits about two requests in one forward. It only
makes each request wait longer (0.37 s to 0.64 s with two drafters from `c = 4`
to `c = 8`) and lets the store pass the high watermark by up to `c x b`
samples.

## Limits of this version

- Online disaggregated training only. No resume, inbox relay, NVLink or
  non-text input, and each drafter's trainer runs on one node.
- Prompts are prepared with the first drafter's rules, so drafters should
  accept the same prompts. The DFlash and DSpark families do.
- Drafters should stop on the same prompt, for example with equal
  `max_steps` and batch size. A drafter that stops early keeps getting samples,
  and they are freed only when the run ends.
- One failed trainer ends the run for every drafter.
- The capture servers need a plugin that understands `outputs`.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| One producer per drafter, with a broker that merges equal requests | The broker waits for every producer on each prompt, so a slow or dead one blocks the rest, and it is one more service |
| One trainer process for all drafters | Ties GPUs, parallelism and checkpoints together inside the most complex component |
| One stored copy of the layer union, shared by the drafters | About half the store memory when layers overlap, but it needs a delete shared across trainers |
