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
