# SpecForge Documentation

The documentation site at [docs.sglang.io/SpecForge](https://docs.sglang.io/SpecForge/)
is built with [VitePress](https://vitepress.dev) from the `docs/` folder and
deployed by CI on every push to `main`. This guide covers:

1. [Running and building the docs](#1-run-and-build-the-docs)
2. [Adding a new doc page](#2-add-a-new-doc)
3. [Adding a new recipe](#3-add-a-new-recipe)
4. [Adding a model and performance results to SpecBundle](#4-add-a-model-and-performance-results-to-specbundle)

Writing documentation is a good first contribution: it is the fastest way to
learn how the SpecForge codebase fits together.

## Layout

```text
docs/
├── README.md          this guide (not published)
├── sections/          documentation pages, grouped by topic (Markdown)
│   ├── get_started/
│   ├── concepts/
│   ├── basic_usage/
│   ├── advanced_features/
│   ├── benchmarks/
│   └── examples/
├── recipes/           training recipes, one Markdown file each
│   └── index.md       the Recipes landing page (lists every recipe automatically)
├── data/              SpecBundle data, plain JSON that anyone can edit
│   ├── specbundle_models.json      draft models shown in the gallery
│   └── specbundle_benchmarks.json  results shown in the performance dashboard
└── web/               the VitePress site
    ├── .vitepress/config.mts     nav, sidebar, search, edit links, URL rewrites
    ├── .vitepress/theme/         brand CSS and Vue components (gallery, dashboard, recipe cards)
    ├── index.md                  landing page
    ├── specbundle.md             SpecBundle page: gallery + performance dashboard
    ├── data -> ../data           symlink so the site can import the JSON above
    ├── scripts/update_specbundle.py
    ├── public/                   static assets (logos), served from the site root
    └── package.json
```

URL mapping:

| Source file                                  | Published URL                     |
| -------------------------------------------- | --------------------------------- |
| `docs/sections/basic_usage/training.md`      | `/basic_usage/training.html`      |
| `docs/recipes/kimi-k3-dspark-disaggregated.md` | `/recipes/kimi-k3-dspark-disaggregated.html` |
| `docs/web/index.md`                          | `/`                               |
| `docs/web/specbundle.md`                     | `/specbundle.html`                |

Pages under `docs/sections/` lose the `sections/` prefix; everything else
keeps its path. The rewrites live in `docs/web/.vitepress/config.mts`.

## 1. Run and build the docs

Requirements: Node.js 18 or newer and npm (CI uses Node 22).

```bash
cd docs/web
npm install
```

Preview with live reload while you write:

```bash
npm run dev          # http://localhost:5173/SpecForge/
```

Build the static site and check it the way CI does:

```bash
npm run build        # output in docs/web/.vitepress/dist
npm run preview      # serve the built site locally
```

The build fails on dead internal links, so run it before opening a pull
request. Also run the repository hooks:

```bash
pre-commit run --all-files
```

<details>
<summary>The dev server fails with <code>EMFILE: too many open files, watch ...</code></summary>

The host has run out of inotify instances (common in containers, where the
default `fs.inotify.max_user_instances` is 128 and shared across sessions).
Either raise the limit:

```bash
sudo sysctl -w fs.inotify.max_user_instances=1024
```

or fall back to polling for this run:

```bash
CHOKIDAR_USEPOLLING=1 npm run dev
```

</details>

## 2. Add a new doc

1. **Create the page** as a Markdown file under the matching topic in
   `docs/sections/`, for example `docs/sections/basic_usage/my_topic.md`.
   Start it with a level-one heading; that heading is the page title.
2. **Register it in the sidebar.** Open `docs/web/.vitepress/config.mts`, find
   the `sidebar` block and add an entry to the right group using the public
   path (without `sections/` and without `.md`):

   ```ts
   {
     text: 'Basic Usage',
     items: [
       { text: 'Training', link: '/basic_usage/training' },
       { text: 'My Topic', link: '/basic_usage/my_topic' },   // new
     ],
   },
   ```

   To start a new topic, add a folder under `docs/sections/`, a new sidebar
   group, and the folder name to the `Docs` nav item's `activeMatch` pattern in
   the same file.
3. **Link correctly.** Link to other docs pages with relative Markdown links
   (`../concepts/EAGLE3.md`) and to files in the repository with full GitHub
   URLs (`https://github.com/sgl-project/SpecForge/blob/main/...`). Relative
   links into the repository do not resolve on the published site.
4. **Preview and build** as described above.

Available Markdown features: GitHub-flavored tables, `::: tip`,
`::: warning` and `::: details` containers, `::: code-group` tabs for
alternative commands, LaTeX math via `$...$`, and the Vue components
registered in `docs/web/.vitepress/theme/index.ts`. Keep headings free of
emoji so anchors stay readable.

## 3. Add a new recipe

Recipes are end-to-end reproductions of real training runs: the exact source
revisions, configs, node layout and launch commands. They live in
`docs/recipes/` and appear on the Recipes page at `/recipes/` and in its
sidebar.

1. Create `docs/recipes/<slug>.md`, where `<slug>` is lowercase with hyphens,
   for example `qwen3-8b-eagle3-online.md`.
2. Start the file with this frontmatter, then the recipe itself:

   ```md
   ---
   title: Qwen3 8B EAGLE3 Online
   description: One or two sentences on what the recipe reproduces and on what hardware.
   target: Qwen/Qwen3-8B
   method: EAGLE3
   topology: Disaggregated
   ---

   # Qwen3 8B EAGLE3 online reproduction

   ## Required source revisions
   ...
   ```

   | Field         | Used for                                                     |
   | ------------- | ------------------------------------------------------------ |
   | `title`       | card title on the Recipes page and the sidebar entry         |
   | `description` | card text                                                    |
   | `target`      | Hugging Face id of the target model, shown on the card       |
   | `method`      | `EAGLE3`, `DFlash`, `DFlash2`, `DSpark`, `Domino` or `MTP`; colors the tag |
   | `topology`    | free text such as `Disaggregated` or `Colocated`             |

3. Build or run the dev server. The card and the sidebar entry are generated
   from the folder at build time, so no config change is needed.

A good recipe states the SpecForge and SGLang revisions it was validated
with, lists the artifacts it produces, and gives one copy-pasteable command
per node.

## 4. Add a model and performance results to SpecBundle

The SpecBundle page reads two JSON files in `docs/data/`. Both are the source
of truth and can be edited directly; no Vue code needs to change.

### 4a. Add a model to the gallery

Edit `docs/data/specbundle_models.json` and append an entry to the `models`
array:

```json
{
  "id": "lmsys/SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge",
  "name": "SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge",
  "provider": { "name": "lmsys", "fullname": "LMSYS Org (SGLang)", "avatar": null, "type": "org" },
  "method": "EAGLE3",
  "target": "meta-llama/Llama-3.1-8B-Instruct",
  "dataset": "frankleeeee/PerfectBlend-Regenerated-Llama-3.1-8B-Instruct",
  "downloads": 0,
  "likes": 0,
  "numParameters": null,
  "lastModified": null,
  "gated": false
}
```

| Field                | Meaning                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| `id`                 | Hugging Face repo id of the draft model                                  |
| `name`               | the part of `id` after the slash                                         |
| `provider.name`      | Hugging Face account; `fullname` is the display name, `avatar` may be `null` |
| `method`             | `EAGLE3`, `DFlash`, `DSpark`, `Domino` or `Other`                        |
| `target`             | Hugging Face id of the model the draft was trained for                   |
| `dataset`            | regenerated training dataset on Hugging Face, or `null`                  |
| counters             | `downloads`, `likes`, `numParameters`, `lastModified` may be left at `0` / `null`; the page refreshes them from Hugging Face in the browser |

The model should also be added to the
[Hugging Face SpecBundle collection](https://huggingface.co/collections/lmsys/specbundle);
open an issue if you cannot add it yourself.

To pull models that joined the collection and refresh the counters, run:

```bash
python3 docs/web/scripts/update_specbundle.py
```

The script keeps every hand-edited `target`, `dataset`, `method` and provider
name already in the file, and keeps models that are in the file but not yet
in the collection, so it is safe to run at any time.

### 4b. Add performance results to the dashboard

Edit `docs/data/specbundle_benchmarks.json`. It is keyed by target model, then
by benchmark. Each benchmark holds one entry per speculative decoding
configuration, and each entry records the baseline run without a draft model
(`"Wihtout EAGLE3"`, spelled exactly like that) plus every draft model measured
with that configuration:

```json
{
  "Qwen3-30B-A3B-Instruct-2507": {
    "gsm8k": {
      "benchmark_name": "gsm8k",
      "results": [
        {
          "batch_size": 8,
          "steps": 3,
          "topk": 1,
          "num_draft_tokens": 4,
          "metrics": [
            { "Name": "Wihtout EAGLE3", "output_throughput": 1071.29, "accept_length": 1.0 },
            { "Name": "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge", "output_throughput": 1488.36, "accept_length": 2.64 }
          ]
        }
      ]
    }
  }
}
```

| Field                        | Meaning                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| top-level key                | target model name shown in the dashboard's target selector     |
| `benchmark_name`             | one of `gsm8k`, `math500`, `mtbench`, `humaneval`, `livecodebench`, `financeqa`, `gpqa` |
| `batch_size`, `steps`, `topk`, `num_draft_tokens` | the SGLang speculative decoding configuration; shown as `8-3-1-4` |
| `metrics[].Name`             | `Wihtout EAGLE3` for the baseline, otherwise the draft model's Hugging Face id |
| `output_throughput`          | output tokens per second                                       |
| `accept_length`              | mean acceptance length                                         |

To add results for a **new target model**, add a new top-level key with one
block per benchmark you measured. To add a **new draft model or configuration**
for an existing target, append to the `results` array of each benchmark,
always including the baseline metric so speedups can be computed. Results were
measured with SGLang on NVIDIA H200; note it in your pull request if you used
other hardware. Use `python3 -m json.tool docs/data/specbundle_benchmarks.json`
to check the file is valid JSON, then build the site and open
`/specbundle.html#performance-dashboard` to check the new rows.
