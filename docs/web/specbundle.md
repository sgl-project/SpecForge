---
title: SpecBundle
titleTemplate: SpecForge
sidebar: false
pageClass: sb-page
aside: false
outline: false
prev: false
next: false
description: Production-grade draft models for speculative decoding, trained with SpecForge by the SGLang team and industry partners.
---

<script setup>
import { withBase } from 'vitepress'
</script>

<div class="sb-hero">
  <img :src="withBase('/specbundle-logo.png')" alt="SpecBundle" class="sb-logo" width="360" />
  <p class="sb-tagline">
    <strong>Production-Grade Draft Models</strong> Built by Our Community
  </p>
  <p class="sb-tagline-sub">Trained with SpecForge. Served with SGLang.</p>
  <p class="sb-links">
    <a href="https://huggingface.co/collections/lmsys/specbundle" target="_blank" rel="noopener"><span aria-hidden="true">🤗</span> Collection on Hugging Face</a>
    <a href="#performance-dashboard"><span aria-hidden="true">📈</span> Performance Dashboard</a>
    <a href="#usage"><span aria-hidden="true">🚀</span> Usage</a>
  </p>
</div>

## Models

<ModelGallery />

## Performance Dashboard

We evaluate SpecBundle draft models on conversation (MT-Bench), general
knowledge (GPQA, FinanceQA), math (GSM8K, Math500) and coding (HumanEval,
LiveCodeBench) benchmarks under different speculative decoding configurations
(steps, top-k and number of draft tokens). Pick a target model to compare its
draft models against the baseline without speculative decoding.

<BenchmarkDashboard />

## Usage

Launch an SGLang server with a target model and its SpecBundle draft. Add
`--tp-size`, `--ep` and `--mem-fraction-static` when you run into memory limits.

::: code-group

```bash [EAGLE3]
python3 -m sglang.launch_server \
    --model <target-model-path> \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <draft-model-path> \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

```bash [DFlash / DSpark]
python3 -m sglang.launch_server \
    --model <target-model-path> \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path <draft-model-path> \
    --speculative-dflash-block-size 8
```

:::

Each model card on Hugging Face documents the exact launch flags and the
serving configuration used for its published benchmarks.

<style>
/* Every rule is prefixed with `.sb-hero` so it outranks VitePress' `.vp-doc p` / `.vp-doc a`. */
.sb-hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 24px 0 20px;
}
.sb-hero .sb-logo {
  display: block;
  width: min(360px, 80%);
  height: auto;
  margin: 0;
}
.sb-hero .sb-tagline {
  max-width: 26ch;
  margin: 28px 0 0;
  font-size: 28px;
  line-height: 1.25;
  font-weight: 700;
  letter-spacing: -0.02em;
  text-wrap: balance;
  color: var(--vp-c-text-1);
}
.sb-hero .sb-tagline strong { color: var(--vp-c-brand-1); font-weight: 700; }
.sb-hero .sb-tagline-sub {
  max-width: 44ch;
  margin: 12px 0 0;
  font-size: 17px;
  line-height: 1.55;
  text-wrap: balance;
  color: var(--vp-c-text-2);
}
.sb-hero .sb-links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  margin: 28px 0 0;
}
.sb-hero .sb-links a {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 38px;
  padding: 0 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  font-size: 14px;
  font-weight: 500;
  line-height: 1;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
  text-decoration: none;
  transition: border-color 0.2s, color 0.2s;
}
.sb-hero .sb-links a:hover { border-color: var(--vp-c-brand-1); color: var(--vp-c-brand-1); }
@media (max-width: 640px) {
  .sb-hero { padding-top: 12px; }
  .sb-hero .sb-tagline { font-size: 22px; margin-top: 22px; }
  .sb-hero .sb-tagline-sub { font-size: 15px; }
}
</style>
