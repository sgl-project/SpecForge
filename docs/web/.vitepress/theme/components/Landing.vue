<script setup lang="ts">
/**
 * SpecForge landing page. Layout follows the sglang.io homepage structure
 * (hero, three feature columns, "get started" configurator band,
 * support tiles, community cards) using the SpecForge logo and blue palette.
 */
import { withBase } from 'vitepress'
import InstallSelector from './InstallSelector.vue'
import snapshot from '../../../data/specbundle_models.json'

const REPO = 'https://github.com/sgl-project/SpecForge'
const SLACK = 'https://sgl-fru7574.slack.com/archives/C09784E3EN6'
const HF = 'https://huggingface.co/collections/lmsys/specbundle'

const modelCount = snapshot.models.length
const providerCount = new Set(snapshot.models.map((m) => m.provider.name)).size

const features = [
  {
    title: 'SGLang-ready',
    icon: 'M13 3 4 14h6l-1 7 9-11h-6l1-7Z',
    text: 'Draft models trained with SpecForge export straight into SGLang serving. Most methods use specforge export; MTP merges via scripts/merge_mtp_to_base.py.',
  },
  {
    title: 'Every draft family',
    icon: 'M4 6h16M4 12h10M4 18h7M19 15v6m-3-3h6',
    text: 'EAGLE3, P-EAGLE, DFlash, DFlash2, DSpark, Domino and MTP drafts all train through one runtime with shared data, evaluation and export.',
  },
  {
    title: 'Any scale, any accelerator',
    icon: 'M4 4h16v16H4zM9 9h6v6H9zM2 9h2M2 15h2M20 9h2M20 15h2M9 2v2M15 2v2M9 20v2M15 20v2',
    text: 'Online and offline training, colocated or disaggregated, from a single GPU to multi-server capture on NVIDIA CUDA, AMD ROCm and Ascend NPU.',
  },
]

const steps = [
  { title: 'Install SpecForge', text: 'From source or PyPI, on the accelerator you have.', link: '/get_started/installation' },
  { title: 'Prepare a dataset', text: 'Regenerate responses with the target model so the draft learns its distribution.', link: '/basic_usage/data_preparation' },
  { title: 'Train the draft', text: 'Point <code>specforge train</code> at a YAML recipe; capture hidden states online or offline.', link: '/basic_usage/training' },
  { title: 'Export and serve', text: 'Run <code>specforge export</code> and launch SGLang with the draft for faster decoding.', link: '/get_started/about' },
]

const methods = ['EAGLE3', 'P-EAGLE', 'DFlash', 'DFlash2', 'DSpark', 'Domino', 'MTP']
const hardware = ['NVIDIA GPUs', 'AMD GPUs', 'Ascend NPUs']
const families = ['Llama', 'Qwen', 'Kimi', 'DeepSeek', 'GLM', 'gpt-oss', 'Step', 'Ling', 'Inkling']

const community = [
  {
    title: 'GitHub',
    text: 'Report bugs, request features and contribute code.',
    href: REPO,
    icon: 'M12 2a10 10 0 0 0-3.2 19.5c.5.1.7-.2.7-.5v-1.8c-2.8.6-3.4-1.2-3.4-1.2-.4-1.1-1.1-1.4-1.1-1.4-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.6 2.4 1.1 3 .9.1-.7.4-1.1.6-1.4-2.2-.2-4.6-1.1-4.6-4.9 0-1.1.4-2 1-2.7-.1-.3-.4-1.3.1-2.7 0 0 .8-.3 2.8 1a9.5 9.5 0 0 1 5 0c1.9-1.3 2.7-1 2.7-1 .5 1.4.2 2.4.1 2.7.6.7 1 1.6 1 2.7 0 3.8-2.3 4.7-4.6 4.9.4.3.7 1 .7 1.9v2.8c0 .3.2.6.7.5A10 10 0 0 0 12 2Z',
  },
  {
    title: 'Slack',
    text: 'Chat with the SGLang and SpecForge maintainers in real time.',
    href: SLACK,
    icon: 'M6 15a2 2 0 1 1-2-2h2v2Zm1 0a2 2 0 0 1 4 0v5a2 2 0 0 1-4 0v-5Zm2-9a2 2 0 1 1 2-2v2H9Zm0 1a2 2 0 0 1 0 4H4a2 2 0 0 1 0-4h5Zm9 2a2 2 0 1 1 2 2h-2V9Zm-1 0a2 2 0 0 1-4 0V4a2 2 0 0 1 4 0v5Zm-2 9a2 2 0 1 1-2 2v-2h2Zm0-1a2 2 0 0 1 0-4h5a2 2 0 0 1 0 4h-5Z',
  },
  {
    title: 'Hugging Face',
    text: `Download ${modelCount} SpecBundle draft models from ${providerCount} providers.`,
    href: HF,
    icon: 'M12 2.5a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-3.4 6.2a1 1 0 1 1 0 2 1 1 0 0 1 0-2Zm6.8 0a1 1 0 1 1 0 2 1 1 0 0 1 0-2ZM7.8 12.6c.4-.3.9-.1 1.6.3 1.2.7 4 .7 5.2 0 .7-.4 1.2-.6 1.6-.3.3.3.2.8-.1 1.4a4.9 4.9 0 0 1-8.2 0c-.3-.6-.4-1.1-.1-1.4Z',
  },
  {
    title: 'Dashboard',
    text: 'Compare acceptance length and speedup across benchmarks.',
    href: withBase('/specbundle.html#performance-dashboard'),
    icon: 'M4 20V10m5 10V4m5 16v-7m5 7V8',
    internal: true,
  },
]

const HF_ICON =
  'M12 2.5a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-3.4 6.2a1 1 0 1 1 0 2 1 1 0 0 1 0-2Zm6.8 0a1 1 0 1 1 0 2 1 1 0 0 1 0-2ZM7.8 12.6c.4-.3.9-.1 1.6.3 1.2.7 4 .7 5.2 0 .7-.4 1.2-.6 1.6-.3.3.3.2.8-.1 1.4a4.9 4.9 0 0 1-8.2 0c-.3-.6-.4-1.1-.1-1.4Z'

const p = (path: string) => withBase(path + '.html')
</script>

<template>
  <div class="sf">
    <!-- Hero -->
    <section class="sf-hero sf-band">
      <div class="sf-wrap sf-hero-grid">
        <div class="sf-hero-copy">
          <h1 class="sf-h1">
            <span class="sf-accent">Forge Draft Models</span> for Speculative Decoding
          </h1>
          <p class="sf-sub">
            SpecForge is the SGLang-native framework for training EAGLE3, DFlash, DSpark and Domino
            draft models. Train once, export once, serve at speed.
          </p>
          <div class="sf-actions">
            <a class="sf-btn sf-btn-brand" :href="p('/get_started/installation')">Get Started <span aria-hidden="true">→</span></a>
            <a class="sf-btn sf-btn-alt" :href="p('/specbundle')">Browse SpecBundle</a>
          </div>
        </div>
        <div class="sf-hero-art" aria-hidden="true">
          <div class="sf-hero-glow"></div>
          <img :src="withBase('/logo-mark.png')" alt="" width="360" height="458" />
        </div>
      </div>

      <div class="sf-wrap sf-features">
        <div v-for="f in features" :key="f.title" class="sf-feature">
          <h3>
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" :d="f.icon" /></svg>
            {{ f.title }}
          </h3>
          <p>{{ f.text }}</p>
        </div>
      </div>
    </section>

    <!-- Get started -->
    <section class="sf-band sf-band-soft" id="get-started">
      <div class="sf-wrap sf-start-grid">
        <div>
          <h2 class="sf-h2">Get Started in Seconds</h2>
          <p class="sf-lead">
            Select your hardware and installer, run the command, and you are ready to train your
            first draft model.
          </p>
          <ol class="sf-steps">
            <li v-for="(s, i) in steps" :key="s.title">
              <span class="sf-step-num" aria-hidden="true">{{ i + 1 }}</span>
              <div>
                <a class="sf-step-title" :href="p(s.link)">{{ s.title }}</a>
                <p v-html="s.text"></p>
              </div>
            </li>
          </ol>
          <a class="sf-link" :href="p('/get_started/about')">View Documentation →</a>
        </div>
        <InstallSelector />
      </div>
    </section>

    <!-- Support -->
    <section class="sf-band">
      <div class="sf-wrap sf-center">
        <h2 class="sf-h2">Broad Method &amp; Hardware Support</h2>
        <p class="sf-lead sf-lead-center">One training runtime across draft-model families and accelerators.</p>
        <div class="sf-support-grid">
          <div class="sf-support">
            <h3>
              <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M4 6h16M4 12h10M4 18h7M19 15v6m-3-3h6"/></svg>
              Draft Model Families
            </h3>
            <div class="sf-tiles">
              <span v-for="m in methods" :key="m" class="sf-tile">{{ m }}</span>
            </div>
            <a class="sf-link" :href="p('/basic_usage/training')">See the training guide →</a>
          </div>
          <div class="sf-support">
            <h3>
              <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M4 4h16v16H4zM9 9h6v6H9zM2 9h2M2 15h2M20 9h2M20 15h2M9 2v2M15 2v2M9 20v2M15 20v2"/></svg>
              Supported Hardware
            </h3>
            <div class="sf-tiles">
              <span v-for="h in hardware" :key="h" class="sf-tile">{{ h }}</span>
            </div>
            <a class="sf-link" :href="p('/get_started/installation')">See accelerator setup →</a>
          </div>
        </div>
      </div>
    </section>

    <!-- SpecBundle -->
    <section class="sf-band sf-band-soft">
      <div class="sf-wrap sf-bundle">
        <div class="sf-bundle-copy">
          <img :src="withBase('/specbundle-logo.png')" alt="SpecBundle" class="sf-bundle-logo" width="240" loading="lazy" />
          <h2 class="sf-h2">Production-grade draft models, ready to serve</h2>
          <p class="sf-lead">
            SpecBundle collects {{ modelCount }} draft models for mainstream open LLMs, contributed by the
            SGLang team and industry partners including Ant Group, Meituan, Nex-AGI, EigenAI and RadixArk.
            Every model ships with SGLang launch flags and benchmark results.
          </p>
          <div class="sf-actions">
            <a class="sf-btn sf-btn-brand" :href="p('/specbundle')">Browse {{ modelCount }} models <span aria-hidden="true">→</span></a>
            <a class="sf-btn sf-btn-alt" :href="HF" target="_blank" rel="noopener">
              <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" :d="HF_ICON" /></svg>
              Hugging Face collection
            </a>
          </div>
        </div>
        <div class="sf-tiles sf-tiles-wide">
          <span v-for="f in families" :key="f" class="sf-tile">{{ f }}</span>
        </div>
      </div>
    </section>

    <!-- Community -->
    <section class="sf-band">
      <div class="sf-wrap sf-center">
        <h2 class="sf-h2">Join the Community</h2>
        <p class="sf-lead sf-lead-center">
          From a first training run to production draft models, the SpecForge community is open to everyone.
        </p>
        <div class="sf-community">
          <a
            v-for="c in community"
            :key="c.title"
            class="sf-card"
            :href="c.href"
            :target="c.internal ? undefined : '_blank'"
            :rel="c.internal ? undefined : 'noopener'"
          >
            <span class="sf-card-icon">
              <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" :d="c.icon" /></svg>
            </span>
            <strong>{{ c.title }}</strong>
            <p>{{ c.text }}</p>
          </a>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.sf {
  --sf-max: 1152px;
  color: var(--vp-c-text-1);
}
.sf-wrap {
  max-width: var(--sf-max);
  margin: 0 auto;
  padding: 0 24px;
}
@media (min-width: 640px) { .sf-wrap { padding: 0 48px; } }
@media (min-width: 960px) { .sf-wrap { padding: 0 64px; } }

.sf-band { padding: 72px 0; background: var(--vp-c-bg); }
.sf-band-soft { background: var(--vp-c-bg-soft); border-top: 1px solid var(--vp-c-divider); border-bottom: 1px solid var(--vp-c-divider); }
.sf-band + .sf-band:not(.sf-band-soft) { border-top: 1px solid var(--vp-c-divider); }
@media (min-width: 960px) { .sf-band { padding: 88px 0; } }

.sf-center { text-align: center; }
.sf-h1 {
  margin: 0;
  font-size: 40px;
  line-height: 1.12;
  font-weight: 800;
  letter-spacing: -0.03em;
}
@media (min-width: 640px) { .sf-h1 { font-size: 52px; } }
@media (min-width: 960px) { .sf-h1 { font-size: 58px; } }
.sf-accent { color: var(--vp-c-brand-1); }
.sf-h1, .sf-h2 { text-wrap: balance; }
.sf-sub, .sf-lead, .sf-feature p, .sf-steps p, .sf-card p { text-wrap: pretty; }
.sf-h2 {
  margin: 0;
  font-size: 32px;
  line-height: 1.2;
  font-weight: 800;
  letter-spacing: -0.025em;
}
@media (min-width: 640px) { .sf-h2 { font-size: 38px; } }
.sf-sub, .sf-lead {
  margin: 20px 0 0;
  font-size: 18px;
  line-height: 1.6;
  color: var(--vp-c-text-2);
  max-width: 560px;
}
.sf-lead { font-size: 17px; margin-top: 14px; }
.sf-lead-center { margin-left: auto; margin-right: auto; max-width: 620px; }

.sf-actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 28px; }
.sf-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 11px 22px;
  border-radius: 10px;
  font-size: 15px;
  font-weight: 600;
  text-decoration: none;
  transition: transform 0.15s, box-shadow 0.15s, background 0.15s, border-color 0.15s;
}
.sf-btn-brand {
  color: #fff;
  background: var(--vp-c-brand-1);
  box-shadow: 0 6px 18px rgba(31, 127, 192, 0.3);
}
.sf-btn-brand:hover { background: var(--vp-c-brand-2); transform: translateY(-1px); }
.sf-btn:active { transform: translateY(0); }
.sf-btn:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 3px; }
.sf-btn-alt {
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
}
.sf-btn-alt:hover { border-color: var(--vp-c-brand-1); color: var(--vp-c-brand-1); }

.sf-link { display: inline-block; margin-top: 20px; color: var(--vp-c-brand-1); font-weight: 600; text-decoration: none; }
.sf-link:hover { text-decoration: underline; }

/* Hero */
.sf-hero { padding-top: 56px; }
.sf-hero-grid {
  display: grid;
  gap: 40px;
  align-items: center;
}
.sf-hero-grid > *, .sf-start-grid > *, .sf-bundle > * { min-width: 0; }
@media (min-width: 960px) {
  .sf-hero-grid { grid-template-columns: 1.15fr 0.85fr; gap: 48px; }
}
.sf-hero-art {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 300px;
}
.sf-hero-art img {
  position: relative;
  width: min(360px, 70vw);
  height: auto;
  filter: drop-shadow(0 18px 30px rgba(31, 127, 192, 0.25));
}
.sf-hero-glow {
  position: absolute;
  inset: 10%;
  border-radius: 50%;
  background: radial-gradient(closest-side, rgba(79, 179, 230, 0.45), rgba(79, 179, 230, 0));
  filter: blur(20px);
}

.sf-features {
  display: grid;
  gap: 32px;
  margin-top: 64px;
}
@media (min-width: 768px) { .sf-features { grid-template-columns: repeat(3, 1fr); gap: 40px; } }
.sf-feature { text-align: center; }
.sf-feature h3 {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--vp-c-text-1);
}
.sf-feature h3 svg { color: var(--vp-c-brand-1); }
.sf-feature p { margin: 12px 0 0; font-size: 15px; line-height: 1.65; color: var(--vp-c-text-2); }

/* Get started */
.sf-start-grid { display: grid; gap: 40px; align-items: start; }
@media (min-width: 960px) { .sf-start-grid { grid-template-columns: 1fr 1.1fr; gap: 56px; } }
.sf-steps { list-style: none; margin: 28px 0 0; padding: 0; display: grid; gap: 18px; }
.sf-steps li { display: flex; gap: 14px; align-items: flex-start; }
.sf-step-title { display: inline-block; font-size: 15px; font-weight: 600; color: var(--vp-c-text-1); text-decoration: none; }
.sf-step-title:hover { color: var(--vp-c-brand-1); }
.sf-steps code {
  padding: 1px 5px;
  border-radius: 4px;
  background: var(--vp-c-default-soft);
  font-family: var(--vp-font-family-mono);
  font-size: 0.9em;
  color: var(--vp-c-text-1);
}
.sf-steps p { margin: 2px 0 0; font-size: 14px; line-height: 1.55; color: var(--vp-c-text-2); }
.sf-step-num {
  flex: none;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-size: 13px;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

/* Support */
.sf-support-grid { display: grid; gap: 40px; margin-top: 48px; text-align: center; }
@media (min-width: 768px) { .sf-support-grid { grid-template-columns: 1fr 1fr; gap: 64px; } }
.sf-support h3 {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 20px;
  font-size: 18px;
  font-weight: 700;
}
.sf-support h3 svg { color: var(--vp-c-brand-1); }
.sf-tiles {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.sf-tiles-wide { grid-template-columns: repeat(3, minmax(0, 1fr)); align-content: center; }
.sf-tile {
  padding: 14px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg);
  font-size: 14px;
  font-weight: 600;
  text-align: center;
}

/* SpecBundle */
.sf-bundle { display: grid; gap: 40px; align-items: center; }
@media (min-width: 960px) { .sf-bundle { grid-template-columns: 1.1fr 0.9fr; gap: 64px; } }
.sf-bundle-logo { display: block; width: 240px; max-width: 60%; height: auto; margin-bottom: 20px; }
.sf-bundle .sf-h2 { font-size: 30px; }
@media (min-width: 640px) { .sf-bundle .sf-h2 { font-size: 34px; } }

/* Community */
.sf-community { display: grid; gap: 16px; margin-top: 44px; }
@media (min-width: 640px) { .sf-community { grid-template-columns: repeat(2, 1fr); } }
@media (min-width: 960px) { .sf-community { grid-template-columns: repeat(4, 1fr); } }
.sf-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 28px 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  text-decoration: none;
  transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.sf-card:hover, .sf-card:focus-visible { border-color: var(--vp-c-brand-1); transform: translateY(-2px); box-shadow: 0 10px 28px rgba(31, 127, 192, 0.12); }
.sf-card:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 3px; }
.sf-card-icon { color: var(--vp-c-brand-1); }
.sf-card strong { font-size: 16px; }
.sf-card p { margin: 0; font-size: 14px; line-height: 1.55; color: var(--vp-c-text-2); }
</style>
