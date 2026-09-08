import { readdirSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitepress'

// Layout:
//   docs/README.md          contributor guide (not a page)
//   docs/sections/**/*.md   documentation pages, served without the `sections/` prefix
//   docs/recipes/*.md       training recipes, served at /recipes/ with their own sidebar
//   docs/data/*.json        SpecBundle model list and benchmark results (community-editable)
//   docs/web/               this VitePress app (config, theme, landing + SpecBundle pages);
//                           docs/web/data is a symlink to docs/data
const version = readFileSync(
  fileURLToPath(new URL('../../../version.txt', import.meta.url)),
  'utf8'
).trim()

const nodeModule = (name: string) =>
  fileURLToPath(new URL(`../node_modules/${name}`, import.meta.url))

// Sidebar entries for docs/recipes/*.md, one per file, titled from frontmatter
// (falling back to the first heading) so a new recipe only needs a Markdown file.
function recipeSidebar() {
  const dir = fileURLToPath(new URL('../../recipes/', import.meta.url))
  return readdirSync(dir)
    .filter((f) => f.endsWith('.md') && f !== 'index.md')
    .sort()
    .map((f) => {
      const src = readFileSync(dir + f, 'utf8')
      const title =
        src.match(/^---[\s\S]*?^title:\s*(.+?)\s*$[\s\S]*?^---/m)?.[1] ??
        src.match(/^#\s+(.+)$/m)?.[1] ??
        f.replace(/\.md$/, '')
      return { text: title.replace(/^['"]|['"]$/g, ''), link: `/recipes/${f.replace(/\.md$/, '')}` }
    })
}

const repo = 'https://github.com/sgl-project/SpecForge'

export default defineConfig({
  title: 'SpecForge',
  description:
    'SpecForge is the SGLang-native framework for training speculative decoding draft models (EAGLE3, P-EAGLE, DFlash, DFlash2, DSpark, Domino, MTP).',
  base: '/SpecForge/',
  lang: 'en-US',
  lastUpdated: true,
  cleanUrls: false,

  // Pages are collected from docs/ (the parent of this app directory).
  srcDir: '..',
  srcExclude: [
    '**/node_modules/**',
    '**/.venv/**',
    '**/_build/**',
    'README.md',
  ],
  // Strip the on-disk prefixes so URLs stay stable:
  //   docs/sections/basic_usage/training.md -> /basic_usage/training.html
  //   docs/web/index.md                     -> /index.html
  rewrites: {
    'sections/:path(.*)': ':path',
    'web/:page(index|specbundle).md': ':page.md',
  },

  head: [
    ['link', { rel: 'icon', type: 'image/x-icon', href: '/SpecForge/logo.ico' }],
    ['link', { rel: 'apple-touch-icon', href: '/SpecForge/logo-icon.png' }],
    ['meta', { name: 'theme-color', content: '#2b8fd0' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'SpecForge' }],
    ['meta', {
      property: 'og:description',
      content: 'Train speculative decoding draft models for SGLang.',
    }],
    ['meta', { property: 'og:image', content: 'https://docs.sglang.io/SpecForge/logo.png' }],
  ],

  markdown: {
    math: true,
    lineNumbers: false,
  },

  vite: {
    // Static assets live next to the app rather than under docs/public.
    publicDir: fileURLToPath(new URL('../public', import.meta.url)),
    resolve: {
      // Pages under docs/sections have no node_modules of their own, so make
      // the compiled Markdown modules resolve `vue` from this app's copy.
      // (`vitepress` itself is already aliased by the VitePress plugin.)
      alias: [
        { find: /^vue$/, replacement: nodeModule('vue') },
        { find: /^vue\//, replacement: nodeModule('vue') + '/' },
      ],
    },
    server: {
      // Keep the dev-server file watcher away from large, irrelevant trees.
      watch: {
        ignored: ['**/node_modules/**', '**/.venv/**', '**/_build/**'],
      },
    },
  },

  themeConfig: {
    logo: { src: '/logo-icon.png', alt: 'SpecForge' },
    siteTitle: 'SpecForge',

    nav: [
      { text: 'Docs', link: '/get_started/about', activeMatch: '^/(get_started|concepts|basic_usage|advanced_features|benchmarks|examples)/' },
      { text: 'Recipes', link: '/recipes/', activeMatch: '^/recipes/' },
      { text: 'SpecBundle', link: '/specbundle', activeMatch: '^/specbundle' },
      {
        text: `v${version}`,
        items: [
          { text: 'Releases', link: `${repo}/releases` },
          { text: 'PyPI', link: 'https://pypi.org/project/specforge/' },
          { text: 'Changelog', link: `${repo}/commits/main` },
        ],
      },
    ],

    sidebar: {
      '/recipes/': [
        {
          text: 'Recipes',
          items: [{ text: 'All Recipes', link: '/recipes/' }, ...recipeSidebar()],
        },
      ],
      '/': [
        {
          text: 'Get Started',
          items: [
            { text: 'About SpecForge', link: '/get_started/about' },
            { text: 'Installation', link: '/get_started/installation' },
          ],
        },
        {
          text: 'Concepts',
          items: [
            { text: 'Speculative Decoding', link: '/concepts/speculative_decoding' },
            { text: 'EAGLE3', link: '/concepts/EAGLE3' },
            { text: 'DFlash2', link: '/concepts/DFlash2' },
          ],
        },
        {
          text: 'Basic Usage',
          items: [
            { text: 'Data Preparation', link: '/basic_usage/data_preparation' },
            { text: 'Training', link: '/basic_usage/training' },
            { text: 'Disaggregated Training', link: '/basic_usage/disaggregated_training' },
            { text: 'AMD ROCm Tutorial', link: '/basic_usage/AMD/amd_rocm' },
            { text: 'Ascend NPU Tutorial', link: '/basic_usage/Ascend/ascend_npu' },
          ],
        },
        {
          text: 'Advanced Features',
          items: [
            { text: 'Customize a Training Run', link: '/advanced_features/customization' },
          ],
        },
        {
          text: 'Benchmarks',
          items: [
            { text: 'Benchmarking Inference Serving', link: '/benchmarks/benchmark' },
            { text: 'EAGLE3 Disaggregated Parity', link: '/benchmarks/eagle3-disaggregated-parity' },
            { text: 'Domino Disaggregated Performance', link: '/benchmarks/domino-disaggregated-performance' },
          ],
        },
        {
          text: 'Examples',
          items: [
            { text: 'Llama 3.1 8B EAGLE3: Online', link: '/examples/llama3-eagle3-online' },
            { text: 'Llama 3.1 8B EAGLE3: Offline', link: '/examples/llama3-eagle3-offline' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: repo },
      {
        icon: {
          svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true"><title>Hugging Face</title><path d="M12 2.5a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-3.4 6.2a1 1 0 1 1 0 2 1 1 0 0 1 0-2Zm6.8 0a1 1 0 1 1 0 2 1 1 0 0 1 0-2ZM7.8 12.6c.4-.3.9-.1 1.6.3 1.2.7 4 .7 5.2 0 .7-.4 1.2-.6 1.6-.3.3.3.2.8-.1 1.4a4.9 4.9 0 0 1-8.2 0c-.3-.6-.4-1.1-.1-1.4Z"/></svg>',
        },
        link: 'https://huggingface.co/collections/lmsys/specbundle',
        ariaLabel: 'SpecBundle on Hugging Face',
      },
    ],

    editLink: {
      pattern: `${repo}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub',
    },

    search: {
      provider: 'local',
    },

    lastUpdated: {
      text: 'Last updated',
      formatOptions: { dateStyle: 'medium', forceLocale: true },
    },

    outline: { level: [2, 3] },

    footer: {
      message: 'Released under the MIT License.',
      copyright: `Copyright © 2025-${new Date().getFullYear()} SpecForge Team · SGLang`,
    },
  },
})
