<script setup lang="ts">
/**
 * Searchable gallery of the draft models in the Hugging Face SpecBundle
 * collection (https://huggingface.co/collections/lmsys/specbundle).
 *
 * Renders from the checked-in snapshot in `docs/data/specbundle_models.json` so the
 * page works at build time and offline, then refreshes download/like counts
 * and picks up newly added models from the Hugging Face API in the browser.
 * Hand-curated fields (target model, regenerated dataset) live in the snapshot
 * and are merged onto the live data by repo id.
 */
import { computed, onMounted, ref, watch } from 'vue'
import snapshot from '../../../data/specbundle_models.json'

interface Provider {
  name: string
  fullname: string
  avatar: string | null
  type: string
}

interface Model {
  id: string
  name: string
  provider: Provider
  method: string
  target: string | null
  dataset: string | null
  downloads: number
  likes: number
  numParameters: number | null
  lastModified: string | null
  gated: boolean
}

type SortKey = 'downloads' | 'likes' | 'recent' | 'name'

const METHOD_ORDER = ['EAGLE3', 'DFlash', 'DSpark', 'Domino', 'Other']

const models = ref<Model[]>(snapshot.models as Model[])
const updated = ref<string>(snapshot.updated)
const live = ref(false)

const query = ref('')
const method = ref('All')
const provider = ref('All')
const sort = ref<SortKey>('downloads')

const PAGE_SIZE = 9
const page = ref(1)

function detectMethod(id: string): string {
  const s = id.toLowerCase()
  if (s.includes('dspark')) return 'DSpark'
  if (s.includes('domino')) return 'Domino'
  if (s.includes('dflash')) return 'DFlash'
  if (s.includes('eagle')) return 'EAGLE3'
  return 'Other'
}

onMounted(async () => {
  try {
    const res = await fetch(snapshot.collectionApi, { headers: { Accept: 'application/json' } })
    if (!res.ok) return
    const data = await res.json()
    const curated = new Map(models.value.map((m) => [m.id, m]))
    const merged: Model[] = []
    for (const item of data.items ?? []) {
      if (item.type !== 'model') continue
      const prev = curated.get(item.id)
      const author = item.authorData ?? {}
      merged.push({
        id: item.id,
        name: item.id.split('/').slice(1).join('/'),
        provider: prev?.provider ?? {
          name: item.author ?? item.id.split('/')[0],
          fullname: author.fullname ?? item.author ?? item.id.split('/')[0],
          avatar: author.avatarUrl ?? null,
          type: author.type ?? 'user',
        },
        method: prev?.method ?? detectMethod(item.id),
        target: prev?.target ?? null,
        dataset: prev?.dataset ?? null,
        downloads: item.downloads ?? prev?.downloads ?? 0,
        likes: item.likes ?? prev?.likes ?? 0,
        numParameters: item.numParameters ?? prev?.numParameters ?? null,
        lastModified: item.lastModified ?? prev?.lastModified ?? null,
        gated: Boolean(item.gated),
      })
    }
    if (merged.length) {
      models.value = merged
      updated.value = data.lastUpdated ?? new Date().toISOString()
      live.value = true
    }
  } catch {
    /* offline or blocked: keep the snapshot */
  }
})

const methods = computed(() => {
  const present = new Set(models.value.map((m) => m.method))
  return ['All', ...METHOD_ORDER.filter((m) => present.has(m))]
})

const providers = computed(() => {
  const map = new Map<string, Provider>()
  for (const m of models.value) if (!map.has(m.provider.name)) map.set(m.provider.name, m.provider)
  return [...map.values()].sort((a, b) => a.fullname.localeCompare(b.fullname))
})

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  const terms = q ? q.split(/\s+/) : []
  const list = models.value.filter((m) => {
    if (method.value !== 'All' && m.method !== method.value) return false
    if (provider.value !== 'All' && m.provider.name !== provider.value) return false
    if (!terms.length) return true
    const hay = [m.id, m.provider.name, m.provider.fullname, m.method, m.target ?? '', m.dataset ?? '']
      .join(' ')
      .toLowerCase()
    return terms.every((t) => hay.includes(t))
  })
  const s = sort.value
  return list.sort((a, b) => {
    if (s === 'downloads') return b.downloads - a.downloads
    if (s === 'likes') return b.likes - a.likes || b.downloads - a.downloads
    if (s === 'recent') return (b.lastModified ?? '').localeCompare(a.lastModified ?? '')
    return a.name.localeCompare(b.name)
  })
})

const pageCount = computed(() => Math.max(1, Math.ceil(filtered.value.length / PAGE_SIZE)))
const paged = computed(() => {
  const start = (page.value - 1) * PAGE_SIZE
  return filtered.value.slice(start, start + PAGE_SIZE)
})
const pageRange = computed(() => {
  const start = (page.value - 1) * PAGE_SIZE + 1
  return { start, end: Math.min(page.value * PAGE_SIZE, filtered.value.length) }
})

// Any change to the filters restarts from the first page.
watch([query, method, provider, sort], () => {
  page.value = 1
})

function goTo(n: number) {
  page.value = Math.min(Math.max(1, n), pageCount.value)
  const el = document.querySelector<HTMLElement>('#models')
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

/** Page numbers to show: first, last, and a window around the current page. */
const pageItems = computed<(number | '…')[]>(() => {
  const n = pageCount.value
  const c = page.value
  if (n <= 7) return Array.from({ length: n }, (_, i) => i + 1)
  const items: (number | '…')[] = [1]
  if (c > 3) items.push('…')
  for (let i = Math.max(2, c - 1); i <= Math.min(n - 1, c + 1); i++) items.push(i)
  if (c < n - 2) items.push('…')
  items.push(n)
  return items
})

const totalDownloads = computed(() => models.value.reduce((n, m) => n + m.downloads, 0))

function fmtCount(n: number): string {
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k'
  return String(n)
}

function fmtParams(n: number | null): string | null {
  if (!n) return null
  if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B'
  if (n >= 1e6) return Math.round(n / 1e6) + 'M'
  return String(n)
}

function fmtDate(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

function initials(p: Provider): string {
  return p.fullname
    .split(/[\s\-_/]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]!.toUpperCase())
    .join('')
}

function hfUrl(id: string, kind: 'model' | 'dataset' = 'model'): string {
  return kind === 'dataset' ? `https://huggingface.co/datasets/${id}` : `https://huggingface.co/${id}`
}

function clear() {
  query.value = ''
  method.value = 'All'
  provider.value = 'All'
}
</script>

<template>
  <section class="mg">
    <div class="mg-toolbar">
      <label class="mg-search">
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" d="M10.5 18a7.5 7.5 0 1 1 0-15 7.5 7.5 0 0 1 0 15Zm10.5 3-5.2-5.2" />
        </svg>
        <input
          v-model="query"
          type="search"
          placeholder="Search by model, target, provider or method…"
          aria-label="Search SpecBundle models"
          autocomplete="off"
          spellcheck="false"
        />
        <button v-if="query" type="button" class="mg-clear" aria-label="Clear search" @click="query = ''">×</button>
      </label>

      <div class="mg-controls">
        <label class="mg-select">
          <span>Provider</span>
          <select v-model="provider" aria-label="Filter by provider">
            <option value="All">All providers</option>
            <option v-for="p in providers" :key="p.name" :value="p.name">{{ p.fullname }}</option>
          </select>
        </label>
        <label class="mg-select">
          <span>Sort</span>
          <select v-model="sort" aria-label="Sort models">
            <option value="downloads">Most downloads</option>
            <option value="likes">Most likes</option>
            <option value="recent">Recently updated</option>
            <option value="name">Name (A–Z)</option>
          </select>
        </label>
      </div>
    </div>

    <div class="mg-chips" role="group" aria-label="Filter by method">
      <button
        v-for="m in methods"
        :key="m"
        type="button"
        class="mg-chip"
        :class="{ active: method === m }"
        :aria-pressed="method === m"
        @click="method = m"
      >
        {{ m }}
        <span class="mg-chip-count">{{ m === 'All' ? models.length : models.filter((x) => x.method === m).length }}</span>
      </button>
    </div>

    <p class="mg-status">
      <span>
        Showing <strong>{{ filtered.length ? `${pageRange.start}–${pageRange.end}` : 0 }}</strong> of {{ filtered.length }} matching draft models ·
        {{ fmtCount(totalDownloads) }} downloads across the collection
      </span>
      <span class="mg-status-src">
        {{ live ? 'Live from Hugging Face' : 'Snapshot' }} · updated {{ fmtDate(updated) }}
      </span>
    </p>

    <div v-if="filtered.length" class="mg-grid">
      <article v-for="m in paged" :key="m.id" class="mg-card">
        <header class="mg-card-head">
          <a :href="`https://huggingface.co/${m.provider.name}`" target="_blank" rel="noopener" class="mg-avatar" :title="m.provider.fullname">
            <img v-if="m.provider.avatar" :src="m.provider.avatar" :alt="m.provider.fullname" loading="lazy" width="40" height="40" />
            <span v-else class="mg-avatar-fallback" aria-hidden="true">{{ initials(m.provider) }}</span>
          </a>
          <div class="mg-provider">
            <a :href="`https://huggingface.co/${m.provider.name}`" target="_blank" rel="noopener" class="mg-provider-name">{{ m.provider.fullname }}</a>
            <span class="mg-provider-handle">@{{ m.provider.name }}</span>
          </div>
          <span class="mg-method" :data-method="m.method">{{ m.method }}</span>
        </header>

        <h3 class="mg-title">
          <a :href="hfUrl(m.id)" target="_blank" rel="noopener">{{ m.name }}</a>
        </h3>

        <dl class="mg-meta">
          <div v-if="m.target">
            <dt>Target</dt>
            <dd><a :href="hfUrl(m.target)" target="_blank" rel="noopener">{{ m.target }}</a></dd>
          </div>
          <div v-if="m.dataset">
            <dt>Dataset</dt>
            <dd><a :href="hfUrl(m.dataset, 'dataset')" target="_blank" rel="noopener">{{ m.dataset }}</a></dd>
          </div>
        </dl>

        <footer class="mg-card-foot">
          <span class="mg-stat" title="Downloads (last 30 days)">
            <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="M12 4v11m0 0-4-4m4 4 4-4M5 20h14"/></svg>
            {{ fmtCount(m.downloads) }}
          </span>
          <span class="mg-stat" title="Likes">
            <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10Z"/></svg>
            {{ fmtCount(m.likes) }}
          </span>
          <span v-if="fmtParams(m.numParameters)" class="mg-stat" title="Draft model parameters">{{ fmtParams(m.numParameters) }} params</span>
          <span class="mg-stat mg-stat-date" :title="m.lastModified ?? ''">{{ fmtDate(m.lastModified) }}</span>
          <a :href="hfUrl(m.id)" target="_blank" rel="noopener" class="mg-open">Open on Hugging Face ↗</a>
        </footer>
      </article>
    </div>

    <div v-else class="mg-empty">
      <p>No models match your search.</p>
      <button type="button" class="mg-chip active" @click="clear">Reset filters</button>
    </div>

    <nav v-if="pageCount > 1" class="mg-pager" aria-label="Model pages">
      <button type="button" class="mg-page mg-page-arrow" :disabled="page === 1" aria-label="Previous page" @click="goTo(page - 1)">
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="m15 6-6 6 6 6" /></svg>
      </button>
      <template v-for="(item, i) in pageItems" :key="i">
        <span v-if="item === '…'" class="mg-page-gap" aria-hidden="true">…</span>
        <button
          v-else
          type="button"
          class="mg-page"
          :class="{ active: item === page }"
          :aria-current="item === page ? 'page' : undefined"
          @click="goTo(item)"
        >
          {{ item }}
        </button>
      </template>
      <button type="button" class="mg-page mg-page-arrow" :disabled="page === pageCount" aria-label="Next page" @click="goTo(page + 1)">
        <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="m9 6 6 6-6 6" /></svg>
      </button>
    </nav>
  </section>
</template>

<style scoped>
.mg { margin: 8px 0 32px; }

.mg-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
}
.mg-search {
  position: relative;
  display: flex;
  align-items: center;
  flex: 1 1 320px;
  gap: 8px;
  padding: 0 12px;
  height: 44px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  transition: border-color 0.2s, box-shadow 0.2s;
}
.mg-search:focus-within {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft);
}
.mg-search input {
  flex: 1;
  min-width: 0;
  height: 100%;
  border: 0;
  background: transparent;
  color: var(--vp-c-text-1);
  font-size: 15px;
  outline: none;
}
.mg-search input::-webkit-search-cancel-button { display: none; }
.mg-search input::placeholder { color: var(--vp-c-text-2); opacity: 1; }
.mg-clear {
  border: 0;
  background: transparent;
  color: var(--vp-c-text-2);
  font-size: 20px;
  line-height: 1;
  cursor: pointer;
  padding: 0 4px;
}
.mg-clear:hover { color: var(--vp-c-text-1); }

.mg-controls { display: flex; flex-wrap: wrap; gap: 12px; }
.mg-select {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 44px;
  padding: 0 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg-soft);
  font-size: 14px;
  color: var(--vp-c-text-2);
}
.mg-select select {
  border: 0;
  background: transparent;
  color: var(--vp-c-text-1);
  font-size: 14px;
  font-weight: 500;
  outline: none;
  cursor: pointer;
  max-width: 220px;
}
.mg-select:focus-within {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft);
}

.mg-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.mg-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 999px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}
.mg-chip:hover { color: var(--vp-c-text-1); border-color: var(--vp-c-text-3); }
.mg-chip:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.mg-chip.active {
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  border-color: transparent;
}
.mg-chip-count {
  padding: 0 6px;
  border-radius: 999px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 11px;
  font-weight: 600;
}
.mg-chip.active .mg-chip-count { background: var(--vp-c-bg); color: var(--vp-c-brand-1); }

.mg-status {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 8px;
  margin: 16px 0 12px;
  font-size: 13px;
  color: var(--vp-c-text-2);
}
.mg-status-src { color: var(--vp-c-text-2); }

.mg-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}
.mg-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.mg-card:hover,
.mg-card:focus-within {
  border-color: var(--vp-c-brand-2);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(43, 143, 208, 0.12);
}

.mg-card-head { display: flex; align-items: center; gap: 10px; }
.mg-avatar {
  flex: none;
  width: 40px;
  height: 40px;
  border-radius: 10px;
  overflow: hidden;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  display: grid;
  place-items: center;
}
.mg-avatar img { width: 100%; height: 100%; object-fit: cover; display: block; }
.mg-avatar-fallback {
  font-weight: 700;
  font-size: 14px;
  color: var(--vp-c-brand-1);
}
.mg-provider { display: flex; flex-direction: column; min-width: 0; flex: 1; }
.mg-provider-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  text-decoration: none;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.mg-provider-name:hover { color: var(--vp-c-brand-1); }
.mg-provider-handle { font-size: 12px; color: var(--vp-c-text-2); }

.mg-method {
  flex: none;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: #fff;
  background: var(--vp-c-text-3);
}
.mg-method[data-method='EAGLE3'] { background: #1f7fc0; }
.mg-method[data-method='DFlash'] { background: #7c3aed; }
.mg-method[data-method='DSpark'] { background: #d97706; }
.mg-method[data-method='Domino'] { background: #059669; }

.mg-title {
  margin: 0;
  padding: 0;
  border: 0;
  font-size: 16px;
  font-weight: 700;
  line-height: 1.35;
  letter-spacing: -0.01em;
  overflow-wrap: anywhere;
}
.mg-title a { color: var(--vp-c-text-1); text-decoration: none; }
.mg-title a:focus-visible, .mg-open:focus-visible, .mg-avatar:focus-visible, .mg-provider-name:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
  border-radius: 4px;
}
.mg-title a:hover { color: var(--vp-c-brand-1); }

.mg-meta { margin: 0; display: grid; gap: 4px; font-size: 13px; }
.mg-meta > div { display: grid; grid-template-columns: 58px 1fr; gap: 8px; align-items: baseline; }
.mg-meta dt { color: var(--vp-c-text-2); font-weight: 500; }
.mg-meta dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.mg-meta a { color: var(--vp-c-text-2); text-decoration: none; }
.mg-meta a:hover { color: var(--vp-c-brand-1); text-decoration: underline; }

.mg-card-foot {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 12px;
  margin-top: auto;
  padding-top: 10px;
  border-top: 1px dashed var(--vp-c-divider);
  font-size: 12px;
  color: var(--vp-c-text-2);
}
.mg-stat { display: inline-flex; align-items: center; gap: 4px; }
.mg-stat-date { color: var(--vp-c-text-2); }
.mg-open {
  margin-left: auto;
  font-weight: 600;
  color: var(--vp-c-brand-1);
  text-decoration: none;
}
.mg-open:hover { text-decoration: underline; }

.mg-empty {
  padding: 40px 16px;
  text-align: center;
  border: 1px dashed var(--vp-c-divider);
  border-radius: 14px;
  color: var(--vp-c-text-2);
}
.mg-empty p { margin: 0 0 12px; }

.mg-pager {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 6px;
  margin-top: 24px;
}
.mg-page {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 36px;
  height: 36px;
  padding: 0 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 14px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  cursor: pointer;
  transition: border-color 0.15s, color 0.15s, background 0.15s;
}
.mg-page:hover:not(:disabled) { color: var(--vp-c-text-1); border-color: var(--vp-c-text-3); }
.mg-page.active {
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
  border-color: transparent;
}
.mg-page:disabled { opacity: 0.4; cursor: default; }
.mg-page:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.mg-page-gap { padding: 0 4px; color: var(--vp-c-text-3); }

@media (max-width: 640px) {
  .mg-grid { grid-template-columns: 1fr; }
  .mg-controls { width: 100%; }
  .mg-select { flex: 1; }
}
</style>
