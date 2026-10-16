<script setup lang="ts">
/**
 * Interactive SpecBundle performance dashboard, embedded in the SpecBundle
 * page. Renders the benchmark results checked in at
 * `docs/data/specbundle_benchmarks.json` (see `data.ts` for the shape).
 */
import { computed, ref, watch } from 'vue'
import { defineClientComponent } from 'vitepress'
import raw from '../../../../data/specbundle_benchmarks.json'
import FilterControls from './FilterControls.vue'
import BenchmarkTable from './BenchmarkTable.vue'
import {
  averageSpeedup,
  fmt,
  isBaseline,
  processAll,
  type Benchmark,
  type MetricKey,
  type RawData,
} from './data'

// ECharts needs `window`, so the chart is loaded on the client only.
const BenchmarkChart = defineClientComponent(() => import('./BenchmarkChart.vue'))

const byTarget = processAll(raw as RawData)
const targets = Object.keys(byTarget)

const target = ref(targets[0] ?? '')
const draft = ref('all')
const benchmark = ref<Benchmark | 'all'>('all')
const metric = ref<MetricKey>('throughput')

const targetRows = computed(() => byTarget[target.value] ?? [])

const drafts = computed(() => {
  const seen = new Set<string>()
  for (const r of targetRows.value) if (r.draftModel) seen.add(r.draftModel)
  return [...seen]
})

watch(target, () => {
  if (draft.value !== 'all' && !drafts.value.includes(draft.value)) draft.value = 'all'
})

const rows = computed(() =>
  draft.value === 'all'
    ? targetRows.value
    : targetRows.value.filter((r) => isBaseline(r) || r.draftModel === draft.value)
)

const bestSpeedup = computed(() => {
  let best: number | null = null
  for (const r of rows.value) {
    if (isBaseline(r)) continue
    const s = averageSpeedup(r)
    if (s !== null && (best === null || s > best)) best = s
  }
  return best
})

const configCount = computed(() => rows.value.filter((r) => !isBaseline(r)).length)
</script>

<template>
  <section class="bd">
    <FilterControls
      v-model:target="target"
      v-model:draft="draft"
      v-model:benchmark="benchmark"
      v-model:metric="metric"
      :targets="targets"
      :drafts="drafts"
    />

    <div class="bd-stats">
      <div class="bd-stat">
        <span class="bd-stat-label">Target model</span>
        <span class="bd-stat-value bd-stat-text">{{ target }}</span>
      </div>
      <div class="bd-stat">
        <span class="bd-stat-label">Draft models</span>
        <span class="bd-stat-value">{{ drafts.length }}</span>
      </div>
      <div class="bd-stat">
        <span class="bd-stat-label">Configurations</span>
        <span class="bd-stat-value">{{ configCount }}</span>
      </div>
      <div class="bd-stat is-accent">
        <span class="bd-stat-label">Best avg. speedup</span>
        <span class="bd-stat-value">{{ bestSpeedup === null ? '–' : `${fmt(bestSpeedup)}×` }}</span>
      </div>
    </div>

    <div v-if="rows.length" class="bd-card">
      <ClientOnly>
        <BenchmarkChart :rows="rows" :benchmark="benchmark" :metric="metric" />
        <template #fallback>
          <div class="bd-chart-placeholder">Loading chart…</div>
        </template>
      </ClientOnly>
    </div>

    <div v-if="rows.length" class="bd-card">
      <div class="bd-card-title">Detailed results</div>
      <BenchmarkTable :rows="rows" :benchmark="benchmark" :metric="metric" />
    </div>

    <div v-else class="bd-empty">No benchmark results for {{ target }} yet.</div>
  </section>
</template>

<style scoped>
.bd {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin: 8px 0 32px;
}

.bd-stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.bd-stat {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  padding: 14px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg-soft);
}
.bd-stat.is-accent {
  border-color: transparent;
  background: var(--vp-c-brand-soft);
}
.bd-stat-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--vp-c-text-2);
}
.bd-stat.is-accent .bd-stat-label { color: var(--vp-c-brand-1); }
.bd-stat-value {
  font-size: 26px;
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: -0.02em;
  color: var(--vp-c-text-1);
  font-variant-numeric: tabular-nums;
}
.bd-stat.is-accent .bd-stat-value { color: var(--vp-c-brand-1); }
.bd-stat-text {
  font-size: 16px;
  line-height: 1.3;
  overflow-wrap: anywhere;
}

.bd-card {
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg);
}
.bd-card-title {
  margin-bottom: 12px;
  font-size: 15px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}
.bd-chart-placeholder {
  display: grid;
  place-items: center;
  height: 460px;
  color: var(--vp-c-text-2);
  font-size: 14px;
}

.bd-empty {
  padding: 40px 16px;
  text-align: center;
  border: 1px dashed var(--vp-c-divider);
  border-radius: 14px;
  color: var(--vp-c-text-2);
}

@media (max-width: 960px) {
  .bd-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .bd-card { padding: 14px; }
  .bd-stat-value { font-size: 22px; }
}
</style>
