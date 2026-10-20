<script setup lang="ts">
import { computed } from 'vue'
import {
  BENCHMARKS,
  averageSpeedup,
  fmt,
  isBaseline,
  metricValue,
  shortDraftName,
  type Benchmark,
  type MetricKey,
  type Row,
} from './data'

const props = defineProps<{
  rows: Row[]
  benchmark: Benchmark | 'all'
  metric: MetricKey
}>()

const columns = computed<Benchmark[]>(() =>
  props.benchmark === 'all' ? [...BENCHMARKS] : [props.benchmark as Benchmark]
)

function speedupClass(s: number | null): string {
  if (s === null) return ''
  if (s >= 2) return 'is-great'
  if (s >= 1.5) return 'is-good'
  if (s >= 1.1) return 'is-ok'
  return 'is-flat'
}

function hfUrl(id: string): string {
  return `https://huggingface.co/${id}`
}
</script>

<template>
  <div class="bd-table">
    <table>
      <thead>
        <tr>
          <th class="bd-sticky">Draft model</th>
          <th>Config</th>
          <th v-for="b in columns" :key="b" class="bd-col-metric">
            <span class="bd-th">{{ b }}</span>
            <span class="bd-th-sub">acc. len / tokens/s</span>
          </th>
          <th class="bd-col-avg">
            <span class="bd-th">Avg speedup</span>
            <span class="bd-th-sub">throughput vs. baseline</span>
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="`${row.draftModel}|${row.config}`">
          <td class="bd-sticky bd-draft">
            <span v-if="isBaseline(row)" class="bd-badge is-baseline">Baseline</span>
            <a v-else :href="hfUrl(row.draftModel!)" target="_blank" rel="noopener" class="bd-badge" :title="row.draftModel!">
              {{ shortDraftName(row.draftModel) }}
            </a>
          </td>
          <td class="bd-config">{{ isBaseline(row) ? '–' : row.config }}</td>
          <td
            v-for="b in columns"
            :key="b"
            class="bd-cell"
            :class="{ 'is-focus': metric !== 'speedup' && row.metrics[b] }"
          >
            <template v-if="row.metrics[b]">
              <span class="bd-pair">
                <span class="bd-acc" :class="{ 'is-hi': metric === 'accLen' }">{{ fmt(row.metrics[b]!.accLen) }}</span>
                <span class="bd-sep">/</span>
                <span class="bd-thr" :class="{ 'is-hi': metric === 'throughput' }">{{ fmt(row.metrics[b]!.throughput) }}</span>
              </span>
              <span
                v-if="!isBaseline(row) && metricValue(row, b, 'speedup') !== null"
                class="bd-speedup"
                :class="speedupClass(metricValue(row, b, 'speedup'))"
              >
                {{ fmt(metricValue(row, b, 'speedup')) }}×
              </span>
            </template>
            <span v-else class="bd-na">–</span>
          </td>
          <td class="bd-cell bd-col-avg">
            <span v-if="isBaseline(row)" class="bd-na">1.00×</span>
            <span v-else class="bd-speedup is-avg" :class="speedupClass(averageSpeedup(row))">
              {{ fmt(averageSpeedup(row)) }}×
            </span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
/* The wrapper class keeps these rules ahead of VitePress' `.vp-doc table` styles. */
.bd-table {
  overflow-x: auto;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
}
.bd-table table {
  display: table;
  width: 100%;
  margin: 0;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 13px;
}
.bd-table th,
.bd-table td {
  padding: 10px 14px;
  border: 0;
  border-bottom: 1px solid var(--vp-c-divider);
  text-align: left;
  vertical-align: middle;
  white-space: nowrap;
}
.bd-table th {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.bd-table tbody tr { background: var(--vp-c-bg); }
.bd-table tbody tr:nth-child(2n) { background: var(--vp-c-bg); }
.bd-table tbody tr:hover { background: var(--vp-c-bg-soft); }
.bd-table tbody tr:last-child td { border-bottom: 0; }

.bd-sticky {
  position: sticky;
  left: 0;
  z-index: 1;
  background: inherit;
  box-shadow: 1px 0 0 var(--vp-c-divider);
}
.bd-table th.bd-sticky { z-index: 2; background: var(--vp-c-bg-soft); }

.bd-col-metric,
.bd-col-avg { text-align: center; }
.bd-th { display: block; }
.bd-th-sub {
  display: block;
  margin-top: 2px;
  font-size: 11px;
  font-weight: 400;
  letter-spacing: 0;
  text-transform: none;
  color: var(--vp-c-text-2);
}

.bd-badge {
  display: inline-block;
  max-width: 320px;
  padding: 3px 10px;
  border: 1px solid var(--vp-c-brand-soft);
  border-radius: 999px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
  overflow: hidden;
  text-overflow: ellipsis;
  vertical-align: middle;
}
.bd-table a.bd-badge { text-decoration: none; font-weight: 600; }
.bd-table a.bd-badge:hover { border-color: var(--vp-c-brand-1); }
.bd-table a.bd-badge:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 2px; }
.bd-badge.is-baseline {
  border-color: var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
}

.bd-config {
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  color: var(--vp-c-text-2);
}

.bd-cell { text-align: center; }
.bd-pair {
  display: inline-flex;
  align-items: baseline;
  gap: 4px;
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
.bd-acc,
.bd-thr { color: var(--vp-c-text-2); }
.bd-sep { color: var(--vp-c-text-3); }
.bd-pair .is-hi { color: var(--vp-c-text-1); font-weight: 600; }
.bd-na { color: var(--vp-c-text-3); }

.bd-speedup {
  display: inline-block;
  margin-left: 8px;
  padding: 1px 7px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
}
.bd-speedup.is-avg { margin-left: 0; font-size: 12px; padding: 3px 10px; }
.bd-speedup.is-great { background: rgba(5, 150, 105, 0.14); color: #047857; }
.bd-speedup.is-good { background: var(--vp-c-brand-soft); color: var(--vp-c-brand-1); }
.bd-speedup.is-ok { background: rgba(217, 119, 6, 0.14); color: #b45309; }
.dark .bd-speedup.is-great { color: #34d399; }
.dark .bd-speedup.is-ok { color: #fbbf24; }

@media (max-width: 640px) {
  .bd-speedup { display: block; margin: 4px 0 0; }
}
</style>
