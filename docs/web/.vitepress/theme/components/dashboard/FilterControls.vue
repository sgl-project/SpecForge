<script setup lang="ts">
import { BENCHMARKS, METRICS, shortDraftName, type Benchmark, type MetricKey } from './data'

defineProps<{
  targets: string[]
  drafts: string[]
}>()

const target = defineModel<string>('target', { required: true })
const draft = defineModel<string>('draft', { required: true })
const benchmark = defineModel<Benchmark | 'all'>('benchmark', { required: true })
const metric = defineModel<MetricKey>('metric', { required: true })
</script>

<template>
  <div class="bd-filters">
    <label class="bd-field">
      <span>Target model</span>
      <select v-model="target">
        <option v-for="t in targets" :key="t" :value="t">{{ t }}</option>
      </select>
    </label>
    <label class="bd-field">
      <span>Draft model</span>
      <select v-model="draft">
        <option value="all">All draft models</option>
        <option v-for="d in drafts" :key="d" :value="d">{{ shortDraftName(d) }}</option>
      </select>
    </label>
    <label class="bd-field">
      <span>Benchmark</span>
      <select v-model="benchmark">
        <option value="all">All benchmarks</option>
        <option v-for="b in BENCHMARKS" :key="b" :value="b">{{ b }}</option>
      </select>
    </label>
    <label class="bd-field">
      <span>Metric</span>
      <select v-model="metric">
        <option v-for="m in METRICS" :key="m.value" :value="m.value">{{ m.label }}</option>
      </select>
    </label>
  </div>
</template>

<style scoped>
.bd-filters {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.bd-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.bd-field span {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--vp-c-text-2);
}
.bd-field select {
  width: 100%;
  height: 40px;
  padding: 0 34px 0 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background-color: var(--vp-c-bg);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23929295' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  background-size: 14px;
  color: var(--vp-c-text-1);
  font-family: inherit;
  font-size: 14px;
  font-weight: 500;
  appearance: none;
  cursor: pointer;
  text-overflow: ellipsis;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.bd-field select:hover { border-color: var(--vp-c-text-3); }
.bd-field select:focus {
  outline: none;
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 0 0 3px var(--vp-c-brand-soft);
}

@media (max-width: 960px) {
  .bd-filters { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 480px) {
  .bd-filters { grid-template-columns: 1fr; }
}
</style>
