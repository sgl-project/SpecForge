<template>
  <div class="table-container">
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th class="sticky-col">Target Model</th>
            <th>Draft Model</th>
            <th>Config</th>
            <th v-if="showHardware">Hardware</th>
            <th v-for="benchmark in visibleBenchmarks" :key="benchmark" class="metric-header">
              <div class="th-content">
                {{ benchmark }}
                <span class="th-subtitle">Acc Len / Tokens/s</span>
              </div>
            </th>
            <th v-if="highlightMetric === 'speedup'">Avg Speedup</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(row, index) in data" :key="index">
            <td class="model-name sticky-col">
              {{ formatModelName(row.targetModel) }}
              <div class="mobile-label">Target</div>
            </td>
            <td class="draft-model-cell">
              <span :class="['component-badge', getDraftModelClass(row.draftModel)]">
                {{ formatDraftModel(row.draftModel) }}
              </span>
            </td>
            <td class="config-cell">{{ row.config }}</td>
            <td v-if="showHardware" class="hardware-cell">{{ row.hardware }}</td>
            <td
              v-for="benchmark in visibleBenchmarks"
              :key="benchmark"
              class="metric-cell-wrapper"
              :class="{ 'highlight-bg': isBestInRow(row, benchmark) }"
            >
              <div v-if="row.metrics[benchmark]" class="metric-content">
                <div class="metric-pair">
                  <span class="val-acc">{{ formatValue(row.metrics[benchmark].accLen) }}</span>
                  <span class="separator">/</span>
                  <span class="val-thru">{{ formatValue(row.metrics[benchmark].throughput) }}</span>
                </div>

                <div v-if="row.baseline && row.baseline[benchmark]" class="speedup-indicator">
                  <span class="speedup-tag" :class="getSpeedupClass(row.metrics[benchmark].throughput, row.baseline[benchmark].throughput)">
                    {{ calculateSpeedup(row.metrics[benchmark].throughput, row.baseline[benchmark].throughput) }}x
                  </span>
                </div>
              </div>
              <div v-else class="no-data">-</div>
            </td>
            <td v-if="highlightMetric === 'speedup'" class="avg-speedup">
               <span class="speedup-tag primary">
                 {{ calculateAverageSpeedup(row) }}x
               </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { removeSGLangPrefix } from '../utils/dataProcessor';

const props = defineProps({
  data: { type: Array, required: true },
  benchmarks: { type: Array, required: true },
  selectedBenchmark: { type: String, default: 'all' },
  highlightMetric: { type: String, default: 'throughput' }
});

const visibleBenchmarks = computed(() => {
  return props.selectedBenchmark === 'all'
    ? props.benchmarks
    : [props.selectedBenchmark];
});

const showHardware = computed(() => {
  return props.data.some(row => row.hardware && row.hardware !== '-');
});

function formatModelName(model) {
  if (!model) return '-';
  const cleaned = removeSGLangPrefix(model);
  return cleaned.split('/').pop() || cleaned;
}

function formatDraftModel(model) {
  if (!model) return '-';
  if (model === '-' || model === 'None') return 'Baseline';
  if (model.includes('SpecBundle')) return 'SpecBundle';
  // Remove SGLang-EAGLE3 prefix and simplify long names
  const cleaned = removeSGLangPrefix(model);
  return cleaned.split('/').pop() || cleaned;
}

function formatConfigDetails(row) {
  if (row.config === 'baseline' || !row.config) {
    return 'Baseline Configuration';
  }
  const parts = [];
  if (row.batch_size !== undefined) parts.push(`batch_size: ${row.batch_size}`);
  if (row.steps !== undefined) parts.push(`steps: ${row.steps}`);
  if (row.topk !== undefined) parts.push(`topk: ${row.topk}`);
  if (row.num_draft_tokens !== undefined) parts.push(`num_draft_tokens: ${row.num_draft_tokens}`);
  return parts.length > 0 ? parts.join(', ') : row.config;
}

function getDraftModelClass(model) {
  if (!model || model === '-' || model === 'None') return 'badge-baseline';
  if (model.includes('SpecBundle')) return 'badge-spec';
  if (model.toLowerCase().includes('eagle')) return 'badge-eagle';
  return 'badge-default';
}

function formatValue(value) {
  if (value === null || value === undefined) return '-';
  // Always format to 2 decimal places
  return typeof value === 'number' ? value.toFixed(2) : value;
}

function calculateSpeedup(specValue, baselineValue) {
  if (!specValue || !baselineValue || baselineValue === 0) return '-';
  return (specValue / baselineValue).toFixed(2);
}

function getSpeedupClass(spec, base) {
  if (!spec || !base) return '';
  const ratio = spec / base;
  if (ratio >= 2.0) return 'excellent';
  if (ratio >= 1.5) return 'good';
  if (ratio >= 1.1) return 'moderate';
  return 'neutral';
}

function calculateAverageSpeedup(row) {
  let totalSpeedup = 0;
  let count = 0;

  if (!row.baseline) return '-';

  props.benchmarks.forEach(benchmark => {
    const spec = row.metrics[benchmark]?.throughput;
    const base = row.baseline[benchmark]?.throughput;

    if (spec && base && base > 0) {
      totalSpeedup += spec / base;
      count++;
    }
  });

  return count > 0 ? (totalSpeedup / count).toFixed(2) : '-';
}

function isBestInRow(row, benchmark) {
  return false; // Implement if needed
}
</script>

<style scoped>
.table-container {
  overflow: hidden;
  border-radius: var(--radius-xl);
  border: 1px solid #e2e8f0;
}

.table-scroll {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
  background: white;
}

th {
  background: #f8fafc;
  padding: 16px 20px;
  text-align: left;
  font-weight: 600;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
  border-bottom: 2px solid #e2e8f0;
  white-space: nowrap;
}

.metric-header {
  text-align: center;
}

.th-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.th-subtitle {
  text-transform: none;
  font-size: 0.7rem;
  color: var(--color-text-muted);
  font-weight: 400;
}

td {
  padding: 16px 20px;
  border-bottom: 1px solid #f1f5f9;
  color: var(--color-text-main);
  vertical-align: middle;
}

tr:last-child td {
  border-bottom: none;
}

tr:hover {
  background-color: #f8fafc;
}

/* Sticky Column for Target Model */
.sticky-col {
  position: sticky;
  left: 0;
  background: white; /* Match table bg */
  z-index: 10;
  box-shadow: 2px 0 5px rgba(0,0,0,0.02);
}

tr:hover .sticky-col {
  background-color: #f8fafc;
}

th.sticky-col {
  background: #f8fafc;
  z-index: 11;
}

.model-name {
  font-weight: 600;
  color: var(--color-text-main);
  min-width: 180px;
}

.config-cell {
  min-width: 120px;
}

.config-display {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.config-short {
  font-weight: 600;
  color: var(--color-text-main);
  font-size: 0.875rem;
}

.config-details {
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  line-height: 1.3;
}

/* Badges */
.component-badge {
  display: inline-flex;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 600;
  line-height: 1.4;
  white-space: nowrap;
}

.badge-baseline {
  background: #f1f5f9;
  color: #64748b;
  border: 1px solid #e2e8f0;
}

.badge-spec {
  background: #eff6ff; /* blue-50 */
  color: #2563eb; /* blue-600 */
  border: 1px solid #dbeafe;
}

.badge-eagle {
  background: #f0fdf4; /* green-50 */
  color: #16a34a; /* green-600 */
  border: 1px solid #dcfce7;
}

.badge-default {
  background: #f8fafc;
  color: #475569;
}

/* Metrics */
.metric-cell-wrapper {
  text-align: center;
  min-width: 140px;
}

.metric-pair {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  font-family: 'JetBrains Mono', monospace; /* Monospaced for numbers if available, else standard */
  font-feature-settings: "tnum";
}

.val-acc {
  color: var(--color-text-secondary);
  font-size: 0.9em;
}

.separator {
  color: var(--color-text-muted);
  font-size: 0.8em;
}

.val-thru {
  color: var(--color-text-main);
  font-weight: 600;
}

.speedup-indicator {
  margin-top: 6px;
}

.speedup-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 700;
}

.speedup-tag.excellent { background: #dcfce7; color: #166534; } /* Green */
.speedup-tag.good { background: #dbeafe; color: #1e40af; } /* Blue */
.speedup-tag.moderate { background: #fff7ed; color: #9a3412; } /* Orange */
.speedup-tag.neutral { background: #f1f5f9; color: #64748b; } /* Slate */

.speedup-tag.primary {
  background: var(--color-primary);
  color: white;
}

.mobile-label {
  display: none;
}
</style>
