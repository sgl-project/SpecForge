<template>
  <div class="filter-controls">
    <div class="filter-grid">

      <div class="filter-item">
        <label for="targetModel">Target Model</label>
        <div class="select-wrapper">
          <select
            id="targetModel"
            :value="selectedTargetModel"
            @change="$emit('update:targetModel', $event.target.value)"
          >
            <option v-for="model in targetModels" :key="model" :value="model">
              {{ model }}
            </option>
          </select>
          <div class="select-arrow">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </div>
        </div>
      </div>

      <div class="filter-item">
        <label for="draftModel">Draft Model</label>
        <div class="select-wrapper">
          <select
            id="draftModel"
            :value="selectedDraftModel"
            @change="$emit('update:draftModel', $event.target.value)"
          >
            <option value="all">All Draft Models</option>
            <option v-for="model in draftModels" :key="model" :value="model">
              {{ model === 'None' ? 'Baseline (No Draft)' : model }}
            </option>
          </select>
          <div class="select-arrow">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </div>
        </div>
      </div>

      <div class="filter-item">
        <label for="benchmark">Benchmark</label>
        <div class="select-wrapper">
          <select
            id="benchmark"
            :value="selectedBenchmark"
            @change="$emit('update:benchmark', $event.target.value)"
          >
            <option value="all">All Benchmarks</option>
            <option v-for="benchmark in benchmarks" :key="benchmark" :value="benchmark">
              {{ benchmark }}
            </option>
          </select>
          <div class="select-arrow">
             <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </div>
        </div>
      </div>

      <div class="filter-item">
        <label for="metric">Metric</label>
        <div class="select-wrapper">
          <select
            id="metric"
            :value="selectedMetric"
            @change="$emit('update:metric', $event.target.value)"
          >
            <option v-for="metric in metrics" :key="metric.value" :value="metric.value">
              {{ metric.label }}
            </option>
          </select>
          <div class="select-arrow">
             <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  targetModels: { type: Array, required: true },
  selectedTargetModel: { type: String, default: 'all' },
  draftModels: { type: Array, default: () => [] },
  selectedDraftModel: { type: String, default: 'all' },
  benchmarks: { type: Array, required: true },
  selectedBenchmark: { type: String, required: true },
  metrics: { type: Array, required: true },
  selectedMetric: { type: String, required: true }
});

defineEmits(['update:targetModel', 'update:draftModel', 'update:benchmark', 'update:metric']);
</script>

<style scoped>
.filter-controls {
  background: white;
  border-radius: var(--radius-xl);
  padding: 24px;
  box-shadow: var(--shadow-sm);
  border: 1px solid #f1f5f9;
}

.filter-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
}

.filter-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

label {
  font-size: 0.8125rem;
  font-weight: 600;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.select-wrapper {
  position: relative;
  width: 100%;
}

select {
  width: 100%;
  padding: 12px 16px;
  padding-right: 40px;
  border: 1px solid #e2e8f0;
  border-radius: var(--radius-lg);
  font-family: var(--font-sans);
  font-size: 0.9375rem;
  font-weight: 500;
  color: var(--color-text-main);
  background: #f8fafc;
  appearance: none;
  cursor: pointer;
  transition: all 0.2s ease;
}

select:hover {
  border-color: #cbd5e1;
  background: white;
}

select:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
  background: white;
}

.select-arrow {
  position: absolute;
  right: 16px;
  top: 50%;
  transform: translateY(-50%);
  pointer-events: none;
  color: var(--color-text-secondary);
  display: flex;
}

@media (max-width: 1024px) {
  .filter-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 640px) {
  .filter-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }
}
</style>
