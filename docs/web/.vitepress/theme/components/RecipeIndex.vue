<script setup lang="ts">
/**
 * Cards for every recipe in docs/recipes/, rendered on the Recipes page.
 * The list comes from the build-time loader in ../recipes.data.ts.
 */
import { withBase } from 'vitepress'
import { data as recipes } from '../recipes.data'
</script>

<template>
  <div class="rx">
    <ul v-if="recipes.length" class="rx-list">
      <li v-for="r in recipes" :key="r.url" class="rx-item">
        <a :href="withBase(r.url)" class="rx-card">
          <span class="rx-tags">
            <span v-if="r.method" class="rx-tag" :data-method="r.method">{{ r.method }}</span>
            <span v-if="r.topology" class="rx-tag">{{ r.topology }}</span>
          </span>
          <span class="rx-title">{{ r.title }}</span>
          <span v-if="r.description" class="rx-desc">{{ r.description }}</span>
          <span class="rx-foot">
            <span v-if="r.target" class="rx-target">
              <span class="rx-target-label">Target</span>
              {{ r.target }}
            </span>
            <span class="rx-open">Read the recipe →</span>
          </span>
        </a>
      </li>
    </ul>
    <p v-else class="rx-empty">No recipes yet. Be the first to contribute one below.</p>
  </div>
</template>

<style scoped>
.rx { margin: 24px 0 8px; }
.rx-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}
.rx-item { position: relative; margin: 0; }
.rx-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  height: 100%;
  padding: 18px 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  text-decoration: none;
  font-weight: 400;
  transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.rx-card:hover,
.rx-card:focus-visible {
  border-color: var(--vp-c-brand-2);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(43, 143, 208, 0.12);
}
.rx-card:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 3px; }

.rx-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.rx-tag {
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
}
.rx-tag[data-method] { color: #fff; border-color: transparent; background: var(--vp-c-text-3); }
.rx-tag[data-method='EAGLE3'] { background: #1f7fc0; }
.rx-tag[data-method='DFlash'], .rx-tag[data-method='DFlash2'] { background: #7c3aed; }
.rx-tag[data-method='DSpark'] { background: #d97706; }
.rx-tag[data-method='Domino'] { background: #059669; }

.rx-title {
  font-size: 17px;
  font-weight: 700;
  line-height: 1.3;
  letter-spacing: -0.01em;
  text-wrap: balance;
}
.rx-card:hover .rx-title { color: var(--vp-c-brand-1); }
.rx-desc {
  font-size: 14px;
  line-height: 1.55;
  color: var(--vp-c-text-2);
  text-wrap: pretty;
}
.rx-foot {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 6px 12px;
  margin-top: auto;
  padding-top: 12px;
  border-top: 1px dashed var(--vp-c-divider);
  font-size: 13px;
}
.rx-target { color: var(--vp-c-text-2); overflow-wrap: anywhere; }
.rx-target-label { color: var(--vp-c-text-2); font-weight: 500; margin-right: 6px; }
.rx-open { font-weight: 600; color: var(--vp-c-brand-1); white-space: nowrap; }

.rx-empty {
  margin: 0;
  padding: 32px 16px;
  text-align: center;
  border: 1px dashed var(--vp-c-divider);
  border-radius: 14px;
  color: var(--vp-c-text-2);
}
</style>
