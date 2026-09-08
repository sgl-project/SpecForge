<script setup lang="ts">
/**
 * "Get Started in Seconds" configurator: pick hardware, installer and source
 * and get the matching install command with a copy button.
 */
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

type Hardware = 'cuda' | 'rocm' | 'npu'
type Installer = 'uv' | 'pip'
type Source = 'source' | 'pypi'

const HARDWARE: { key: Hardware; label: string }[] = [
  { key: 'cuda', label: 'NVIDIA CUDA' },
  { key: 'rocm', label: 'AMD ROCm' },
  { key: 'npu', label: 'Ascend NPU' },
]
const INSTALLERS: { key: Installer; label: string }[] = [
  { key: 'uv', label: 'uv' },
  { key: 'pip', label: 'pip' },
]
const SOURCES: { key: Source; label: string }[] = [
  { key: 'source', label: 'From source' },
  { key: 'pypi', label: 'PyPI' },
]

const hardware = ref<Hardware>('cuda')
const installer = ref<Installer>('uv')
const source = ref<Source>('source')
const copied = ref(false)

const STORAGE = 'specforge-install-config'
onMounted(() => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE) ?? '{}')
    if (HARDWARE.some((h) => h.key === saved.hardware)) hardware.value = saved.hardware
    if (INSTALLERS.some((i) => i.key === saved.installer)) installer.value = saved.installer
    if (SOURCES.some((s) => s.key === saved.source)) source.value = saved.source
  } catch {
    /* ignore */
  }
})
function persist() {
  try {
    localStorage.setItem(
      STORAGE,
      JSON.stringify({ hardware: hardware.value, installer: installer.value, source: source.value })
    )
  } catch {
    /* ignore */
  }
}

const REPO = 'https://github.com/sgl-project/SpecForge.git'

const lines = computed<string[]>(() => {
  const hw = hardware.value
  const useUv = installer.value === 'uv'
  const noDeps = hw !== 'cuda'
  const out: string[] = []

  if (hw === 'rocm') out.push('# Run inside the SGLang ROCm release container')
  if (hw === 'npu') out.push('# After installing torch and torch_npu for your CANN release')

  if (source.value === 'source') {
    out.push(`git clone ${REPO}`, 'cd SpecForge')
    if (hw === 'cuda') {
      out.push(
        useUv ? 'uv venv -p 3.11 --seed' : 'python -m venv .venv',
        'source .venv/bin/activate'
      )
    }
    const flags = ['-e', '.']
    if (noDeps) flags.push('--no-deps')
    if (useUv) out.push(`uv pip install${hw === 'cuda' ? '' : ' --system'} ${flags.join(' ')}`)
    else out.push(`${hw === 'cuda' ? 'pip' : 'python -m pip'} install ${flags.join(' ')}`)
  } else {
    const pkg = ['specforge']
    if (noDeps) pkg.push('--no-deps')
    if (useUv) out.push(`uv pip install${hw === 'cuda' ? '' : ' --system'} ${pkg.join(' ')}`)
    else out.push(`${hw === 'cuda' ? 'pip' : 'python -m pip'} install ${pkg.join(' ')}`)
  }
  return out
})

const command = computed(() => lines.value.join('\n'))

const note = computed(() => {
  switch (hardware.value) {
    case 'rocm':
      return {
        text: 'Install without dependencies so pip does not pull CUDA wheels over the ROCm PyTorch and SGLang already in the container.',
        link: '/basic_usage/AMD/amd_rocm',
        label: 'AMD ROCm tutorial',
      }
    case 'npu':
      return {
        text: 'Install the vendor-matched PyTorch, torch_npu and a compatible SGLang/Mooncake service first. The launcher detects the NPU and selects HCCL.',
        link: '/basic_usage/Ascend/ascend_npu',
        label: 'Ascend NPU tutorial',
      }
    default:
      return {
        text: 'Install a CUDA build of PyTorch that matches the host driver. Installing from source is recommended so you get the latest recipes and patches.',
        link: '/get_started/installation',
        label: 'Installation guide',
      }
  }
})

const KEYWORDS = new Set(['git', 'cd', 'uv', 'pip', 'python', 'source'])
function tokens(line: string): { t: string; c: string }[] {
  if (line.startsWith('#')) return [{ t: line, c: 'c' }]
  return line.split(' ').map((word, i) => {
    let c = ''
    if (i === 0 && KEYWORDS.has(word)) c = 'k'
    else if (word === 'install' || word === 'clone' || word === 'venv' || word === 'activate') c = 'f'
    else if (word.startsWith('-')) c = 'o'
    else if (word.startsWith('http')) c = 's'
    return { t: word, c }
  })
}

async function copy() {
  try {
    await navigator.clipboard.writeText(command.value)
    copied.value = true
    setTimeout(() => (copied.value = false), 1600)
  } catch {
    /* clipboard unavailable */
  }
}
</script>

<template>
  <div class="is">
    <div class="is-row">
      <span class="is-label">Hardware</span>
      <div class="is-pills" role="radiogroup" aria-label="Hardware">
        <button
          v-for="h in HARDWARE"
          :key="h.key"
          type="button"
          role="radio"
          class="is-pill"
          :class="{ active: hardware === h.key }"
          :aria-checked="hardware === h.key"
          @click="hardware = h.key; persist()"
        >{{ h.label }}</button>
      </div>
    </div>
    <div class="is-row">
      <span class="is-label">Installer</span>
      <div class="is-pills" role="radiogroup" aria-label="Installer">
        <button
          v-for="i in INSTALLERS"
          :key="i.key"
          type="button"
          role="radio"
          class="is-pill"
          :class="{ active: installer === i.key }"
          :aria-checked="installer === i.key"
          @click="installer = i.key; persist()"
        >{{ i.label }}</button>
      </div>
    </div>
    <div class="is-row">
      <span class="is-label">Package</span>
      <div class="is-pills" role="radiogroup" aria-label="Package source">
        <button
          v-for="s in SOURCES"
          :key="s.key"
          type="button"
          role="radio"
          class="is-pill"
          :class="{ active: source === s.key }"
          :aria-checked="source === s.key"
          @click="source = s.key; persist()"
        >{{ s.label }}</button>
      </div>
    </div>

    <div class="is-cmd-head">
      <span>Run this command:</span>
      <button type="button" class="is-copy" :class="{ copied }" :aria-label="copied ? 'Copied' : 'Copy command'" @click="copy">
        <svg v-if="!copied" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" d="M9 9h10v10H9zM5 15V5h10"/></svg>
        <svg v-else viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" d="m5 12 5 5 9-10"/></svg>
        <span>{{ copied ? 'Copied' : 'Copy' }}</span>
      </button>
    </div>
    <pre class="is-cmd" tabindex="0"><code><template v-for="(line, i) in lines" :key="i"><span class="is-line"><template v-for="(tok, j) in tokens(line)" :key="j"><span :class="tok.c ? `tk-${tok.c}` : undefined">{{ tok.t }}</span>{{ j < tokens(line).length - 1 ? ' ' : '' }}</template></span>{{ i < lines.length - 1 ? '\n' : '' }}</template></code></pre>

    <p class="is-note">
      {{ note.text }}
      <a :href="withBase(note.link + '.html')">{{ note.label }} →</a>
    </p>
  </div>
</template>

<style scoped>
.is {
  display: flex;
  flex-direction: column;
  min-width: 0;
  gap: 14px;
  padding: 24px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 16px;
  background: var(--vp-c-bg);
  box-shadow: 0 12px 40px rgba(20, 60, 90, 0.08);
}
.is-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.is-label { font-size: 14px; font-weight: 600; color: var(--vp-c-text-1); }
.is-pills { display: flex; flex-wrap: wrap; gap: 6px; }
.is-pill {
  padding: 5px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}
.is-pill:hover { color: var(--vp-c-text-1); border-color: var(--vp-c-text-3); }
.is-pill:focus-visible, .is-copy:focus-visible, .is-cmd:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}
.is-pill.active {
  background: var(--vp-c-text-1);
  border-color: var(--vp-c-text-1);
  color: var(--vp-c-bg);
}
.is-cmd-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 6px;
  padding-top: 14px;
  border-top: 1px solid var(--vp-c-divider);
  font-size: 14px;
  font-weight: 600;
}
.is-copy {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}
.is-copy:hover { color: var(--vp-c-brand-1); border-color: var(--vp-c-brand-1); }
.is-copy.copied { color: #059669; border-color: #059669; }
.is-cmd {
  margin: 0;
  padding: 14px 16px;
  border-radius: 10px;
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  line-height: 1.7;
  color: var(--vp-c-text-1);
  overflow-x: auto;
  white-space: pre;
}
.tk-c { color: var(--vp-c-text-3); font-style: italic; }
.tk-k { color: var(--vp-c-brand-1); font-weight: 600; }
.tk-f { color: #7c3aed; }
.dark .tk-f, :global(.dark) .tk-f { color: #b794f6; }
.tk-o { color: #d97706; }
.tk-s { color: var(--vp-c-text-2); }
.is-note { margin: 0; font-size: 13px; line-height: 1.6; color: var(--vp-c-text-2); }
.is-note a { color: var(--vp-c-brand-1); font-weight: 500; text-decoration: none; white-space: nowrap; }
.is-note a:hover { text-decoration: underline; }
</style>
