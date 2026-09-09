<script setup lang="ts">
/**
 * Install configurator: pick hardware, SpecForge version, installer and
 * optional extras, and get the matching install command with a copy button.
 * Used on the landing page ("Get Started in Seconds") and embedded in
 * docs/sections/get_started/installation.md.
 *
 * Keep the generated commands in sync with pyproject.toml extras and the
 * hand-written notes in the installation guide.
 */
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'

declare const __SPECFORGE_VERSION__: string
const VERSION = __SPECFORGE_VERSION__

type Hardware = 'cuda' | 'rocm' | 'npu'
type Version = 'main' | 'release'
type Installer = 'uv' | 'pip'
type Extra = 'fa' | 'liger' | 'dev'

const HARDWARE: { key: Hardware; label: string }[] = [
  { key: 'cuda', label: 'NVIDIA CUDA' },
  { key: 'rocm', label: 'AMD ROCm' },
  { key: 'npu', label: 'Ascend NPU' },
]
const VERSIONS: { key: Version; label: string }[] = [
  { key: 'main', label: 'main (source)' },
  { key: 'release', label: `v${VERSION} (PyPI)` },
]
const INSTALLERS: { key: Installer; label: string }[] = [
  { key: 'uv', label: 'uv' },
  { key: 'pip', label: 'pip' },
]
const EXTRAS: { key: Extra; label: string; hint: string; cudaOnly?: boolean }[] = [
  { key: 'fa', label: 'flash-attn', hint: 'FlashAttention, built from source against the installed torch', cudaOnly: true },
  { key: 'liger', label: 'liger', hint: 'liger-kernel, enables model.use_liger_kernel for DFlash training' },
  { key: 'dev', label: 'dev', hint: 'pre-commit hooks for contributors' },
]

const hardware = ref<Hardware>('cuda')
const version = ref<Version>('main')
const installer = ref<Installer>('uv')
const extras = ref<Extra[]>([])
const copied = ref(false)

const STORAGE = 'specforge-install-config'
onMounted(() => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE) ?? '{}')
    if (HARDWARE.some((h) => h.key === saved.hardware)) hardware.value = saved.hardware
    if (VERSIONS.some((v) => v.key === saved.version)) version.value = saved.version
    if (INSTALLERS.some((i) => i.key === saved.installer)) installer.value = saved.installer
    if (Array.isArray(saved.extras)) {
      extras.value = saved.extras.filter((e: unknown) => EXTRAS.some((x) => x.key === e))
    }
  } catch {
    /* ignore */
  }
})
function persist() {
  try {
    localStorage.setItem(
      STORAGE,
      JSON.stringify({
        hardware: hardware.value,
        version: version.value,
        installer: installer.value,
        extras: extras.value,
      })
    )
  } catch {
    /* ignore */
  }
}

function setHardware(hw: Hardware) {
  hardware.value = hw
  // flash-attn is CUDA only.
  if (hw !== 'cuda') extras.value = extras.value.filter((e) => e !== 'fa')
  persist()
}
function toggleExtra(e: Extra) {
  extras.value = extras.value.includes(e)
    ? extras.value.filter((x) => x !== e)
    : EXTRAS.map((x) => x.key).filter((k) => k === e || extras.value.includes(k))
  persist()
}
function extraDisabled(e: { key: Extra; cudaOnly?: boolean }) {
  return !!e.cudaOnly && hardware.value !== 'cuda'
}

const REPO = 'https://github.com/sgl-project/SpecForge.git'

const lines = computed<string[]>(() => {
  const hw = hardware.value
  const useUv = installer.value === 'uv'
  const fromSource = version.value === 'main'
  const out: string[] = []

  if (hw === 'cuda') {
    // The cuda extra pins torch / sglang-kernel to CUDA 13 wheels; pre-releases
    // are allowed because sglang pins a pre-release cuda-tile wheel. flash-attn
    // is built separately so it can see the installed torch. uv routes torch
    // and sglang-kernel to the cu130 indexes via [tool.uv.sources] for a
    // source install; pip does not read that, so it gets both indexes on the
    // command line (also needed for the PyPI release under either installer).
    const spec = ['cuda', ...extras.value.filter((e) => e !== 'fa')].join(',')
    const index =
      '--extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://sgl-project.github.io/whl/cu130/'
    const pipInstall = useUv ? 'uv pip install --prerelease=allow' : 'pip install --pre'
    if (fromSource) {
      out.push(
        `git clone ${REPO}`,
        'cd SpecForge',
        useUv ? 'uv venv -p 3.11 --seed' : 'python -m venv .venv',
        'source .venv/bin/activate',
        useUv ? `${pipInstall} -e ".[${spec}]"` : `${pipInstall} -e ".[${spec}]" ${index}`
      )
    } else {
      out.push(
        useUv ? 'uv venv -p 3.11 --seed' : 'python -m venv .venv',
        'source .venv/bin/activate',
        `${pipInstall} "specforge[${spec}]" ${index}`
      )
    }
    if (extras.value.includes('fa')) {
      out.push(
        '# flash-attn builds from source against the torch installed above',
        `${useUv ? 'uv pip' : 'pip'} install ninja packaging`,
        `MAX_JOBS=8 ${useUv ? 'uv pip' : 'pip'} install flash-attn --no-build-isolation`
      )
    }
    return out
  }

  if (hw === 'npu') {
    // The npu extra pins a CPU torch plus torch_npu / triton / triton_ascend.
    // pip does not read [tool.uv.sources], so it needs the PyTorch CPU index
    // on the command line; uv only needs it for the PyPI release, which
    // carries no source routing. The NPU build of SGLang, sgl_kernel_npu and
    // hccl are not on PyPI and must already be installed from the CANN stack.
    // No --pre: the torch_npu pre-release is an exact pin, and a global --pre
    // would pull pre-release builds of unrelated packages. Python 3.11 is the
    // newest interpreter with triton_ascend wheels.
    const spec = ['npu', ...extras.value].join(',')
    const index = '--extra-index-url https://download.pytorch.org/whl/cpu'
    const pipInstall = useUv ? 'uv pip install' : 'pip install'
    out.push('# On an Ascend host with CANN, an NPU-enabled SGLang and sgl_kernel_npu installed')
    if (fromSource) {
      out.push(
        `git clone ${REPO}`,
        'cd SpecForge',
        useUv ? 'uv venv -p 3.11 --seed' : 'python -m venv .venv',
        'source .venv/bin/activate',
        useUv ? `${pipInstall} -e ".[${spec}]"` : `${pipInstall} -e ".[${spec}]" ${index}`
      )
    } else {
      out.push(
        useUv ? 'uv venv -p 3.11 --seed' : 'python -m venv .venv',
        'source .venv/bin/activate',
        `${pipInstall} "specforge[${spec}]" ${index}`
      )
    }
    return out
  }

  // ROCm: the accelerator stack (torch, SGLang) already exists in the
  // container, so install SpecForge without dependencies.
  const pip = useUv ? 'uv pip install --system' : 'python -m pip install'
  out.push('# Run inside the SGLang ROCm release container')
  if (fromSource) {
    out.push(`git clone ${REPO}`, 'cd SpecForge', `${pip} -e . --no-deps`)
  } else {
    out.push(`${pip} specforge --no-deps`)
  }
  const pkgs: string[] = []
  if (extras.value.includes('liger')) pkgs.push('liger-kernel')
  if (extras.value.includes('dev')) pkgs.push('pre-commit')
  if (pkgs.length) out.push(`${pip} ${pkgs.join(' ')}`)
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
        text: 'The npu extra pins a CPU PyTorch plus torch_npu, triton and triton_ascend from the PyTorch CPU index and PyPI. The NPU build of SGLang, sgl_kernel_npu and hccl come from your CANN stack and must be installed first. The launcher detects the NPU and selects HCCL.',
        link: '/basic_usage/Ascend/ascend_npu',
        label: 'Ascend NPU tutorial',
      }
    default:
      return {
        text:
          'The cuda extra pins CUDA 13 builds of PyTorch and sglang-kernel; CUDA 13 is the only supported NVIDIA path.' +
          (version.value === 'release'
            ? ' The hardware extras ship with the first release after 0.2.0; on older releases install from source.'
            : ' Installing from source is recommended so you get the latest recipes and patches.'),
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
    if ((i === 0 || (i === 1 && line.startsWith('MAX_JOBS'))) && KEYWORDS.has(word)) c = 'k'
    else if (word === 'install' || word === 'clone' || word === 'venv' || word === 'activate') c = 'f'
    else if (word.startsWith('-')) c = 'o'
    else if (word.startsWith('http')) c = 's'
    else if (i === 0 && word.includes('=')) c = 'o'
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
          @click="setHardware(h.key)"
        >{{ h.label }}</button>
      </div>
    </div>
    <div class="is-row">
      <span class="is-label">Version</span>
      <div class="is-pills" role="radiogroup" aria-label="SpecForge version">
        <button
          v-for="v in VERSIONS"
          :key="v.key"
          type="button"
          role="radio"
          class="is-pill"
          :class="{ active: version === v.key }"
          :aria-checked="version === v.key"
          @click="version = v.key; persist()"
        >{{ v.label }}</button>
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
      <span class="is-label">Extras</span>
      <div class="is-pills" role="group" aria-label="Optional extras">
        <button
          v-for="e in EXTRAS"
          :key="e.key"
          type="button"
          role="checkbox"
          class="is-pill is-check"
          :class="{ active: extras.includes(e.key) }"
          :aria-checked="extras.includes(e.key)"
          :disabled="extraDisabled(e)"
          :title="extraDisabled(e) ? 'flash-attn is CUDA only' : e.hint"
          @click="toggleExtra(e.key)"
        >{{ e.label }}</button>
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
/* Inside a docs page the default theme adds margins to <p>/<pre>; reset them. */
.vp-doc .is { margin: 20px 0; }
.vp-doc .is p, .vp-doc .is pre { margin: 0; }
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
.is-pill:hover:not(:disabled) { color: var(--vp-c-text-1); border-color: var(--vp-c-text-3); }
.is-pill:disabled { opacity: 0.45; cursor: not-allowed; }
.is-pill:focus-visible, .is-copy:focus-visible, .is-cmd:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}
.is-pill.active {
  background: var(--vp-c-text-1);
  border-color: var(--vp-c-text-1);
  color: var(--vp-c-bg);
}
.is-check::before {
  content: '+';
  display: inline-block;
  width: 1em;
  margin-right: 2px;
  font-weight: 600;
  opacity: 0.7;
}
.is-check.active::before { content: '✓'; opacity: 1; }
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
