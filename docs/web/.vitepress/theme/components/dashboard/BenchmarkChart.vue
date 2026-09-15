<script setup lang="ts">
/**
 * Bar chart for the SpecBundle dashboard, rendered with ECharts.
 *
 * Two layouts:
 *  - one benchmark selected: one bar per (draft model, configuration) row;
 *  - all benchmarks: benchmarks on the x-axis, one series per row.
 *
 * ECharts touches `window`, so this component is only ever loaded on the
 * client (see `BenchmarkDashboard.vue`).
 */
import { computed } from 'vue'
import { useData } from 'vitepress'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import {
  BENCHMARKS,
  METRICS,
  fmt,
  isBaseline,
  metricValue,
  shortDraftName,
  type Benchmark,
  type MetricKey,
  type Row,
} from './data'

use([CanvasRenderer, BarChart, TitleComponent, TooltipComponent, LegendComponent, GridComponent, DataZoomComponent])

const props = defineProps<{
  rows: Row[]
  benchmark: Benchmark | 'all'
  metric: MetricKey
}>()

const { isDark } = useData()

// Series colours. SpecForge-trained drafts use the site brand blue; the
// baseline is neutral; anything else cycles through the accent palette.
const ACCENTS = ['#7c3aed', '#d97706', '#059669', '#db2777', '#0891b2', '#65a30d', '#ea580c', '#4f46e5']
const SERIES = ['#1f7fc0', '#7c3aed', '#059669', '#d97706', '#db2777', '#0891b2', '#65a30d', '#ea580c', '#4f46e5', '#0d9488']

function cssVar(name: string, fallback: string): string {
  if (typeof document === 'undefined') return fallback
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}

function rowLabel(row: Row, multiline: boolean): string {
  if (isBaseline(row)) return 'Baseline'
  const sep = multiline ? '\n' : ' '
  return `${shortDraftName(row.draftModel)}${sep}(${row.config})`
}

const option = computed(() => {
  // Read after `isDark` so the palette follows theme toggles.
  const dark = isDark.value
  const text1 = cssVar('--vp-c-text-1', dark ? '#dfdfd6' : '#3c3c43')
  const text2 = cssVar('--vp-c-text-2', dark ? '#98989f' : '#67676c')
  const divider = cssVar('--vp-c-divider', dark ? '#2e2e32' : '#e2e2e3')
  const brand = cssVar('--vp-c-brand-1', '#1f7fc0')
  const brand2 = cssVar('--vp-c-brand-3', '#4fb3e6')
  const bgElv = cssVar('--vp-c-bg-elv', dark ? '#202127' : '#ffffff')
  const baselineColor = dark ? '#5b5b63' : '#a8a8ad'
  const brandGradient = {
    type: 'linear',
    x: 0, y: 0, x2: 0, y2: 1,
    colorStops: [{ offset: 0, color: brand2 }, { offset: 1, color: brand }],
  }

  const metricInfo = METRICS.find((m) => m.value === props.metric)!
  const all = props.benchmark === 'all'
  const benchmarks: Benchmark[] = all ? [...BENCHMARKS] : [props.benchmark as Benchmark]
  const rows = props.rows.filter((r) => benchmarks.some((b) => metricValue(r, b, props.metric) !== null))

  // Single-benchmark view: SpecForge-trained drafts share the brand colour and
  // other drafts get an accent each. All-benchmarks view: every series needs
  // its own colour, so walk the palette (brand first).
  let accent = 0
  const colorFor = (row: Row) => {
    if (isBaseline(row)) return baselineColor
    if (all) return accent === 0 ? (accent++, brandGradient) : SERIES[accent++ % SERIES.length]
    if (/specforge|spec for ge/i.test(row.draftModel ?? '')) return brandGradient
    return ACCENTS[accent++ % ACCENTS.length]
  }

  const valueLabel = {
    show: true,
    position: 'top',
    distance: 2,
    fontSize: 10,
    color: text2,
    formatter: ({ value }: { value: number }) => (value ? fmt(value) : ''),
  }

  let xAxisData: string[]
  let series: Record<string, unknown>[]

  if (all) {
    xAxisData = benchmarks
    series = rows.map((row) => ({
      name: rowLabel(row, false),
      type: 'bar',
      data: benchmarks.map((b) => {
        const v = metricValue(row, b, props.metric)
        return v === null ? null : Number(v.toFixed(2))
      }),
      itemStyle: { color: colorFor(row), borderRadius: [4, 4, 0, 0] },
      // Per-bar labels only stay legible with a handful of series.
      label: { ...valueLabel, fontSize: 9, show: rows.length <= 3 },
    }))
  } else {
    const b = benchmarks[0]
    xAxisData = rows.map((row) => rowLabel(row, true))
    const colors = rows.map(colorFor)
    series = [
      {
        name: metricInfo.label,
        type: 'bar',
        barMaxWidth: 56,
        data: rows.map((row) => Number(metricValue(row, b, props.metric)!.toFixed(2))),
        itemStyle: {
          borderRadius: [6, 6, 0, 0],
          color: ({ dataIndex }: { dataIndex: number }) => colors[dataIndex],
        },
        label: { ...valueLabel, show: rows.length <= 16 },
      },
    ]
  }

  const showSlider = !all && xAxisData.length > 8

  return {
    backgroundColor: 'transparent',
    textStyle: { fontFamily: cssVar('--vp-font-family-base', 'Inter, ui-sans-serif, system-ui, sans-serif') },
    title: {
      text: all ? `${metricInfo.label} across benchmarks` : `${props.benchmark} · ${metricInfo.label}`,
      subtext: 'Measured with SGLang on NVIDIA H200. Config = batch size - steps - top-k - draft tokens.',
      left: 0,
      textStyle: { fontSize: 15, fontWeight: 600, color: text1 },
      subtextStyle: { fontSize: 12, color: text2 },
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: bgElv,
      borderColor: divider,
      textStyle: { color: text1, fontSize: 12 },
      valueFormatter: (v: number | null) => (v === null ? '–' : `${fmt(v)} ${metricInfo.short}`),
    },
    legend: {
      show: all,
      type: 'scroll',
      bottom: 0,
      left: 'center',
      itemGap: 14,
      textStyle: { fontSize: 11, color: text2 },
      pageTextStyle: { color: text2 },
      pageIconColor: brand,
      pageIconInactiveColor: divider,
    },
    grid: {
      left: 0,
      right: 12,
      top: 72,
      bottom: all ? 48 : showSlider ? 40 : 12,
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: xAxisData,
      axisLine: { lineStyle: { color: divider } },
      axisTick: { show: false },
      axisLabel: { interval: 0, rotate: all ? 0 : 28, fontSize: 10, color: text2 },
    },
    yAxis: {
      type: 'value',
      name: metricInfo.short,
      nameTextStyle: { color: text2, align: 'left', padding: [0, 0, 0, 4] },
      splitLine: { lineStyle: { type: 'dashed', color: divider } },
      axisLabel: { color: text2, fontSize: 11 },
    },
    dataZoom: [
      {
        type: 'slider',
        show: showSlider,
        start: 0,
        end: showSlider ? Math.min(100, (8 / xAxisData.length) * 100) : 100,
        bottom: 4,
        height: 16,
        borderColor: 'transparent',
        backgroundColor: cssVar('--vp-c-bg-soft', dark ? '#202127' : '#f6f6f7'),
        fillerColor: cssVar('--vp-c-brand-soft', 'rgba(43, 143, 208, 0.14)'),
        handleStyle: { color: brand },
        textStyle: { color: text2 },
      },
      { type: 'inside', zoomOnMouseWheel: false, moveOnMouseWheel: true },
    ],
    series,
  }
})
</script>

<template>
  <VChart class="bd-chart" :option="option" autoresize />
</template>

<style scoped>
.bd-chart {
  width: 100%;
  height: 460px;
}
@media (max-width: 640px) {
  .bd-chart { height: 380px; }
}
</style>
