/**
 * Data model for the SpecBundle performance dashboard.
 *
 * The raw results live in `docs/data/specbundle_benchmarks.json`, keyed by
 * target model, then by benchmark. Every benchmark holds a list of speculative
 * decoding configurations (batch size, steps, top-k, draft tokens) with the
 * measured throughput and acceptance length for the baseline run and for each
 * draft model. `processTarget` pivots that into one row per
 * (draft model, configuration) with the metrics of every benchmark attached,
 * which is what the chart and table render.
 */

export const BASELINE_NAME = 'Wihtout EAGLE3' // sic: matches the raw data

export const BENCHMARKS = [
  'gsm8k',
  'math500',
  'mtbench',
  'humaneval',
  'livecodebench',
  'financeqa',
  'gpqa',
] as const

export type Benchmark = (typeof BENCHMARKS)[number]
export type MetricKey = 'throughput' | 'accLen' | 'speedup'

export const METRICS: { value: MetricKey; label: string; short: string }[] = [
  { value: 'throughput', label: 'Throughput (tokens/s)', short: 'tokens/s' },
  { value: 'accLen', label: 'Acceptance length', short: 'acc. len' },
  { value: 'speedup', label: 'Speedup vs. baseline', short: '× baseline' },
]

export interface Measurement {
  throughput: number
  accLen: number
}

export interface RawMetric {
  Name: string
  output_throughput: number
  accept_length: number
}

export interface RawResult {
  batch_size: number
  steps: number
  topk: number
  num_draft_tokens: number
  metrics: RawMetric[]
}

export interface RawBenchmark {
  benchmark_name: string
  results: RawResult[]
}

export type RawData = Record<string, Record<string, RawBenchmark>>

export interface Row {
  targetModel: string
  /** Draft model repo id, or `null` for the baseline run without a draft. */
  draftModel: string | null
  /** `batch-steps-topk-draftTokens`, or `baseline`. */
  config: string
  batchSize: number
  steps: number
  topk: number
  numDraftTokens: number
  metrics: Partial<Record<Benchmark, Measurement>>
  baseline: Partial<Record<Benchmark, Measurement>>
}

export function isBaseline(row: Pick<Row, 'draftModel'>): boolean {
  return row.draftModel === null
}

export function processTarget(targetModel: string, data: Record<string, RawBenchmark>): Row[] {
  const rows = new Map<string, Row>()

  for (const benchmark of Object.values(data)) {
    const name = benchmark.benchmark_name as Benchmark
    for (const result of benchmark.results ?? []) {
      const baseline = result.metrics.find((m) => m.Name === BASELINE_NAME)
      for (const metric of result.metrics) {
        const base = metric.Name === BASELINE_NAME
        const draftModel = base ? null : metric.Name
        const config = base
          ? 'baseline'
          : `${result.batch_size}-${result.steps}-${result.topk}-${result.num_draft_tokens}`
        const key = `${draftModel ?? ''}|${config}`

        let row = rows.get(key)
        if (!row) {
          row = {
            targetModel,
            draftModel,
            config,
            batchSize: result.batch_size,
            steps: result.steps,
            topk: result.topk,
            numDraftTokens: result.num_draft_tokens,
            metrics: {},
            baseline: {},
          }
          rows.set(key, row)
        }
        row.metrics[name] = { throughput: metric.output_throughput, accLen: metric.accept_length }
        if (baseline) {
          row.baseline[name] = { throughput: baseline.output_throughput, accLen: baseline.accept_length }
        }
      }
    }
  }

  // Baseline first, then draft models in insertion order.
  return [...rows.values()].sort((a, b) => Number(isBaseline(b)) - Number(isBaseline(a)))
}

export function processAll(raw: RawData): Record<string, Row[]> {
  const out: Record<string, Row[]> = {}
  for (const [target, data] of Object.entries(raw)) out[target] = processTarget(target, data)
  return out
}

/** Value of `metric` for `row` on `benchmark`, or `null` when not measured. */
export function metricValue(row: Row, benchmark: Benchmark, metric: MetricKey): number | null {
  const m = row.metrics[benchmark]
  if (!m) return null
  if (metric === 'speedup') {
    const b = row.baseline[benchmark]
    return b && b.throughput > 0 ? m.throughput / b.throughput : null
  }
  return m[metric] ?? null
}

/** Mean throughput speedup over every benchmark the row was measured on. */
export function averageSpeedup(row: Row): number | null {
  let total = 0
  let n = 0
  for (const b of BENCHMARKS) {
    const s = metricValue(row, b, 'speedup')
    if (s !== null) {
      total += s
      n++
    }
  }
  return n ? total / n : null
}

/** Strip the `SGLang-EAGLE3-` naming prefix used by the lmsys uploads. */
export function shortDraftName(draftModel: string | null): string {
  if (draftModel === null) return 'Baseline (no draft model)'
  const cleaned = draftModel.replace(/(^|\/)SGLang-EAGLE3[-/]/i, '$1')
  return cleaned.split('/').pop() || cleaned
}

export function fmt(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || Number.isNaN(value) ? '–' : value.toFixed(digits)
}
