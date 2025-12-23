<template>
  <div class="chart-wrapper">
    <v-chart :option="chartOption" :autoresize="true" style="height: 100%; width: 100%" />
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { removeSGLangPrefix } from '../utils/dataProcessor';
import { use } from 'echarts/core';
import { CanvasRenderer } from 'echarts/renderers';
import { BarChart, LineChart } from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent
} from 'echarts/components';
import VChart from 'vue-echarts';

use([
  CanvasRenderer,
  BarChart,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  DataZoomComponent
]);

const props = defineProps({
  data: { type: Array, required: true },
  benchmark: { type: String, default: 'MTBench' },
  metric: { type: String, default: 'throughput' }
});

const chartOption = computed(() => {
  const isAllBenchmarks = props.benchmark === 'all';
  const benchmarksList = isAllBenchmarks ? ['gsm8k', 'math500', 'mtbench', 'humaneval', 'livecodebench', 'financeqa', 'gpqa'] : [props.benchmark];
  const metricKey = props.metric === 'accLen' ? 'accLen' : 'throughput';
  const metricLabel = props.metric === 'accLen' ? 'Acceptance Length' : 'Throughput (tokens/s)';
  const isSpeedup = props.metric === 'speedup';

  // Filter data that has at least one valid metric
  const validData = props.data.filter((d) => {
    return benchmarksList.some(b => d.metrics[b] && d.metrics[b][metricKey] != null);
  });

  // Extract parallel config for current target model(s)
  const parallelConfigs = [...new Set(validData.map(d => d.parallelConfig).filter(c => c && c !== '-' && c !== null))];
  const parallelConfigText = parallelConfigs.length > 0 ? ` (${parallelConfigs.join(', ')})` : '';

  // Calculate values
  const series = [];
  let xAxisData = [];

  // Helper to determine color based on draft model type
  const getColor = (draftModel) => {
    const draft = draftModel ? draftModel.toLowerCase() : '';
    if (draft.includes('specbundle')) {
        return {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [{ offset: 0, color: '#4f46e5' }, { offset: 1, color: '#4338ca' }]
        };
    } else if (draft.includes('eagle')) {
        return '#10b981'; // Emerald
    } else if (draft === '-' || draft === 'none' || draft.includes('baseline')) {
        return '#94a3b8'; // Slate
    }
    return '#f59e0b'; // Amber
  };

  const getLabel = (d) => {
      const isBaseline = d.draftModel === '-' || d.draftModel === 'None' || d.draftModel.toLowerCase().includes('baseline');
      if (isBaseline) {
          return 'Baseline (No Draft Model)';
      }
      const cleanedModel = removeSGLangPrefix(d.draftModel);
      const modelName = cleanedModel.split('/').pop();
      const config = (d.config === '0-0-0' || !d.config || d.config === '-') ? '' : d.config;
      return config ? `${modelName}\n(${config})` : modelName;
  };

  if (isAllBenchmarks) {
    // PIVOTED VIEW: X-Axis = Benchmarks, Series = Configurations
    xAxisData = benchmarksList;

    // Palette for distinct series
    const palette = [
      '#06b6d4', // Cyan
      '#8b5cf6', // Violet
      '#ec4899', // Pink
      '#f59e0b', // Amber
      '#14b8a6', // Teal
      '#f43f5e', // Rose
      '#3b82f6', // Blue
      '#84cc16'  // Lime
    ];
    let colorIndex = 0;

    validData.forEach((d) => {
        const seriesName = getLabel(d).replace(/\n/g, ' '); // Flatten for legend

        // Determine Color
        const draft = d.draftModel ? d.draftModel.toLowerCase() : '';
        const isBaseline = draft === '-' || draft === 'none' || draft.includes('baseline');

        let itemStyleColor;

        if (draft.includes('specbundle')) {
           itemStyleColor = {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [{ offset: 0, color: '#4f46e5' }, { offset: 1, color: '#4338ca' }]
           };
        } else if (isBaseline) {
           itemStyleColor = '#94a3b8'; // Slate
        } else {
           itemStyleColor = palette[colorIndex % palette.length];
           colorIndex++;
        }

        const data = benchmarksList.map(b => {
            const val = d.metrics[b]?.[metricKey];
            if (val == null) return 0;
            if (isSpeedup) {
                const baseline = d.baseline && d.baseline[b] ? d.baseline[b][metricKey] : null;
                return baseline ? parseFloat((val / baseline).toFixed(2)) : 0;
            }
            return typeof val === 'number' ? parseFloat(val.toFixed(2)) : val;
        });

        series.push({
            name: seriesName,
            type: 'bar',
            data: data,
            itemStyle: {
                borderRadius: [4, 4, 0, 0],
                color: itemStyleColor
            },
            label: {
                show: true,
                position: 'top',
                fontSize: 9,
                formatter: (params) => {
                    const val = params.value;
                    return typeof val === 'number' ? val.toFixed(2) : val;
                },
                distance: 2
            }
        });
    });

  } else {
    // SINGLE BENCHMARK VIEW: X-Axis = Models
    xAxisData = validData.map(getLabel);

    const values = validData.map((d) => {
      const val = d.metrics[benchmarksList[0]][metricKey];
      if (isSpeedup) {
        const baseline = d.baseline && d.baseline[benchmarksList[0]] ? d.baseline[benchmarksList[0]][metricKey] : null;
        return baseline ? parseFloat((val / baseline).toFixed(2)) : 0;
      }
      return typeof val === 'number' ? parseFloat(val.toFixed(2)) : val;
    });

    series.push({
      name: isSpeedup ? 'Speedup' : metricLabel,
      type: 'bar',
      data: values,
      barMaxWidth: 60,
      itemStyle: {
        borderRadius: [6, 6, 0, 0],
        color: function (params) {
          const dataIndex = params.dataIndex;
          return getColor(validData[dataIndex].draftModel);
        }
      },
      label: {
        show: xAxisData.length <= 15,
        position: 'top',
        formatter: (params) => {
            const val = params.value;
            return typeof val === 'number' ? val.toFixed(2) : val;
        },
        fontSize: 10,
        color: '#64748b'
      }
    });
  }

  const displayTitle = isAllBenchmarks
    ? `Hardware: H200 ${parallelConfigText} | Metric: Throughput (tokens/s)`
    : `${benchmarksList[0]} Performance`;


  return {
    textStyle: { fontFamily: 'Inter, sans-serif' },
    title: {
      text: displayTitle,
      left: 'left',
      textStyle: {
        fontSize: 16,
        fontWeight: 600,
        color: '#1e293b'
      }
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: '#e2e8f0',
      textStyle: { color: '#334155', fontSize: 12 },
      axisPointer: { type: 'shadow' }
    },
    legend: {
      show: isAllBenchmarks,
      bottom: 0,
      left: 'center',
      width: '90%',
      itemGap: 15,
      textStyle: {
        fontSize: 11
      },
      data: isAllBenchmarks ? series.map(s => s.name) : []
    },
    grid: {
      left: 0,
      right: 10,
      bottom: 100, // Increased to fit 2 rows of legend + axis labels
      top: 60,
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: xAxisData,
      axisLine: { lineStyle: { color: '#e2e8f0' } },
      axisTick: { show: false },
      axisLabel: {
        interval: 0, // Force show all labels
        rotate: isAllBenchmarks ? 0 : 30, // Rotate 30 deg for single view
        fontSize: 10,
        color: '#64748b',
        // No truncation formatter
      }
    },
    yAxis: {
      type: 'value',
      name: isSpeedup ? 'x Baseline' : '',
      nameTextStyle: { align: 'right', color: '#94a3b8' },
      splitLine: { lineStyle: { type: 'dashed', color: '#f1f5f9' } },
      axisLabel: { color: '#64748b', fontSize: 11 }
    },
    dataZoom: [
      {
        type: 'slider',
        show: !isAllBenchmarks && xAxisData.length > 8,
        start: 0,
        end: isAllBenchmarks ? 100 : Math.min(100, (8 / xAxisData.length) * 100),
        bottom: 0,
        height: 16,
        borderColor: 'transparent',
        backgroundColor: '#f8fafc',
        fillerColor: 'rgba(79, 70, 229, 0.1)',
        handleSize: '100%',
        handleStyle: { color: '#818cf8' }
      },
      {
        type: 'inside',
        start: 0,
        end: 100,
        zoomOnMouseWheel: false,
        moveOnMouseMove: true,
        moveOnMouseWheel: true
      }
    ],
    series: series
  };
});
</script>

<style scoped>
.chart-wrapper {
  width: 100%;
  height: 100%;
}
</style>
