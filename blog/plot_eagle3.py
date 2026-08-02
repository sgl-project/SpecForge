import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BASELINE_COLOR = "#898781"   # de-emphasis gray: context
EAGLE3_COLOR = "#2a78d6"     # categorical slot 1: the series the chart is about

DATASETS = ["GSM8K", "MATH500", "HumanEval", "MT-Bench"]

# Output tokens/s, all at the same drafting configuration:
# --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
# rows: dataset; cols: (baseline, EAGLE3)

# benchmarks/results/stepfun-3.5-flash_results_20260731_192947.jsonl, concurrency 16
STEP = np.array([
    [1136.7, 1634.3],
    [1304.5, 2307.6],
    [1199.1, 2040.4],
    [1065.3, 1961.2],
])

# benchmarks/results/qwen3-32b_results_20260801_020842.jsonl, concurrency 16
QWEN = np.array([
    [1957.0, 4049.3],
    [2047.6, 4585.5],
    [1994.0, 3964.5],
    [1775.1, 3160.0],
])

# https://huggingface.co/AQ-MedAI/Kimi-K2.7-Code-eagle3 model card, concurrency 8
KIMI = np.array([
    [483.0, 656.0],
    [621.0, 920.0],
    [604.0, 882.0],
    [587.0, 822.0],
])

PANELS = [
    (STEP, "Step-3.5-Flash · 4 × H200 · concurrency 16", 500),
    (QWEN, "Qwen3-32B · 4 × H200 · concurrency 16", 1000),
    (KIMI, "Kimi-K2.7-Code · 8 × H200 · concurrency 8", 200),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "svg.fonttype": "none",
})

fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), facecolor=SURFACE)

width = 0.16
gap = 0.012              # ~2px surface gap between the two bars
x = np.arange(len(DATASETS))

for ax, (data, title, step) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=9)

    half = (width + gap) / 2
    ax.bar(x - half, data[:, 0], width, color=BASELINE_COLOR, linewidth=0,
           label="Baseline", zorder=3)
    ax.bar(x + half, data[:, 1], width, color=EAGLE3_COLOR, linewidth=0,
           label="EAGLE3", zorder=3)

    top = data.max() * 1.18
    ax.set_ylim(0, top)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, color=INK2, fontsize=9)
    ticks = np.arange(0, top, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:,.0f}" for v in ticks])
    ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", pad=10, loc="left")

    # direct labels: the speedup EAGLE3 delivers over the baseline
    for j in range(len(DATASETS)):
        ax.text(x[j] + half, data[j, 1] + top * 0.025,
                f"{data[j, 1] / data[j, 0]:.2f}×",
                ha="center", va="bottom", fontsize=9, color=INK2, zorder=4)

axes[0].set_ylabel("Output tokens / s", color=INK2, fontsize=9.5, labelpad=8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.955),
           ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2,
           handlelength=0.9, handleheight=0.9, handletextpad=0.5, columnspacing=1.6)

fig.suptitle("EAGLE3 draft models — output throughput vs. the autoregressive baseline",
             x=0.045, y=1.02, ha="left", fontsize=12.5, color=INK, fontweight="bold")

fig.subplots_adjust(left=0.065, right=0.995, top=0.76, bottom=0.09, wspace=0.19)
fig.savefig("/data/projects/speculative/SpecForge/blog/eagle3-speedup.svg",
            facecolor=SURFACE, bbox_inches="tight")
fig.savefig("/tmp/claude-0/-data-projects-speculative-SpecForge/5e63e05b-4a76-443f-9b3f-a3362bf893b8/scratchpad/eagle3-speedup.png",
            facecolor=SURFACE, dpi=140, bbox_inches="tight")
print("ok")
