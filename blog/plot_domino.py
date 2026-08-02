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
DOMINO_COLOR = "#2a78d6"     # categorical slot 1: the series the chart is about

DATASETS = ["GSM8K", "MATH500", "HumanEval", "MT-Bench"]

# Output tokens/s from the Qwen3.6-27B-Domino model card
# (2 x A100 80GB, TP2, BF16, FlashInfer, O4096, thinking enabled, greedy).
# rows: dataset; cols: (baseline AR, Domino bs=8)
C1 = np.array([
    [47.2, 206.0],
    [47.3, 217.7],
    [47.2, 198.5],
    [47.1, 155.7],
])
C32 = np.array([
    [862.9, 1817.5],
    [963.3, 2021.3],
    [953.4, 1879.7],
    [866.8, 1286.9],
])

# Speedups as reported on the model card (not recomputed, to avoid rounding drift).
SPEEDUP_C1 = [4.36, 4.60, 4.20, 3.31]
SPEEDUP_C32 = [2.11, 2.10, 1.97, 1.48]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "svg.fonttype": "none",
})

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), facecolor=SURFACE)

width = 0.16
gap = 0.012              # ~2px surface gap between the two bars
x = np.arange(len(DATASETS))

for ax, data, speedup, title, step in (
    (axes[0], C1, SPEEDUP_C1, "Concurrency = 1", 50),
    (axes[1], C32, SPEEDUP_C32, "Concurrency = 32", 500),
):
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
    ax.bar(x + half, data[:, 1], width, color=DOMINO_COLOR, linewidth=0,
           label="Domino block=8", zorder=3)

    top = data.max() * 1.18
    ax.set_ylim(0, top)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, color=INK2, fontsize=9.5)
    ticks = np.arange(0, top, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:,.0f}" for v in ticks])
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", pad=10, loc="left")

    # direct labels: the speedup Domino delivers over the baseline
    for j in range(len(DATASETS)):
        ax.text(x[j] + half, data[j, 1] + top * 0.025,
                f"{speedup[j]:.2f}×",
                ha="center", va="bottom", fontsize=9, color=INK2, zorder=4)

axes[0].set_ylabel("Output tokens / s", color=INK2, fontsize=9.5, labelpad=8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.955),
           ncol=2, frameon=False, fontsize=9.5, labelcolor=INK2,
           handlelength=0.9, handleheight=0.9, handletextpad=0.5, columnspacing=1.6)

fig.suptitle("Qwen3.6-27B Domino (block size 8) vs. autoregressive baseline — 2 × A100",
             x=0.06, y=1.02, ha="left", fontsize=12.5, color=INK, fontweight="bold")

fig.subplots_adjust(left=0.09, right=0.995, top=0.76, bottom=0.09, wspace=0.16)
fig.savefig("/data/projects/speculative/SpecForge/blog/domino-speedup.svg",
            facecolor=SURFACE, bbox_inches="tight")
fig.savefig("/tmp/claude-0/-data-projects-speculative-SpecForge/5e63e05b-4a76-443f-9b3f-a3362bf893b8/scratchpad/domino-speedup.png",
            facecolor=SURFACE, dpi=140, bbox_inches="tight")
print("ok")
