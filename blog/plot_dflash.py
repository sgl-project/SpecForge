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

BASELINE_COLOR = "#898781"  # de-emphasis gray: context
DFLASH_COLOR = "#2a78d6"  # categorical slot 1: the series the chart is about

DATASETS = ["GSM8K", "MATH500", "HumanEval", "MT-Bench"]

# Output tokens/s from the lmsys/Qwen3.5-397B-A17B-DFlash model card
# (SGLang on 8 x B200, TP8, bfloat16, greedy, thinking enabled, max 4096 output tokens).
# rows: dataset; cols: (baseline, DFlash block size 8)
C1 = np.array(
    [
        [204.6, 689.0],
        [204.4, 762.6],
        [202.9, 752.5],
        [202.6, 545.9],
    ]
)
C32 = np.array(
    [
        [2597.7, 6158.9],
        [2615.5, 6910.5],
        [2452.7, 6666.0],
        [2537.3, 4763.2],
    ]
)

# Speedups as reported on the model card (not recomputed, to avoid rounding drift).
SPEEDUP_C1 = [3.37, 3.73, 3.71, 2.69]
SPEEDUP_C32 = [2.37, 2.64, 2.72, 1.88]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "svg.fonttype": "none",
    }
)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), facecolor=SURFACE)

width = 0.16
gap = 0.012  # ~2px surface gap between the two bars
x = np.arange(len(DATASETS))

for ax, data, speedup, title, step in (
    (axes[0], C1, SPEEDUP_C1, "Concurrency = 1", 200),
    (axes[1], C32, SPEEDUP_C32, "Concurrency = 32", 2000),
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
    ax.bar(
        x - half,
        data[:, 0],
        width,
        color=BASELINE_COLOR,
        linewidth=0,
        label="Baseline",
        zorder=3,
    )
    ax.bar(
        x + half,
        data[:, 1],
        width,
        color=DFLASH_COLOR,
        linewidth=0,
        label="DFlash block=8",
        zorder=3,
    )

    top = data.max() * 1.18
    ax.set_ylim(0, top)
    ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, color=INK2, fontsize=9.5)
    ticks = np.arange(0, top, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:,.0f}" for v in ticks])
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", pad=10, loc="left")

    # direct labels: the speedup DFlash delivers over the baseline
    for j in range(len(DATASETS)):
        ax.text(
            x[j] + half,
            data[j, 1] + top * 0.025,
            f"{speedup[j]:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK2,
            zorder=4,
        )

axes[0].set_ylabel("Output tokens / s", color=INK2, fontsize=9.5, labelpad=8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=(0.06, 0.955),
    ncol=2,
    frameon=False,
    fontsize=9.5,
    labelcolor=INK2,
    handlelength=0.9,
    handleheight=0.9,
    handletextpad=0.5,
    columnspacing=1.6,
)

fig.suptitle(
    "Qwen3.5-397B-A17B DFlash (block size 8) vs. autoregressive baseline — 8 × B200",
    x=0.06,
    y=1.02,
    ha="left",
    fontsize=12.5,
    color=INK,
    fontweight="bold",
)

fig.subplots_adjust(left=0.095, right=0.995, top=0.76, bottom=0.09, wspace=0.18)
fig.savefig(
    "/data/projects/speculative/SpecForge/blog/dflash-speedup.svg",
    facecolor=SURFACE,
    bbox_inches="tight",
)
fig.savefig(
    "/tmp/claude-0/-data-projects-speculative-SpecForge/5e63e05b-4a76-443f-9b3f-a3362bf893b8/scratchpad/dflash-speedup.png",
    facecolor=SURFACE,
    dpi=140,
    bbox_inches="tight",
)
print("ok")
