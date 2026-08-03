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
DSPARK_COLOR = "#2a78d6"     # categorical slot 1: the series the chart is about

TARGET = "Kimi-K3"
SETUP = "8 × B300"           # serving setup shown in the title

CONCURRENCY = [1, 2, 4, 8, 16]

# blog/combined.csv, columns:
#   dataset, concurrency, baseline tok/s, DSpark tok/s, speedup,
#   mean accepted length, baseline wall time (s), DSpark wall time (s),
#   baseline output tokens, DSpark output tokens
# rows below: one per concurrency level; cols: (baseline, DSpark, speedup as reported)
DATA = {
    "GSM8K": np.array([
        [87.42, 274.74, 3.14],
        [150.21, 397.59, 2.65],
        [240.39, 544.55, 2.27],
        [357.96, 668.72, 1.87],
        [460.39, 1085.15, 2.36],
    ]),
    "MATH500": np.array([
        [90.33, 203.79, 2.26],
        [155.30, 291.95, 1.88],
        [258.88, 411.98, 1.59],
        [395.14, 576.43, 1.46],
        [546.17, 840.97, 1.54],
    ]),
    "HumanEval": np.array([
        [90.74, 268.53, 2.96],
        [156.28, 386.45, 2.47],
        [262.06, 552.75, 2.11],
        [400.04, 749.42, 1.87],
        [545.13, 1078.93, 1.98],
    ]),
    "MBPP": np.array([
        [90.87, 248.66, 2.74],
        [157.18, 361.02, 2.30],
        [267.69, 525.21, 1.96],
        [410.59, 736.76, 1.79],
        [574.21, 1066.52, 1.86],
    ]),
    "MT-Bench": np.array([
        [90.92, 176.62, 1.94],
        [156.42, 261.00, 1.67],
        [265.08, 376.18, 1.42],
        [389.94, 546.10, 1.40],
        [546.13, 747.93, 1.37],
    ]),
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "svg.fonttype": "none",
})

fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), facecolor=SURFACE, sharey=True)
flat = axes.ravel()

width = 0.30
gap = 0.015              # ~2px surface gap between the two bars
half = (width + gap) / 2
x = np.arange(len(CONCURRENCY))

# one shared y scale across panels: the datasets are directly comparable
top = max(d[:, 1].max() for d in DATA.values()) * 1.18

for ax, (dataset, data) in zip(flat, DATA.items()):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=9)

    ax.bar(x - half, data[:, 0], width, color=BASELINE_COLOR, linewidth=0,
           label="Baseline", zorder=3)
    ax.bar(x + half, data[:, 1], width, color=DSPARK_COLOR, linewidth=0,
           label="DSpark", zorder=3)

    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, len(CONCURRENCY) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in CONCURRENCY], color=INK2, fontsize=9)
    ax.set_yticks(np.arange(0, top, 250))
    ax.set_yticklabels([f"{v:,.0f}" for v in np.arange(0, top, 250)])
    ax.set_title(dataset, color=INK, fontsize=10.5, fontweight="bold", pad=10, loc="left")

    # direct labels: the speedup DSpark delivers over the baseline
    for j in range(len(CONCURRENCY)):
        ax.text(x[j] + half, data[j, 1] + top * 0.02,
                f"{data[j, 2]:.2f}×",
                ha="center", va="bottom", fontsize=8.5, color=INK2, zorder=4)

# bottom-most panel in every column carries the x label
for ax in (flat[3], flat[4], flat[2]):
    ax.set_xlabel("Concurrency", color=INK2, fontsize=9.5, labelpad=6)

for ax in (flat[0], flat[3]):
    ax.set_ylabel("Output tokens / s", color=INK2, fontsize=9.5, labelpad=8)

# the free sixth slot carries the legend and the reading instruction
legend_ax = flat[5]
legend_ax.set_facecolor(SURFACE)
legend_ax.axis("off")
handles, labels = flat[0].get_legend_handles_labels()
legend_ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.86),
                 ncol=1, frameon=False, fontsize=10, labelcolor=INK2,
                 handlelength=0.9, handleheight=0.9, handletextpad=0.6,
                 labelspacing=0.8)
legend_ax.text(0.06, 0.42, "Bars: output throughput.\nLabels: DSpark speedup\nover the baseline.",
               transform=legend_ax.transAxes, ha="left", va="top",
               fontsize=9.5, color=MUTED, linespacing=1.6)

title = f"{TARGET} DSpark — output throughput vs. the autoregressive baseline"
if SETUP:
    title = f"{title} · {SETUP}"
fig.suptitle(title, x=0.045, y=1.0, ha="left", fontsize=12.5, color=INK, fontweight="bold")

fig.subplots_adjust(left=0.065, right=0.995, top=0.92, bottom=0.075,
                    wspace=0.12, hspace=0.32)
fig.savefig("/data/projects/speculative/SpecForge/blog/dspark-speedup.svg",
            facecolor=SURFACE, bbox_inches="tight")
fig.savefig("/tmp/claude-0/-data-projects-speculative-SpecForge/35941c0a-904a-4675-a8c7-4219521ba5d9/scratchpad/dspark-speedup.png",
            facecolor=SURFACE, dpi=140, bbox_inches="tight")
print("ok")
