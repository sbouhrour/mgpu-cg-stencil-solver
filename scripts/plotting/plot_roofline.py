#!/usr/bin/env python3
"""
Roofline of the stencil SpMV kernels against cuSPARSE CSR on an A100-SXM4-80GB.

Every point is placed from two measured quantities, never from a percentage reported by the
profiler:
  - DRAM bytes per row: dram__bytes_read.sum + dram__bytes_write.sum from Nsight Compute,
    divided by the number of rows;
  - kernel time: gpu__time_duration.sum from Nsight Compute, run with --clock-control none so
    the GPU keeps its own clocks (the default locks them to base, which changes time but not bytes).
The FLOP count is the useful work of the operator, 2 x nnz, identical for both implementations
of the same matrix.

Arithmetic intensity = useful FLOP / DRAM bytes; performance = useful FLOP / time.

Hardware: 1 of 8 A100-SXM4-80GB, driver 580.105.08, CUDA 13.0, Nsight Compute 2025.3.1.
Values are medians over the profiled launches (see docs/profiling-2d.md for the raw table).
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# A100-SXM4-80GB: HBM2e, 5120-bit bus at 1593 MHz (memory clock read with nvidia-smi on the node)
PEAK_MEMORY_BW = 2 * 1593e6 * 5120 / 8 / 1e9  # 2039 GB/s
# 108 SMs x 32 FP64 FMA per clock x 2 FLOP x 1.41 GHz (non-tensor-core FP64)
PEAK_FP64 = 108 * 32 * 2 * 1.41  # 9746 GFLOP/s

# name: (rows, useful FLOP per row, DRAM bytes per row, kernel time in ms)
# cuSPARSE rows include its small csr_partition kernel, launched on every call.
KERNELS = {
    "2D 5-point": {
        "rows": 100_000_000,
        "flop_per_row": 2 * 499_960_000 / 100_000_000,
        "custom": (56.0, 3.291),
        "cusparse": (82.8 + 0.5, 5.682 + 0.055),
    },
    "3D 7-point": {
        "rows": 256**3,
        "flop_per_row": 2 * (7 * 256**3 - 6 * 256**2) / 256**3,
        "custom": (81.1, 0.831),
        "cusparse": (105.0 + 0.6, 1.302 + 0.019),
    },
    "3D 27-point": {
        "rows": 256**3,
        "flop_per_row": 2 * 449_455_096 / 256**3,
        "custom": (272.1, 3.439),
        "cusparse": (353.9 + 1.7, 3.870 + 0.035),
    },
}

# Reference categorical palette, slots 1 and 2 (implementation identity)
COLOR = {"custom": "#2a78d6", "cusparse": "#eb6834"}
LABEL = {"custom": "Stencil kernel (this project)", "cusparse": "cuSPARSE CSR"}
MARKER = {"2D 5-point": "o", "3D 7-point": "s", "3D 27-point": "^"}
INK, MUTED, GRID = "#2b2b29", "#6f6e69", "#e4e3dd"


def point(k, impl):
    b_per_row, t_ms = k[impl]
    ai = k["flop_per_row"] / b_per_row
    gflops = k["flop_per_row"] * k["rows"] / (t_ms * 1e-3) / 1e9
    bw_pct = 100 * b_per_row * k["rows"] / (t_ms * 1e-3) / 1e9 / PEAK_MEMORY_BW
    return ai, gflops, bw_pct


# Label offsets in points, per (operator, implementation), for the zoomed panel
OFFSET = {
    ("2D 5-point", "custom"): (9, -4), ("3D 7-point", "custom"): (-14, -22),
    ("3D 27-point", "custom"): (10, -4), ("2D 5-point", "cusparse"): (-12, -22),
    ("3D 7-point", "cusparse"): (8, -16), ("3D 27-point", "cusparse"): (8, 2),
}
SHORT = {"2D 5-point": "2D 5-pt", "3D 7-point": "3D 7-pt", "3D 27-point": "3D 27-pt"}


def draw_roof(ax, ai_lo, ai_hi):
    ai = np.logspace(np.log10(ai_lo), np.log10(ai_hi), 400)
    ax.loglog(ai, np.minimum(PEAK_MEMORY_BW * ai, PEAK_FP64), color=INK, linewidth=2, zorder=2)


def style(ax):
    ax.grid(True, which="both", color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=9, which="both")
    for s in ax.spines.values():
        s.set_color(GRID)


def main():
    fig, (ax, az) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("white")
    pts = {(n, i): point(k, i) for n, k in KERNELS.items() for i in ("custom", "cusparse")}

    # Left: the whole roofline, to show how far below the ridge SpMV sits
    draw_roof(ax, 0.01, 100)
    ridge = PEAK_FP64 / PEAK_MEMORY_BW
    ax.axvline(ridge, color=GRID, linewidth=1, linestyle="--", zorder=1)
    ax.text(ridge * 1.1, 60, f"ridge {ridge:.1f} FLOP/B", fontsize=8, color=MUTED)
    ax.text(0.012, PEAK_MEMORY_BW * 0.012 * 2.6, f"DRAM {PEAK_MEMORY_BW:,.0f} GB/s",
            rotation=33, fontsize=9, color=MUTED)
    ax.text(6, PEAK_FP64 * 1.2, f"FP64 {PEAK_FP64 / 1000:.1f} TFLOP/s", fontsize=9, color=MUTED)
    for (n, i), (x, y, _) in pts.items():
        ax.scatter(x, y, s=45, marker=MARKER[n], color=COLOR[i], edgecolors="white",
                   linewidths=1.5, zorder=5)
    ax.add_patch(plt.Rectangle((0.10, 130), 0.12, 290, fill=False, edgecolor=MUTED, linewidth=1))
    ax.set_xlim(0.01, 100)
    ax.set_ylim(20, 30000)
    ax.set_xlabel("Arithmetic intensity (useful FLOP per DRAM byte)", fontsize=10, color=INK)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=10, color=INK)
    ax.set_title("Full roofline: every kernel is 24-40x below the ridge", fontsize=10,
                 color=INK, loc="left")
    style(ax)

    # Right: zoom on the memory roof, labelled with the fraction of DRAM peak reached
    draw_roof(az, 0.10, 0.22)
    for (n, i), (x, y, pct) in pts.items():
        az.scatter(x, y, s=80, marker=MARKER[n], color=COLOR[i], edgecolors="white",
                   linewidths=2, zorder=5)
        az.annotate(f"{SHORT[n]}  {pct:.0f}%", xy=(x, y), xytext=OFFSET[(n, i)],
                    textcoords="offset points", fontsize=9, color=INK,
                    ha="right" if OFFSET[(n, i)][0] < 0 else "left")
    az.set_xlim(0.10, 0.22)
    az.set_ylim(130, 420)
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (az.xaxis, az.yaxis):
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(fmt)
    az.set_xlabel("Arithmetic intensity (useful FLOP per DRAM byte)", fontsize=10, color=INK)
    az.set_title("Zoom: % = achieved DRAM bandwidth / 2,039 GB/s peak", fontsize=10, color=INK,
                 loc="left")
    style(az)

    impl_handles = [plt.Line2D([], [], linestyle="", marker="o", color=COLOR[i], markersize=8,
                               label=LABEL[i]) for i in ("custom", "cusparse")]
    op_handles = [plt.Line2D([], [], linestyle="", marker=MARKER[n], color=MUTED, markersize=8,
                             label=n) for n in KERNELS]
    leg1 = az.legend(handles=impl_handles, loc="upper left", frameon=False, fontsize=9)
    az.add_artist(leg1)
    az.legend(handles=op_handles, loc="lower right", frameon=False, fontsize=9)

    fig.suptitle("SpMV on A100-SXM4-80GB, FP64: Nsight Compute DRAM bytes and kernel time",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.99, 0.01, "2D: 10k x 10k grid. 3D: 256^3 grid. CUDA 13.0, Nsight Compute 2025.3.1.",
             ha="right", fontsize=8, color=MUTED)
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    for out in ("docs/figures/roofline_spmv_comparison.png",
                "profiling/images/roofline_spmv_comparison.png"):
        plt.savefig(out, dpi=150, facecolor="white")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
