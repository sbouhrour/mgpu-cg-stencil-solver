#!/usr/bin/env python3
"""
Single-GPU SpMV: cuSPARSE CSR against the stencil kernel on an A100-SXM4-80GB.

Reads docs/results_spmv_a100.json (median of 10 runs per point) and writes
docs/figures/spmv_format_comparison_a100.png: execution time per grid size, and the speedup.
"""

import json

import matplotlib.pyplot as plt

CSR, STENCIL, INK, MUTED, GRID = "#eb6834", "#2a78d6", "#2b2b29", "#6f6e69", "#e4e3dd"


def main():
    with open("docs/results_spmv_a100.json") as f:
        rows = [r for r in json.load(f)["results"] if r["cusparse"] == "CUDA 12.8"]
    n = [r["size"] for r in rows]
    labels = [f"{s // 1000}k×{s // 1000}k" for s in n]

    fig, (ax, az) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor("white")

    ax.plot(labels, [r["csr_time_ms"] for r in rows], "o-", color=CSR, lw=2, label="cuSPARSE CSR")
    ax.plot(labels, [r["stencil5_time_ms"] for r in rows], "s-", color=STENCIL, lw=2,
            label="Stencil kernel")
    for x, r in zip(labels, rows):
        ax.annotate(f"{r['csr_time_ms']:.2f}", (x, r["csr_time_ms"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=INK)
        ax.annotate(f"{r['stencil5_time_ms']:.2f}", (x, r["stencil5_time_ms"]),
                    textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color=INK)
    ax.set_ylabel("SpMV time (ms)", color=INK)
    ax.set_ylim(0, 30)
    ax.set_title("Execution time", loc="left", color=INK)
    ax.legend(frameon=False)

    bars = az.bar(labels, [r["speedup"] for r in rows], color=STENCIL, width=0.5)
    az.axhline(1.0, color=MUTED, lw=1, ls="--")
    for b, r in zip(bars, rows):
        az.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04, f"{r['speedup']:.2f}×",
                ha="center", fontsize=10, color=INK)
    az.set_ylim(0, 2.5)
    az.set_ylabel("Speedup over cuSPARSE", color=INK)
    az.set_title("Stencil kernel speedup", loc="left", color=INK)

    for a in (ax, az):
        a.set_xlabel("Grid", color=INK)
        a.grid(True, axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED)
        for s in a.spines.values():
            s.set_color(GRID)

    fig.suptitle("SpMV on A100-SXM4-80GB, FP64, cuSPARSE of CUDA 12.8", x=0.01, ha="left",
                 color=INK, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig("docs/figures/spmv_format_comparison_a100.png", dpi=150, facecolor="white")
    print("Saved: docs/figures/spmv_format_comparison_a100.png")


if __name__ == "__main__":
    main()
