#!/usr/bin/env python3
"""Time per CG iteration against GPU count, one line per communication backend.

    plot_comm_backends.py --input-dir=out/comm_matrix --output=docs/figures/comm_backends.png
                          [--stencil=27] [--sizes=128,512] [--title="8x A100-SXM4 (NVLink)"]

Reads the JSON files written by scripts/benchmarking/comm_matrix.sh, whose names carry the
configuration (e.g. 27pt_N128_np8_nccl_device_graph.json). One panel per grid size. Each backend
is shown in the mode the study retains for it (LINES below); on one GPU there is no halo, so every
custom-solver line starts from the same single-GPU point.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NAME = re.compile(r"(\d+)pt_N(\d+)_np(\d+)_(\w+?)_(host|device)(?:_(\w+))?\.json$")

# (column, label, color, style); columns follow comm_table.py
LINES = [
    ("staged/host", "MPI, host staging (synchronous)", "#A23B72", "-o"),
    ("staged/host/overlap", "MPI, host staging + overlap (published)", "#A23B72", "--o"),
    ("gpuaware/host", "CUDA-aware MPI", "#F18F01", "-s"),
    ("nccl/device/graph", "NCCL, device dots, CUDA graph", "#2E86AB", "-D"),
    ("nvshmem/device/fused", "NVSHMEM, halo fused into the p update", "#3B8B5A", "-^"),
    ("amgx/host/mpi", "AmgX (MPI)", "#7F7F7F", "-v"),
    ("amgx/host/mpidirect", "AmgX (MPI_DIRECT)", "#7F7F7F", "--v"),
]


def load(directory, stencil):
    runs = {}
    for path in glob.glob(os.path.join(directory, "*.json")):
        m = NAME.search(os.path.basename(path))
        if not m or int(m.group(1)) != stencil:
            continue
        _, n, np_, be, dots, tag = m.groups()
        d = json.load(open(path))
        iters = d["convergence"]["iterations"]
        if iters > 0:
            col = f"{be}/{dots}" + (f"/{tag}" if tag else "")
            runs[(int(n), int(np_), col)] = d["timing"]["median_ms"] / iters * 1e3  # us/iter
    return runs


def series(runs, n, col):
    pts = {np_: t for (nn, np_, c), t in runs.items() if nn == n and c == col}
    # One GPU: no halo, the backend is irrelevant; AmgX keeps its own single-GPU point
    if not col.startswith("amgx") and 1 not in pts and (n, 1, "staged/host") in runs:
        pts[1] = runs[(n, 1, "staged/host")]
    return sorted(pts.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="out/comm_matrix")
    ap.add_argument("--output", default="docs/figures/comm_backends.png")
    ap.add_argument("--stencil", type=int, default=27)
    ap.add_argument("--sizes", default="128,512")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    runs = load(a.input_dir, a.stencil)
    sizes = [int(s) for s in a.sizes.split(",")]

    fig, axes = plt.subplots(1, len(sizes), figsize=(7 * len(sizes), 5), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        for col, label, color, style in LINES:
            pts = series(runs, n, col)
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, style, color=color, label=label, linewidth=2 if "--" not in style else 1.5,
                    markersize=6, markeredgecolor="black", markeredgewidth=0.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_title(f"{a.stencil}-point stencil, {n}³ grid", fontsize=12)
        ax.set_xlabel("Number of GPUs", fontsize=11)
        ax.set_ylabel("Time per CG iteration (µs)", fontsize=11)
        ax.grid(alpha=0.3, linestyle="--", color="#E8E8E8")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    if a.title:
        fig.suptitle(a.title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    plt.savefig(a.output, dpi=300, bbox_inches="tight")
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
