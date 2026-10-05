#!/usr/bin/env python3
"""Summarises comm_matrix.sh output: time per CG iteration for every configuration, and the speedup
of each backend / dot mode over staged MPI with host dots at the same stencil, size and GPU count.

    comm_table.py out/comm_matrix
"""
import glob
import json
import os
import re
import sys

NAME = re.compile(r"(\d+)pt_N(\d+)_np(\d+)_(\w+?)_(host|device)(?:_(\w+))?\.json$")
COLUMNS = ["staged/host", "staged/host/overlap", "staged/device", "gpuaware/host",
           "gpuaware/host/overlap", "gpuaware/device", "nccl/host", "nccl/host/overlap",
           "nccl/device", "nccl/device/graph", "nccl/device/p2poff", "nvshmem/host",
           "nvshmem/host/overlap", "nvshmem/device", "nvshmem/host/fused", "nvshmem/device/fused",
           "amgx/host/mpi", "amgx/host/mpidirect"]


def load(directory):
    runs = {}
    for path in glob.glob(os.path.join(directory, "*.json")):
        m = NAME.search(os.path.basename(path))
        if not m:
            continue
        st, n, np_, be, dots, tag = m.groups()
        d = json.load(open(path))
        iters = d["convergence"]["iterations"]
        if iters <= 0:
            continue
        col = f"{be}/{dots}" + (f"/{tag}" if tag else "")
        runs[(int(st), int(n), int(np_), col)] = d["timing"]["median_ms"] / iters * 1e3  # us/iter
    return runs


def main(directory):
    runs = load(directory)
    if not runs:
        print(f"no results in {directory}")
        return
    keys = sorted({k[:3] for k in runs})
    cols = [c for c in COLUMNS if any(k[3] == c for k in runs)]
    print("Time per CG iteration (us), and speedup over staged/host at the same stencil, N, GPUs")
    print(f"{'stencil':>7s} {'N':>4s} {'GPUs':>4s} " + " ".join(f"{c:>19s}" for c in cols))
    for st, n, np_ in keys:
        ref = runs.get((st, n, np_, "staged/host"))
        cells = []
        for c in cols:
            t = runs.get((st, n, np_, c))
            if t is None:
                cells.append(f"{'-':>19s}")
            elif ref:
                cells.append(f"{t:10.1f} ({ref / t:5.2f}x)")
            else:
                cells.append(f"{t:10.1f}        ")
        print(f"{st:>6d}p {n:>4d} {np_:>4d} " + " ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/comm_matrix")
