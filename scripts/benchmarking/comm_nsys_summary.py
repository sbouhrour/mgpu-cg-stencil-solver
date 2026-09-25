#!/usr/bin/env python3
"""Per-iteration summary of nsys captures of the CG solvers (exported to SQLite).

    comm_nsys_summary.py [--iters 20] out/comm_profile/nsys/*_r0.sqlite

Counts only what happens inside solves: NVTX ranges CG_Solver* (custom solver) or AmgX_Solve
(AmgX driver), of the most frequent kind (warmup solves of an overlap run are synchronous).
Every solve runs the same fixed number of iterations (--iters, the --max-iters the capture
used, below the iteration count at which the tolerance is reached), so dividing by
solves x iters gives per-iteration figures for every mode, CUDA graphs included (a graph runs
several iterations per NVTX iteration range).

Reports per iteration: CUDA API calls, host synchronizations (stream/device/event synchronize and
synchronous cudaMemcpy), kernels, copies by direction; then the kernels that take the most GPU
time, with their grid size and mean duration (an overlap solver's interior and boundary SpMV share
one kernel and differ by grid size).
"""
import argparse
import bisect
import collections
import os
import sqlite3

COPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
SYNC_CALLS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy")


def windows(c):
    rows = c.execute(
        "SELECT n.start, n.end, COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
        "LEFT JOIN StringIds s ON n.textId = s.id").fetchall()
    solves = [(a, b, label) for a, b, label in rows
              if label and (label.startswith("CG_Solver") or label == "AmgX_Solve") and b]
    # main() warms up with the synchronous solver even for an overlap run: keep the solver the
    # capture is about, the most frequent one
    if not solves:
        return []
    keep = collections.Counter(label for _, _, label in solves).most_common(1)[0][0]
    return sorted((a, b) for a, b, label in solves if label == keep)


def inside(w, starts, t):
    i = bisect.bisect_right(starts, t) - 1
    return i >= 0 and t <= w[i][1]


def summarise(path, iters):
    c = sqlite3.connect(path)
    w = windows(c)
    if not w:
        return f"{os.path.basename(path)}: no solve range found"
    starts = [a for a, _ in w]
    n = len(w) * iters
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    api = collections.Counter()
    for t, name in c.execute("SELECT r.start, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                             "JOIN StringIds s ON r.nameId = s.id"):
        if inside(w, starts, t):
            api[name.split("_v")[0]] += 1
    syncs = sum(api[k] for k in SYNC_CALLS)

    kern = collections.defaultdict(list)
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        for t, e, name, gx in c.execute(
                "SELECT k.start, k.end, s.value, k.gridX FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.shortName = s.id"):
            if inside(w, starts, t):
                kern[(name, gx)].append(e - t)

    copies = collections.Counter()
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for t, kind in c.execute("SELECT start, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            if inside(w, starts, t):
                copies[COPY_KIND.get(kind, str(kind))] += 1

    nk = sum(len(v) for v in kern.values())
    out = [f"== {os.path.basename(path).replace('_r0.sqlite', '')}: {len(w)} solves x {iters} it.",
           f"   per iteration: {sum(api.values()) / n:.1f} API calls, {syncs / n:.2f} host syncs, "
           f"{nk / n:.1f} kernels, copies "
           + ", ".join(f"{k} {v / n:.2f}" for k, v in sorted(copies.items()))]
    top = sorted(kern.items(), key=lambda kv: -sum(kv[1]))[:6]
    for (name, gx), d in top:
        out.append(f"   {name[:44]:44s} grid {gx:>6}  {len(d) / n:5.2f}/it  mean {sum(d) / len(d) / 1e3:8.1f} us")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for f in a.files:
        print(summarise(f, a.iters))


if __name__ == "__main__":
    main()
