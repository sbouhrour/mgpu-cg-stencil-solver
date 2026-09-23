#!/usr/bin/env python3
"""Fit a latency-bandwidth model T(n) = alpha + n / beta to nccl-tests output, and use it to
predict the communication time of one CG iteration of the Z-slab solver.

    comm_model.py out/comm_calibrate/sendrecv_np8.txt out/comm_calibrate/all_reduce_np8.txt

alpha is the median time of the messages of at most 1 KiB (pure latency); beta is the slope
between the two largest messages (bandwidth plateau). The out-of-place time column is used.

The solver prediction is an approximation, declared as such: one halo exchange is modelled as one
sendrecv of a full XY-plane (N^2 doubles), the two Z-directions being assumed to proceed
concurrently; the two dot products of an iteration are two 8-byte all-reduces.
"""
import re
import sys


def parse(path):
    """Returns [(bytes, time_us)] from an nccl-tests table (out-of-place columns)."""
    rows = []
    for line in open(path):
        f = line.split()
        if len(f) < 7 or line.lstrip().startswith("#") or not f[0].isdigit():
            continue
        # size count type redop root time algbw busbw #wrong [in-place ...]
        try:
            rows.append((int(f[0]), float(f[5])))
        except ValueError:
            continue
    return rows


def fit(rows):
    small = sorted(t for n, t in rows if n <= 1024)
    alpha = small[len(small) // 2] if small else float("nan")
    big = sorted(rows)[-2:]
    (n1, t1), (n2, t2) = big
    beta = (n2 - n1) / ((t2 - t1) * 1e-6) / 1e9 if t2 > t1 else float("nan")  # GB/s
    return alpha, beta


def predict(alpha, beta, nbytes):
    return alpha + nbytes / (beta * 1e9) * 1e6  # us


def main(paths):
    models = {}
    for p in paths:
        rows = parse(p)
        if len(rows) < 3:
            print(f"{p}: not enough rows")
            continue
        a, b = fit(rows)
        name = re.sub(r"\.txt$", "", p.split("/")[-1])
        models[name] = (a, b)
        print(f"{name:32s} alpha = {a:8.2f} us   beta = {b:8.1f} GB/s   ({len(rows)} sizes)")

    sr = next((v for k, v in models.items() if k.startswith("sendrecv")), None)
    ar = next((v for k, v in models.items() if k.startswith("all_reduce")), None)
    if not (sr and ar):
        return
    print("\nPredicted communication per CG iteration (model, not a measurement):")
    print(f"{'N':>5s} {'halo plane':>12s} {'halo (us)':>10s} {'2 dots (us)':>12s} {'total (us)':>11s}")
    for n in (128, 256, 512):
        plane = n * n * 8
        halo = predict(*sr, plane)
        dots = 2 * predict(*ar, 8)
        print(f"{n:5d} {plane / 2**20:9.2f} MiB {halo:10.1f} {dots:12.1f} {halo + dots:11.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
