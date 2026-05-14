# Methodology

This document describes how the performance results are measured: timing scope, statistical methodology, reproducibility conditions, compilation flags, and profiling tools.

For build and run instructions, see [`reproducing.md`](reproducing.md). For the full benchmark results, see [`results.md`](results.md).

> **Hardware context.** All headline results were measured on 8× NVIDIA A100-SXM4-80GB (NVLink NV12). Profiling for roofline analysis was performed on RTX 4060 Laptop due to NCU permission constraints on shared A100 hosts. See [`profiling-2d.md`](profiling-2d.md) and [`profiling-3d.md`](profiling-3d.md) for the analyses themselves.

**How results were measured:**

| Parameter | Value |
|-----------|-------|
| Runs per configuration | 10 (median reported) |
| Warmup runs | 3 (discarded) |
| Timing scope | Solver only (excludes I/O, matrix setup) |
| Convergence criterion | Relative residual < 1e-6 |
| Profiling tools | Nsight Systems (timeline), Nsight Compute (roofline) |

**Reproducibility conditions**: Identical test matrices, GPU clocks at default (no boost lock), 3 warmup runs before measurement, separate process per configuration, same binary for all runs.

**Compilation flags** (release build):
```
nvcc -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true -std=c++11
```

**Run benchmarks on your hardware:**
```bash
# Quick test (512×512)
./scripts/run_all.sh --quick

# Full benchmark suite
./scripts/run_all.sh --size=1000
```

Results are saved to `results/raw/` (TXT) and `results/json/` (structured data).

> **Note**: The showcase results (1.44× vs AmgX, multi-GPU scaling) were measured on 8× NVIDIA A100-SXM4-80GB with 10k-20k matrices. To reproduce those specific results, use `--size=10000` (or larger) on equivalent hardware.
