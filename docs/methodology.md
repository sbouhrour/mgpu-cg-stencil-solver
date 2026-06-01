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

**Compilation flags asymmetry.** The Custom CG and the AmgX library are not built with the same settings, and the difference favors AmgX:

1. **Optimization level.** The Custom CG is built with `-O2` (and `--ptxas-options=-O2`, below the `ptxas` default of `-O3`); the AmgX library is built `-O3` (CMake Release). The `-O3` in `external/benchmarks/amgx/Makefile` applies only to the thin benchmark wrapper, not to AmgX's kernels.

2. **Architecture targeting.** The Custom CG ships PTX for a default virtual architecture (no `-arch`/`-gencode`), JIT-compiled to SASS on first launch; the AmgX library ships native SASS for real architectures, including `sm_80` (the A100 of the Key Numbers). This is likely the heavier of the two asymmetries — though for a memory-bound double-precision SpMV its practical effect is expected to be limited, and it has not been measured here.

**Floating-point mode.** Neither build enables `--use_fast_math`, so both use default IEEE arithmetic — not a source of asymmetry. This is deliberate: strict IEEE arithmetic (no flush-to-zero, no approximate reciprocals/square-roots) preserves the precision and reproducibility that matter in production iterative solvers, at little expected cost on a memory-bound kernel.

**Consequence.** Both flag asymmetries favor AmgX, so the measured speedup is conservative: aligning the Custom CG flags (`-O3`, `-arch`) would be expected to increase the advantage, not reduce it. The magnitude is unquantified — only the direction is established — and measuring it is left as future work.

**Run benchmarks on your hardware:**
```bash
# Quick test (512×512)
./scripts/run_all.sh --quick

# Full benchmark suite
./scripts/run_all.sh --size=1000
```

Results are saved to `results/raw/` (TXT) and `results/json/` (structured data).

> **Note**: The showcase results (1.44× vs AmgX, multi-GPU scaling) were measured on 8× NVIDIA A100-SXM4-80GB with 10k-20k matrices. To reproduce those specific results, use `--size=10000` (or larger) on equivalent hardware.
