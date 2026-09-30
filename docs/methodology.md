# Methodology

This document describes how the performance results are measured: timing scope, statistical methodology, reproducibility conditions, compilation flags, and profiling tools.

For build and run instructions, see [`reproducing.md`](reproducing.md). For the full benchmark results, see [`results.md`](results.md).

> **Hardware context.** All headline results were measured on 8× NVIDIA A100-SXM4-80GB (NVLink NV12). The SpMV roofline was profiled with Nsight Compute on the same A100-SXM4-80GB model (DRAM bytes measured per kernel). See [`profiling-2d.md`](profiling-2d.md) and [`profiling-3d.md`](profiling-3d.md) for the analyses themselves.

**How results were measured:**

| Parameter | Value |
|-----------|-------|
| Runs per configuration | 10 (median reported) |
| Warmup runs | 3 (discarded) |
| Timing scope | Solver only (excludes I/O, matrix setup) |
| Convergence criterion | Relative residual < 1e-6 |
| Profiling tools | Nsight Systems (timeline), Nsight Compute (roofline) |

**Reproducibility conditions**: Identical test matrices, GPU clocks at default (no boost lock), 3 warmup runs before measurement, separate process per configuration, same binary for all runs.

**Iso-algorithm comparison.** The benchmark compares the same algorithm on both sides: both the Custom CG and AmgX run unpreconditioned Conjugate Gradient. AmgX is configured as plain CG (multi-GPU: `solver=CG`; single-GPU: `solver=PCG` with `preconditioner=NOSOLVER`, equivalent), with no multigrid or other preconditioner. The comparison therefore measures implementation efficiency on the same algorithm, not algorithmic differences: a stencil-specialized CG against a general-purpose CG, both solving the same system to the same tolerance.

**Test matrices.** Every comparison runs both sides on the same matrix, with right-hand side `b = 1` and initial guess `x0 = 0`. The kernels read the matrix values, so the cost of an iteration does not depend on them: the values set the number of iterations, which is the same on both sides of each comparison.

| Tests | Operator | Values | Iterations to relative residual 1e-6 |
|-------|----------|--------|--------------------------------------|
| 2D: SpMV, Custom CG, AmgX | 5-point Laplacian plus a unit mass term | 5 on the diagonal, -1 per neighbor | 14 from 10k×10k to 20k×20k (condition number below 9) |
| 3D, 7-point | 7-point Laplacian | 6 on the diagonal, -1 per neighbor | 261, 527, 1065 at 128³, 256³, 512³ |
| 3D, 27-point | 27-point Laplacian | 26 on the diagonal, -1 per neighbor | 151, 303, 611 at 128³, 256³, 512³ |

The matrices are built by `./bin/generate_matrix`, `generate_matrix_3d` and `generate_matrix_3d_27pt`, or in memory from a header-only file ([Matrix files](reproducing.md#matrix-files)).

**Compilation flags** (release build):
```
nvcc -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true -std=c++11
```

**Compilation flags asymmetry.** The Custom CG and the AmgX library are not built with the same settings:

1. **Optimization level.** The Custom CG is built with `-O2` (and `--ptxas-options=-O2`, below the `ptxas` default of `-O3`); the AmgX library is built `-O3` (CMake Release). The `-O3` in `external/benchmarks/amgx/Makefile` applies only to the thin benchmark wrapper, not to AmgX's kernels. **Measured effect on the GPU code: none.** Rebuilt for `sm_80` at `-O2` and at `-O3 --ptxas-options=-O3`, all 27 kernels of `spmv_bench` and `cg_solver_mgpu_stencil` produce instruction-for-instruction identical SASS; only host code differs.

2. **Architecture targeting.** The Custom CG ships PTX for a default virtual architecture (no `-arch`/`-gencode`), JIT-compiled to SASS on first launch; the AmgX library ships native SASS for real architectures, including `sm_80`. **Measured on the SpMV kernel: no effect.** On an A100-SXM4-80GB with CUDA 12.8, the stencil SpMV runs in 3.313 ms whether JIT-compiled from PTX or built natively for `sm_80` (10k×10k grid). The other kernels were not timed in both modes.

**Floating-point mode.** Neither build enables `--use_fast_math`, so both use default IEEE arithmetic: not a source of asymmetry.

**Consequence.** Neither asymmetry changes the speed of the custom SpMV kernel, so the reported speedups are neither inflated nor deflated by the build settings on the GPU side. What remains unmeasured is the host code (kernel launches, MPI calls), built at `-O2` against AmgX's `-O3`.

**Run benchmarks on your hardware:**
```bash
# Quick test (512×512)
./scripts/run_all.sh --quick

# Full benchmark suite
./scripts/run_all.sh --size=1000
```

Results are saved to `results/raw/` (TXT) and `results/json/` (structured data).

> **Note**: The published results (1.44× vs AmgX, multi-GPU scaling) were measured on 8× NVIDIA A100-SXM4-80GB with 10k-20k matrices. To reproduce those specific results, use `--size=10000` (or larger) on equivalent hardware.
