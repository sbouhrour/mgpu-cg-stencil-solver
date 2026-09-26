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

**Compilation flags** (release build):
```
nvcc -O2 --ptxas-options=-O2 --ptxas-options=-allow-expensive-optimizations=true -std=c++11
```

**Compilation flags asymmetry.** The Custom CG and the AmgX library are not built with the same settings:

1. **Optimization level.** The Custom CG is built with `-O2` (and `--ptxas-options=-O2`, below the `ptxas` default of `-O3`); the AmgX library is built `-O3` (CMake Release). The `-O3` in `external/benchmarks/amgx/Makefile` applies only to the thin benchmark wrapper, not to AmgX's kernels. **Measured effect on the GPU code: none.** Rebuilt for `sm_80` at `-O2` and at `-O3 --ptxas-options=-O3`, all 27 kernels of `spmv_bench` and `cg_solver_mgpu_stencil` produce instruction-for-instruction identical SASS; only host code differs.

2. **Architecture targeting.** The Custom CG ships PTX for a default virtual architecture (no `-arch`/`-gencode`), JIT-compiled to SASS on first launch; the AmgX library ships native SASS for real architectures, including `sm_80`. **Measured on the SpMV kernel: no effect.** On an A100-SXM4-80GB with CUDA 12.8, the stencil SpMV runs in 3.313 ms whether JIT-compiled from PTX or built natively for `sm_80` (10k×10k grid). The other kernels were not timed in both modes.

**Floating-point mode.** Neither build enables `--use_fast_math`, so both use default IEEE arithmetic, so this is not a source of asymmetry. This is deliberate: strict IEEE arithmetic (no flush-to-zero, no approximate reciprocals/square-roots) preserves the precision and reproducibility that matter in production iterative solvers, at little expected cost on a memory-bound kernel.

**Consequence.** Neither asymmetry changes the speed of the custom SpMV kernel, so the reported speedups are neither inflated nor deflated by the build settings on the GPU side. What remains unmeasured is the host code (kernel launches, MPI calls), built at `-O2` against AmgX's `-O3`.

**Run benchmarks on your hardware:**
```bash
# Quick test (512×512)
./scripts/run_all.sh --quick

# Full benchmark suite
./scripts/run_all.sh --size=1000
```

Results are saved to `results/raw/` (TXT) and `results/json/` (structured data).

> **Note**: The showcase results (1.44× vs AmgX, multi-GPU scaling) were measured on 8× NVIDIA A100-SXM4-80GB with 10k-20k matrices. To reproduce those specific results, use `--size=10000` (or larger) on equivalent hardware.
