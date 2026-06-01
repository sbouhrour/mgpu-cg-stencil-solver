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

**Compilation flags asymmetry.** The Custom CG and the AmgX library are not built with the same compilation settings, and the difference favors AmgX on two counts:

1. **Optimization level.** The Custom CG is built with `-O2` (and `--ptxas-options=-O2`, which lowers device-side optimization below the `ptxas` default of `-O3`). The AmgX library is built by `install_amgx.sh` with `CMAKE_BUILD_TYPE=Release`, i.e. `-O3` (confirmed in the configured build: `CMAKE_CUDA_FLAGS_RELEASE = -O3 -DNDEBUG`). Note: the `-O3` in `external/benchmarks/amgx/Makefile` applies only to the thin benchmark wrapper that calls the AmgX C API; it does not recompile AmgX's kernels, which are built by AmgX's own CMake.

2. **Architecture targeting.** The Custom CG is built without `-arch`/`-gencode`, so it ships PTX for a default virtual architecture that the driver JIT-compiles to native SASS on first launch. The AmgX library, by contrast, is built for real GPU architectures: `install_amgx.sh` passes `CMAKE_CUDA_ARCHITECTURES` when it detects a cloud/container environment, and AmgX's own CMake falls back to explicit architectures otherwise — so AmgX never ships generic PTX. The build used for the published numbers targeted `70;75;80;86;89;90` (including `sm_80`, the A100 of the Key Numbers), i.e. native SASS with no JIT. This second asymmetry is likely the heavier of the two. For a memory-bound double-precision SpMV the practical gap is expected to be limited — the kernel is bound by memory bandwidth, not by the instruction-level differences that architecture-specific SASS would improve — but this has not been measured here, and the direction consistently favors AmgX.

**Floating-point mode.** Neither build enables `--use_fast_math`: the Custom CG Makefile does not set it, and neither `install_amgx.sh` nor AmgX's own CMake build enables it (verified in the configured build tree). Both sides therefore use default IEEE arithmetic, so this is not a source of asymmetry. This is deliberate: strict IEEE arithmetic (no flush-to-zero, no approximate reciprocals/square-roots) preserves the precision and reproducibility that matter in production iterative solvers, at little expected cost on a memory-bound kernel — and it keeps the comparison on equal, full-precision footing.

**Consequence.** Both flag asymmetries favor AmgX, so the speedup measured against AmgX is conservative: aligning the Custom CG flags (`-O3`, `-arch` for the target architecture) would be expected to increase the measured advantage rather than reduce it, though the magnitude has not been quantified here — only the direction of the asymmetry is established. Measuring the aligned-flags delta is left as future work.

**Run benchmarks on your hardware:**
```bash
# Quick test (512×512)
./scripts/run_all.sh --quick

# Full benchmark suite
./scripts/run_all.sh --size=1000
```

Results are saved to `results/raw/` (TXT) and `results/json/` (structured data).

> **Note**: The showcase results (1.44× vs AmgX, multi-GPU scaling) were measured on 8× NVIDIA A100-SXM4-80GB with 10k-20k matrices. To reproduce those specific results, use `--size=10000` (or larger) on equivalent hardware.
