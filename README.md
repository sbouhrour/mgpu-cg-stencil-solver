# Multi-GPU Conjugate Gradient Solver

[![CI](https://github.com/sbouhrour/mgpu-cg-stencil-solver/actions/workflows/ci.yml/badge.svg)](https://github.com/sbouhrour/mgpu-cg-stencil-solver/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://sbouhrour.github.io/mgpu-cg-stencil-solver/)

High-performance multi-GPU Conjugate Gradient solver for large-scale sparse linear systems using CUDA and MPI. Optimized for structured stencil grids with excellent strong scaling efficiency.

This project evaluates GPU sparse matrix–vector multiplication strategies and their impact on iterative solvers, with a focus on stencil-structured workloads common in scientific computing (PDE discretizations, CFD, FEM).

*Built by [Stéphane Bouhrour](https://github.com/sbouhrour) — GPU & parallel performance engineer, available for freelance missions ([contact](#contact)).*

📖 **[Full documentation site →](https://sbouhrour.github.io/mgpu-cg-stencil-solver/)**

## TL;DR — Key Numbers

| Metric | Result |
|--------|--------|
| **Stencil CG vs NVIDIA AmgX** | 1.40× faster (single-GPU), 1.44× faster (8 GPUs) |
| **Stencil SpMV vs cuSPARSE CSR** | 2.07× speedup on A100 80GB |
| **3D overlap (7pt/27pt)** | 88% scaling efficiency on 8 GPUs, up to 1.45× overlap gain |
| **Strong scaling efficiency** | 87–94% (2D), 88% (3D 27pt overlap) from 1→8 GPUs |
| **Problem size tested** | Up to 400M unknowns (2D 20k×20k), 134M unknowns (3D 512³) |

**Hardware**: 8× NVIDIA A100-SXM4-80GB · CUDA 12.8 · Driver 575.57

---

## Performance

Exploiting stencil structure enables consistent performance gains over generic sparse solvers, from single-GPU to multi-GPU.

| Configuration        | Custom Stencil CG | NVIDIA AmgX CG | Speedup   |
|----------------------|------------------:|---------------:|----------:|
| Single-GPU (20k×20k) |          531.4 ms |       746.7 ms | **1.40×** |
| 8 GPUs (20k×20k)     |           71.0 ms |       102.3 ms | **1.44×** |

<sub>*Median of 10 runs per configuration; 3 warmup runs discarded.*</sub>

<p align="center">
  <img src="docs/figures/performance_summary_horizontal.png" alt="Performance Summary: All Gains" width="100%">
</p>

- **SpMV kernel**: 2.07× faster than cuSPARSE CSR (single-GPU)
- **CG solver**: 1.40× faster than NVIDIA AmgX single-GPU, 1.44× at 8 GPUs (same convergence)
- **Multi-GPU strong scaling**: 7.48× on 8 GPUs at 20k×20k (93.5% parallel efficiency)
- **Near-linear 2-GPU scaling**: 1.95–1.97× (97–99% efficiency)
- **Deterministic convergence**: all configurations converge in exactly 14 iterations
- **Efficiency improves with problem size**: 86.8% (10k) → 93.5% (20k)

**Key insight**: Generic solvers cannot exploit known stencil structure for memory access and communication minimization, leading to systematic overhead even when scaling efficiently.

**Multi-GPU strong scaling:**

<p align="center">
  <img src="docs/figures/scaling_main_a100.png" alt="Multi-GPU Strong Scaling" width="100%">
</p>

**Single-GPU SpMV format comparison:**

<p align="center">
  <img src="docs/figures/spmv_format_comparison_a100.png" alt="SpMV Format Comparison" width="100%">
</p>

<details>
<summary><b>📊 Detailed Format Analysis</b></summary>

**Optimization techniques:**
- **Grouped memory accesses**: W-C-E (stride-1) before N-S (stride grid_size) for cache efficiency
- **ELLPACK-based storage**: Exploit stencil structure to eliminate col_idx indirection
- **Interior point fast path**: Direct calculation for 95% of rows (no CSR traversal)
- **Boundary fallback**: Standard CSR traversal for edge cases

**Why STENCIL5 is faster:**
1. Predictable access pattern → better L1/L2 cache utilization
2. Reduced memory traffic (no column index lookups for interior points)
3. Coalesced memory accesses for contiguous elements
4. **Granularity-matched parallelism** (see below)

**Kernel design choice — registers over shared memory:**

With only 5 non-zeros per row, the kernel uses **one thread per row with register-only computation**:
```
5 global loads → registers → 5 FMAs → 1 global store
```

This avoids:
- **Shared memory staging**: Copy overhead exceeds compute for 5 elements
- **Warp-level reductions**: 27 of 32 lanes would be idle; shuffle latency adds no value
- **Synchronization barriers**: No `__syncthreads()` needed

cuSPARSE must handle arbitrary sparsity (1-1000+ nnz/row), so it uses warp-per-row with generic reductions. For fixed 5-point stencils, the simpler approach wins.

> *Low per-row workload → registers beat shared memory.*

**Format choice — CSR over diagonal formats:**

The stencil optimization operates directly on CSR without converting to DIA/ELL/SELL formats. While diagonal formats can be efficient for regular matrices on single-GPU, they become impractical under multi-GPU domain decomposition:

- **Preserved interoperability**: CSR is the standard format for PETSc, Trilinos, AmgX, cuSPARSE, SciPy
- **No conversion overhead**: Direct integration with existing sparse workflows
- **Partition-friendly**: Row-band decomposition maps naturally to CSR; diagonal offsets break across boundaries
- **Drop-in integration**: Works with standard `.mtx` files and halo exchange patterns

> *Optimized stencil SpMV without abandoning CSR or domain decomposition flexibility.*

</details>

See [`results.md`](docs/results.md) for all benchmark tables (2D scaling, SpMV format comparison, AmgX comparison, 3D overlap).

---

## Comparison with NVIDIA AmgX

AmgX is NVIDIA's production-grade multi-GPU solver library, used here as reference implementation. To run AmgX benchmarks: `./scripts/setup/full_setup.sh --amgx` (see [AmgX build instructions](external/benchmarks/amgx/README.md)).

**Hardware**: 8× NVIDIA A100-SXM4-80GB · CUDA 12.8 · Driver 575.57 (same configuration for both solvers)

<p align="center">
  <img src="docs/figures/custom_vs_amgx_overview.png" alt="Custom CG vs NVIDIA AmgX Comparison" width="100%">
</p>

See [`results.md`](docs/results.md#2d--custom-cg-vs-nvidia-amgx) for the full comparison table (10k/15k/20k × Custom CG / AmgX × 1/8 GPUs).

**Key Findings:**
- **~40% faster at every scale**: Custom CG outperforms AmgX on both single-GPU and 8 GPUs
- **Same convergence**: Both solvers converge in 14 iterations with identical tolerance
- **Similar scaling efficiency**: 87-94% for both implementations

**Why the performance difference?**

> **TL;DR:** SpMV dominates CG performance. A stencil-aware kernel improves memory efficiency, yielding faster iterations without relying on communication overlap.

Profiling reveals that AmgX spends **48% of compute time in generic CSR SpMV**. By exploiting the known 5-point stencil structure, the custom kernel achieves 2× higher throughput—translating to 1.4× overall solver speedup.

Performance gains come from a more efficient SpMV kernel and reduced communication volume—not from compute-communication overlap. This is not a limitation of AmgX; it correctly handles arbitrary sparse matrices. The gap reflects the benefit of specialization when problem structure is known.

See [Profiling Analysis (2D)](docs/profiling-2d.md) for the Nsight Systems timeline comparison, roofline analysis, and kernel-level breakdown, and [`external/benchmarks/amgx/BENCHMARK_RESULTS.md`](external/benchmarks/amgx/BENCHMARK_RESULTS.md) for AmgX details.

---

## 3D Stencil Extension: Compute-Communication Overlap

**88% strong scaling efficiency on 8 A100 GPUs** (27-point stencil, 512³ grid, overlap solver).

The solver is extended to realistic 3D stencils (7-point and 27-point) with compute-communication overlap. Each SpMV is split into interior rows (independent of halo data, computed on `stream_compute`) and boundary rows (computed after halo arrival). Halo exchange (D2H + MPI + H2D) runs concurrently on `stream_comm`.

Best results: **1.45× overlap gain** (27pt, 256³, 8 GPUs) and **1.36×** (7pt, 512³, 8 GPUs). Larger grids and the higher-arithmetic-intensity 27-point stencil benefit most from the overlap (more interior work to hide behind communication).

See [3D Profiling Analysis](docs/profiling-3d.md) for full timelines, tables across all configurations (7pt/27pt × 128³/256³/512³ × 1/2/4/8 GPUs), strong scaling efficiency analysis, and key observations.

---

## Methodology

**How results were measured:**

| Parameter | Value |
|-----------|-------|
| Runs per configuration | 10 (median reported) |
| Warmup runs | 3 (discarded) |
| Timing scope | Solver only (excludes I/O, matrix setup) |
| Convergence criterion | Relative residual < 1e-6 |
| Profiling tools | Nsight Systems (timeline), Nsight Compute (roofline) |

Identical test matrices, GPU clocks at default, separate process per configuration. Showcase results measured on 8× NVIDIA A100-SXM4-80GB.

See [`methodology.md`](docs/methodology.md) for full reproducibility conditions, compilation flags, and statistical methodology.

---

## Technical Highlights

### Multi-GPU Architecture
- **MPI explicit staging**: D2H → MPI_Isend/Irecv → H2D for low-latency halo exchange
- **Row-band partitioning**: 1D decomposition with CSR format and halo zone exchange
- **Compute-communication overlap**: interior/boundary decomposition with dual-stream execution hides halo exchange behind SpMV computation (3D stencils)
- **Efficient reductions**: cuBLAS dot products instead of atomics (238× faster)
- **Optimized for A100**: Takes advantage of NVLink/PCIe Gen4 bandwidth

### Algorithm Features
- **Conjugate Gradient (CG)**: Iterative Krylov method for symmetric positive definite systems
- **2D/3D stencils**: Custom CUDA kernels for 5-point (2D), 7-point and 27-point (3D) finite difference discretizations
- **Interior/boundary split**: 3D solver decomposes SpMV into halo-independent interior rows and halo-dependent boundary rows for concurrent execution
- **Halo exchange**: Minimal communication (160 KB per exchange for 10k grid)
- **Convergence criterion**: Relative residual < 1e-6

### Performance Engineering
- **Compared NCCL vs MPI**: MPI staging 43% faster for small repeated messages
- **Profiling-driven**: Nsight Systems analysis to identify bottlenecks
- **Numerical stability**: Deterministic results across all GPU counts
- **Fair benchmarking**: Unified compilation flags (-O2) and consistent test methodology

---

## Quick Start

```bash
git clone https://github.com/sbouhrour/mgpu-cg-stencil-solver.git
cd mgpu-cg-stencil-solver

# Setup (auto-detects GPU, installs dependencies)
./scripts/setup/full_setup.sh

# Run the full benchmark suite
./scripts/run_all.sh

# Quick verification (~2 min)
./scripts/run_all.sh --quick
```

Results are saved to `results/raw/` (TXT), `results/json/` (structured), and `results/figures/` (plots).

See [`reproducing.md`](docs/reproducing.md) for prerequisites, AmgX comparison setup, manual build steps, custom benchmark commands, and profiling instructions.

---

## Architecture Overview

### Communication Pattern

```
Row-band partitioning (8 GPUs, 10k×10k grid):

GPU 0: rows [0, 12.5k)       ┐
GPU 1: rows [12.5k, 25k)     │
GPU 2: rows [25k, 37.5k)     │  Halo exchange:
GPU 3: rows [37.5k, 50k)     │  - 160 KB per GPU
GPU 4: rows [50k, 62.5k)     │  - MPI_Isend/Irecv
GPU 5: rows [62.5k, 75k)     │  - ~2 ms latency
GPU 6: rows [75k, 87.5k)     │
GPU 7: rows [87.5k, 100k)    ┘
```

```
Z-slab partitioning (8 GPUs, 256³ grid):

GPU 0: Z-planes [0, 32)       ┐
GPU 1: Z-planes [32, 64)      │
GPU 2: Z-planes [64, 96)      │  Halo exchange:
GPU 3: Z-planes [96, 128)     │  - 1 XY-plane per neighbor (N² doubles)
GPU 4: Z-planes [128, 160)    │  - 256² × 8 bytes = 512 KB per direction
GPU 5: Z-planes [160, 192)    │  - MPI explicit staging (D2H → MPI → H2D)
GPU 6: Z-planes [192, 224)    │
GPU 7: Z-planes [224, 256)    ┘
```

### CG Algorithm Structure

```c
1. Initial setup:
   - Partition matrix rows across GPUs
   - Exchange halo zones for initial vectors (x, r)

2. CG iteration loop (until convergence):
   a. SpMV: y = A×p (with halo exchange)
   b. Dot products: α = (r,r)/(p,y)  [MPI_Allreduce]
   c. AXPY updates: x += α×p, r -= α×y
   d. Dot products: β = (r_new,r_new)/(r_old,r_old)  [MPI_Allreduce]
   e. Vector update: p = r + β×p
   f. Convergence check: ||r||/||b|| < 1e-6

3. Gather final solution to all ranks
```

**3D overlap variant:** In overlap mode, step (a) is split into three
concurrent phases: the interior SpMV runs on stream_compute while the
halo exchange (D2H + MPI + H2D) runs on stream_comm. Boundary rows
are computed after halo arrival. This hides communication latency
behind useful computation.

**Performance characteristics:**
- **SpMV dominates** (~40-50% of total time)
- **BLAS1 operations** (AXPY, dot products): ~40-45%
- **Reductions** (MPI_Allreduce): ~10-15%
- **Halo exchange**: < 5% for large problems

---

## Repository Structure

```
├── README.md                       # This file
├── scripts/
│   ├── run_all.sh                  # ONE COMMAND to reproduce all results
│   ├── benchmarking/               # Individual benchmark scripts
│   ├── plotting/                   # Python plotting utilities
│   └── visualizations/             # README figure generation scripts
├── results/
│   ├── raw/                        # Raw benchmark outputs (TXT)
│   ├── json/                       # Structured results (JSON)
│   └── figures/                    # Generated plots (PNG)
├── profiling/
│   ├── nsys/                       # Nsight Systems timeline profiles
│   ├── ncu/                        # Nsight Compute roofline analysis
│   └── images/                     # Exported screenshots
├── src/                            # Source code
│   ├── main/                       # Entry points
│   ├── solvers/                    # CG solver implementations
│   ├── spmv/                       # SpMV kernels
│   ├── matrix/                     # Stencil Matrix Market generator
│   └── io/                         # Matrix I/O
├── include/                        # Header files
├── docs/                           # Documentation & pre-generated figures
├── external/benchmarks/amgx/       # NVIDIA AmgX comparison
└── tests/                          # Unit tests (Google Test)
```

---

## Documentation

- **[Results](docs/results.md)**: All benchmark tables — 2D scaling, SpMV comparison, AmgX comparison, 3D overlap
- **[Profiling Analysis (2D)](docs/profiling-2d.md)**: Why stencil specialization wins — kernel breakdown, roofline analysis, speedup attribution
- **[Profiling Analysis (3D)](docs/profiling-3d.md)**: Compute-communication overlap, interior/boundary decomposition
- **[Methodology](docs/methodology.md)**: Measurement protocol, statistical approach, profiling tools
- **[Reproducing the Results](docs/reproducing.md)**: Build, run, and profile on your own hardware
- **[Development](docs/development.md)**: Build system, adding kernels and solvers, running tests

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mgpu_cg_solver,
  author = {Bouhrour, Stephane},
  title = {Multi-GPU Conjugate Gradient Solver with Stencil-Aware SpMV and Compute-Communication Overlap},
  year = {2026},
  url = {https://github.com/sbouhrour/mgpu-cg-stencil-solver},
  note = {2.07× SpMV vs cuSPARSE; 1.44× CG vs NVIDIA AmgX (8× A100, 93.5% scaling); 88% scaling efficiency on 3D 27-point stencil with overlap}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Contact

**Stephane Bouhrour**
Email: bouhrour.stephane@gmail.com
GitHub: [@sbouhrour](https://github.com/sbouhrour)

For questions, issues, or collaboration opportunities, please open an issue on GitHub.
