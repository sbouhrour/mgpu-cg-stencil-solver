# Multi-GPU Conjugate Gradient Solver

[![CI](https://github.com/1fni/cuda-spmv-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/1fni/cuda-spmv-benchmark/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

High-performance multi-GPU Conjugate Gradient solver for large-scale sparse linear systems using CUDA and MPI. Optimized for structured stencil grids with excellent strong scaling efficiency.

This project evaluates GPU sparse matrix–vector multiplication strategies and their impact on iterative solvers, with a focus on stencil-structured workloads common in scientific computing (PDE discretizations, CFD, FEM).

## TL;DR — Key Numbers

| Metric | Result |
|--------|--------|
| **Stencil CG vs NVIDIA AmgX** | 1.40× faster (single-GPU), 1.44× faster (8 GPUs) |
| **Stencil SpMV vs cuSPARSE CSR** | 2.07× speedup on A100 80GB |
| **Strong scaling efficiency** | 87–94% from 1→8 GPUs |
| **Problem size tested** | Up to 400M unknowns (20k×20k stencil) |

**Hardware**: 8× NVIDIA A100-SXM4-80GB · CUDA 12.8 · Driver 575.57

---

## Performance Summary

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
- **CG solver**: 1.40× faster than NVIDIA AmgX (single-GPU, same convergence)
- **Multi-GPU CG**: 1.44× faster than NVIDIA AmgX (8 GPUs, equivalent scaling)

**Key insight**: Generic solvers cannot exploit known stencil structure for memory access and communication minimization, leading to systematic overhead even when scaling efficiently.

---

## Performance Highlights

**Multi-GPU Strong Scaling** on 8× NVIDIA A100-SXM4-80GB

| Problem Size | 1 GPU | 8 GPUs | Speedup | Efficiency |
|--------------|-------|--------|---------|------------|
| **100M unknowns** (10k×10k stencil) | 133.9 ms | 19.3 ms | 6.94× | 86.8% |
| **225M unknowns** (15k×15k stencil) | 300.1 ms | 40.4 ms | 7.43× | 92.9% |
| **400M unknowns** (20k×20k stencil) | 531.4 ms | 71.0 ms | **7.48×** | **93.5%** |

<sub>*Median of 10 runs; 3 warmup runs discarded.*</sub>

**Key Results:**
- **7.48× speedup** on 400M unknowns with 8 GPUs (93.5% parallel efficiency)
- **Near-linear 2-GPU scaling**: 1.95-1.97× speedup (97-99% efficiency)
- **Deterministic convergence**: All configurations converge in exactly 14 iterations
- **Better scaling with larger problems**: Efficiency improves from 86.8% to 93.5%

### Strong Scaling Visualization

<p align="center">
  <img src="docs/figures/scaling_main_a100.png" alt="Multi-GPU Strong Scaling" width="100%">
</p>

<details>
<summary><b>📊 Detailed Analysis</b></summary>

<p align="center">
  <img src="docs/figures/scaling_detailed_a100.png" alt="Detailed Scaling Analysis" width="100%">
</p>

**Performance breakdown:**
- SpMV kernel time scales near-linearly with GPU count
- MPI communication overhead remains < 10% at 8 GPUs
- Reductions (cuBLAS dot products) maintain high efficiency
- Larger problems amortize communication cost more effectively

</details>

<details>
<summary><b>📈 Problem Size Scaling</b></summary>

<p align="center">
  <img src="docs/figures/problem_size_scaling_overview.png" alt="Problem Size Scaling" width="100%">
</p>

**Observations:**
- 20k×20k matrix achieves best parallel efficiency (93.5%)
- Communication overhead decreases relative to computation for larger problems
- All problem sizes maintain > 85% efficiency at 8 GPUs

See [detailed problem size analysis](docs/PROBLEM_SIZE_SCALING_RESULTS.md) for complete results.

</details>

### Single-GPU SpMV Performance

**Format Comparison** on NVIDIA A100 80GB PCIe

<p align="center">
  <img src="docs/figures/spmv_format_comparison_a100.png" alt="SpMV Format Comparison" width="100%">
</p>

| Matrix Size | CSR (cuSPARSE) | STENCIL5 (Custom) | Speedup | Bandwidth Improvement |
|-------------|----------------|-------------------|---------|----------------------|
| **10k×10k** (100M unknowns) | 6.77 ms | 3.25 ms | **2.08×** | 1.98× (1182 → 2339 GB/s) |
| **15k×15k** (225M unknowns) | 15.00 ms | 7.29 ms | **2.06×** | 1.96× (1200 → 2346 GB/s) |
| **20k×20k** (400M unknowns) | 26.77 ms | 12.86 ms | **2.08×** | 1.98× (1195 → 2364 GB/s) |

<sub>*Median of 10 runs; 3 warmup runs discarded.*</sub>

**Key Results:**
- **2.07× average speedup** over cuSPARSE CSR implementation
- **~310 GFLOPS sustained** across all problem sizes (STENCIL5)
- **1.97× bandwidth improvement** through optimized memory access patterns
- **Consistent performance scaling**: Speedup stable at 2.06-2.08× from 100M to 400M unknowns

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

### Comparison with NVIDIA AmgX

AmgX is NVIDIA's production-grade multi-GPU solver library, used here as reference implementation. To run AmgX benchmarks: `./scripts/setup/full_setup.sh --amgx` (see [AmgX build instructions](external/benchmarks/amgx/README.md)).

**Hardware**: 8× NVIDIA A100-SXM4-80GB · CUDA 12.8 · Driver 575.57 (same configuration for both solvers)

<p align="center">
  <img src="docs/figures/custom_vs_amgx_overview.png" alt="Custom CG vs NVIDIA AmgX Comparison" width="100%">
</p>

| Matrix Size     | Implementation  |    1 GPU |   8 GPUs | Speedup | Efficiency |
|-----------------|-----------------|----------|----------|---------|------------|
| **10k×10k**     | Custom CG       | 133.9 ms |  19.3 ms |   6.94× |      86.8% |
| (100M unknowns) | NVIDIA AmgX     | 188.7 ms |  27.0 ms |   6.99× |      87.4% |
|                 |                 |          |          |         |            |
| **15k×15k**     | Custom CG       | 300.1 ms |  40.4 ms |   7.43× |      92.9% |
| (225M unknowns) | NVIDIA AmgX     | 420.0 ms |  57.0 ms |   7.36× |      92.0% |
|                 |                 |          |          |         |            |
| **20k×20k**     | Custom CG       | 531.4 ms |  71.0 ms |   7.48× |      93.5% |
| (400M unknowns) | NVIDIA AmgX     | 746.7 ms | 102.3 ms |   7.30× |      91.3% |

**Key Findings:**
- **~40% faster at every scale**: Custom CG outperforms AmgX on both single-GPU and 8 GPUs
- **Same convergence**: Both solvers converge in 14 iterations with identical tolerance
- **Similar scaling efficiency**: 87-94% for both implementations

**Why the performance difference?**

> **TL;DR:** SpMV dominates CG performance. A stencil-aware kernel improves memory efficiency, yielding faster iterations without relying on communication overlap.

Profiling reveals that AmgX spends **48% of compute time in generic CSR SpMV**. By exploiting the known 5-point stencil structure, the custom kernel achieves 2× higher throughput—translating to 1.4× overall solver speedup.

### Timeline Comparison (Nsight Systems)

**Custom CG** — Stencil SpMV executes faster:
<p align="center">
  <img src="docs/figures/custom_cg_nsys_profile_4k_2n.png" alt="Custom CG Timeline" width="100%">
</p>

**NVIDIA AmgX** — Generic CSR SpMV takes longer:
<p align="center">
  <img src="docs/figures/amgx_cg_nsys_profile_4k_2n.png" alt="AmgX Timeline" width="100%">
</p>

<sub>**Figure 1** — Nsight Systems timeline of one Conjugate Gradient iteration (2 MPI ranks, A100 GPU). Top: custom CG using stencil-optimized CSR SpMV. Bottom: NVIDIA AmgX under the same configuration. CUDA HW tracks show actual GPU kernel execution, MPI tracks highlight halo exchange phases. Although halo exchanges are synchronous in both cases, the custom implementation achieves shorter iteration time due to a faster SpMV kernel and reduced communication volume.</sub>

<sub>*NVTX ranges denote algorithmic phases and do not necessarily correspond to exact GPU kernel execution time; CUDA HW tracks provide the authoritative timing.*</sub>

### Why the SpMV Difference (Roofline)

<p align="center">
  <img src="docs/figures/roofline_spmv_comparison.png" alt="Roofline Analysis" width="80%">
</p>

<sub>**Figure 2** — Roofline analysis of SpMV kernels on NVIDIA A100. Both kernels are memory-bound (positioned on the bandwidth roof). The custom stencil kernel achieves **95% memory throughput** vs 67% for cuSPARSE CSR. Performance gains come from improved memory access patterns and reduced indirections, not increased arithmetic intensity.</sub>

<details>
<summary><b>📊 Detailed Kernel Breakdown</b></summary>

| Kernel Type | AmgX | Custom CG |
|-------------|-----:|----------:|
| SpMV | 48% | 41% |
| AXPY/AXPBY | 28% | 42% |
| Dot products | 10% | 16% |
| Other | 14% | <1% |

**Key optimizations:**
- **No index indirection**: Column indices computed from row index
- **Minimal halo exchange**: 160 KB per neighbor vs 800 MB with AllGather
- **Grouped memory accesses**: W-C-E (stride-1) then N-S (stride grid_size)

</details>

Performance gains come from a more efficient SpMV kernel and reduced communication volume—not from compute-communication overlap. This is not a limitation of AmgX; it correctly handles arbitrary sparse matrices. The gap reflects the benefit of specialization when problem structure is known.

See [Profiling Analysis](docs/PROFILING_ANALYSIS.md) for complete methodology and [`external/benchmarks/amgx/BENCHMARK_RESULTS.md`](external/benchmarks/amgx/BENCHMARK_RESULTS.md) for AmgX details.

---

## 3D Stencil Extension: Compute-Communication Overlap

**88% strong scaling efficiency on 8 A100 GPUs** (27-point stencil, 512³ grid, overlap solver).

This section extends the solver to realistic 3D stencils (7-point and 27-point) with compute-communication overlap via interior/boundary decomposition and dual-stream execution. Each SpMV is split into interior rows (independent of halo data) computed on `stream_compute`, while halo exchange (D2H + MPI + H2D) runs concurrently on `stream_comm`. Boundary rows are computed after halo arrival.

#### Nsight Systems Timeline — Sync vs Overlap

![Sync timeline](docs/figures/profiling_nsys_timeline_synch_512_3d_7pt_4n_a100_nv12.png)

*7-point stencil, 512³, 4 GPUs, Rank 2. One CG iteration takes 4.82 ms. The sequence is strictly serial: SpMV kernel, dot products, then `Halo_Exchange_MPI_3D` (1.57 ms) during which the GPU sits idle. The green `cudaStreamSynchronize` block on the CUDA API row shows the CPU blocking while waiting for the GPU to finish before halo exchange begins.*

![Overlap timeline](docs/figures/profiling_nsys_timeline_overlap_512_3d_7pt_4n_a100_nv12.png)

*Same configuration with `--overlap`. One CG iteration takes 3.76 ms (1.28× faster). On the [All Streams] row, the interior kernel `stencil7_overlap_subrange_kernel_3d` (blue) runs concurrently with Memcpy D2H (pink "Memc..." blocks) — the overlap in action. Simultaneously, MPI interprocess communication is visible as beige/yellow `process_vm_readv` blocks on the OS runtime libraries row and `MPI_Waitall` on the MPI row, all executing in parallel with the interior kernel. The small boundary kernels then execute after the sync point. The 4.82 → 3.76 ms reduction matches the benchmark table.*

```
stream_compute: |--- interior SpMV ---|                  |-- boundary SpMV --|
stream_comm:    |-- D2H --|-- MPI --|-- H2D --|
                                              ↑ sync point
```

### 7-Point Stencil — Sync vs Overlap

**Hardware**: 8× NVIDIA A100-SXM4-80GB (NVLink)

| Grid | GPUs | Sync (ms) | Overlap (ms) | Overlap Gain | Iterations |
|------|------|-----------|--------------|--------------|------------|
| 128³ | 1 | 73.2 | 74.0 | — | 261 |
| 128³ | 2 | 52.8 | 43.9 | 1.20× | 261 |
| 128³ | 4 | 51.4 | 46.7 | 1.10× | 261 |
| 128³ | 8 | 47.8 | 49.7 | 0.96× | 261 |
| 256³ | 1 | 970.3 | 972.4 | — | 527 |
| 256³ | 2 | 583.3 | 515.7 | 1.13× | 527 |
| 256³ | 4 | 409.0 | 318.0 | 1.29× | 527 |
| 256³ | 8 | 304.7 | 265.8 | 1.15× | 527 |
| 512³ | 1 | 15127 | 15129 | — | 1065 |
| 512³ | 2 | 8211 | 7682 | 1.07× | 1065 |
| 512³ | 4 | 5088 | 3944 | 1.29× | 1065 |
| 512³ | 8 | 3323 | 2453 | 1.36× | 1065 |

<sub>1-GPU rows show no overlap gain (no communication to hide). 128³/8GPU shows slight overhead (0.96×): per-GPU workload is too small for dual-stream overhead to pay off.</sub>

### 27-Point Stencil — Sync vs Overlap

| Grid | GPUs | Sync (ms) | Overlap (ms) | Overlap Gain | Iterations |
|------|------|-----------|--------------|--------------|------------|
| 128³ | 1 | 89.2 | 89.6 | — | 151 |
| 128³ | 2 | 57.3 | 51.1 | 1.12× | 151 |
| 128³ | 4 | 47.3 | 36.6 | 1.29× | 151 |
| 128³ | 8 | 40.5 | 33.6 | 1.21× | 151 |
| 256³ | 1 | 1315 | 1315 | — | 303 |
| 256³ | 2 | 718.9 | 680.3 | 1.06× | 303 |
| 256³ | 4 | 447.5 | 367.5 | 1.22× | 303 |
| 256³ | 8 | 294.0 | 203.5 | 1.45× | 303 |
| 512³ | 1 | 22016 | 21997 | — | 611 |
| 512³ | 2 | 11438 | 11142 | 1.03× | 611 |
| 512³ | 4 | 6461 | 5815 | 1.11× | 611 |
| 512³ | 8 | 3809 | 3110 | 1.23× | 611 |

### Strong Scaling Efficiency (overlap solver)

**7-point stencil** — speedup relative to 1-GPU sync baseline:

| Grid | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|------|-------|--------|--------|--------|
| 128³ | 1.00× | 1.69× | 1.59× | 1.49× |
| 256³ | 1.00× | 1.88× | 3.06× | 3.66× |
| 512³ | 1.00× | 1.97× | 3.84× | 6.17× |

<sub>512³ at 8 GPUs: 15127/2453 = 6.17× → 77% parallel efficiency</sub>

**27-point stencil** — speedup relative to 1-GPU sync baseline:

| Grid | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|------|-------|--------|--------|--------|
| 128³ | 1.00× | 1.75× | 2.44× | 2.66× |
| 256³ | 1.00× | 1.93× | 3.58× | 6.47× |
| 512³ | 1.00× | 1.98× | 3.79× | 7.08× |

<sub>512³ at 8 GPUs: 22016/3110 = 7.08× → **88% parallel efficiency**</sub>

### Key Observations

Overlap gain scales with both GPU count and problem size. Larger grids have a larger interior region relative to the halo boundary, giving `stream_compute` more work to hide behind halo exchange. The 27-point stencil benefits more than the 7-point stencil at the same grid size because it is more compute-intensive (27 vs 7 loads per row), which extends interior computation time and increases the fraction of communication that can be masked. Best overlap gains are 1.45× (27pt, 256³, 8 GPUs) and 1.36× (7pt, 512³, 8 GPUs).

Small workloads show diminishing returns. At 128³ on 8 GPUs the per-GPU workload is too brief to mask halo exchange latency, and the 7pt/128³/8GPU case incurs slight overhead (0.96×) from dual-stream management. 1-GPU runs confirm zero overhead: sync and overlap times are equivalent with no communication to hide.

The best scaling result — 88% parallel efficiency on 8 GPUs (27pt, 512³, overlap) — comes from combining kernel specialization with communication hiding. The Nsight timelines above show how a 4.82 ms synchronous iteration (GPU idle during halo exchange) becomes a 3.76 ms overlapped iteration, matching the 1.28× gain in the benchmark table.

### How to Reproduce

```bash
# Generate matrices
./bin/generate_matrix_3d 256 matrix/stencil3d_256.mtx
./bin/generate_matrix_3d_27pt 256 matrix/stencil3d_27pt_256.mtx

# Run sync solver
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27

# Run overlap solver
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27 --overlap

# Verify correctness
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27 --overlap --verify
```

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

> **Note**: Showcase results above were measured on 8× A100-SXM4-80GB with 10k-20k matrices. To reproduce those specific results, use `--size=10000` (or larger) on equivalent hardware.

---

## Technical Highlights

### Multi-GPU Architecture
- **MPI explicit staging**: D2H → MPI_Isend/Irecv → H2D for low-latency halo exchange
- **Row-band partitioning**: 1D decomposition with CSR format and halo zone exchange
- **Efficient reductions**: cuBLAS dot products instead of atomics (238× faster)
- **Optimized for A100**: Takes advantage of NVLink/PCIe Gen4 bandwidth

### Algorithm Features
- **Conjugate Gradient (CG)**: Iterative Krylov method for symmetric positive definite systems
- **5-point stencil**: Custom CUDA kernels for finite difference discretizations
- **Halo exchange**: Minimal communication (160 KB per exchange for 10k grid)
- **Convergence criterion**: Relative residual < 1e-6

### Performance Engineering
- **Compared NCCL vs MPI**: MPI staging 43% faster for small repeated messages
- **Profiling-driven**: Nsight Systems analysis to identify bottlenecks
- **Numerical stability**: Deterministic results across all GPU counts
- **Fair benchmarking**: Unified compilation flags (-O2) and consistent test methodology

---

## Quick Start

### Prerequisites
- **Hardware**: NVIDIA GPUs with Compute Capability ≥ 7.0
- **Software**: CUDA Toolkit ≥ 11.0, OpenMPI or MPICH, C++11 compiler

### Reproduce All Results (One Command)

```bash
git clone https://github.com/1fni/cuda-spmv-benchmark.git
cd cuda-spmv-benchmark

# Setup (auto-detects GPU, installs dependencies)
./scripts/setup/full_setup.sh            # Basic setup
./scripts/setup/full_setup.sh --amgx     # With AmgX comparison (optional, ~15 min build)

# Run benchmarks (auto-detects AmgX if installed)
./scripts/run_all.sh

# Options
./scripts/run_all.sh --quick             # Quick verification (~2 min)
./scripts/run_all.sh --size=10000        # Custom matrix size
```

### Manual Build and Run

```bash
# Build all (spmv_bench, cg_solver_mgpu_stencil) - requires MPI
make

# Build AmgX benchmarks (requires AmgX installed)
make -C external/benchmarks/amgx

# Generate 5-point stencil matrix
./bin/generate_matrix 1000 matrix/stencil_1k.mtx
```

```bash
# SpMV benchmark (single-GPU)
./bin/spmv_bench matrix/stencil_1k.mtx --mode=cusparse-csr,stencil5-csr

# CG solver (single-GPU)
mpirun -np 1 ./bin/cg_solver_mgpu_stencil matrix/stencil_1k.mtx

# CG solver (multi-GPU)
mpirun -np 2 ./bin/cg_solver_mgpu_stencil matrix/stencil_1k.mtx

# AmgX comparison (if installed)
./external/benchmarks/amgx/amgx_cg_solver matrix/stencil_1k.mtx
mpirun -np 2 ./external/benchmarks/amgx/amgx_cg_solver_mgpu matrix/stencil_1k.mtx
```

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

### CG Algorithm Structure

```c
1. Initial setup:
   - Partition matrix rows across GPUs
   - Exchange halo zones for initial vectors (x, r)

2. CG iteration loop (14 iterations):
   a. SpMV: y = A×p (with halo exchange)
   b. Dot products: α = (r,r)/(p,y)  [MPI_Allreduce]
   c. AXPY updates: x += α×p, r -= α×y
   d. Dot products: β = (r_new,r_new)/(r_old,r_old)  [MPI_Allreduce]
   e. Vector update: p = r + β×p
   f. Convergence check: ||r||/||b|| < 1e-6

3. Gather final solution to all ranks
```

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
│   └── plotting/                   # Python visualization (matplotlib)
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

- **[Profiling Analysis](docs/PROFILING_ANALYSIS.md)**: Why stencil specialization wins — kernel breakdown, roofline analysis, speedup attribution
- **[Strong Scaling Results](docs/SHOWCASE_SCALING_RESULTS.md)**: Detailed analysis of 15k×15k scaling
- **[Problem Size Scaling](docs/PROBLEM_SIZE_SCALING_RESULTS.md)**: Multi-size benchmark results
- **[Performance Summary](docs/scaling_summary.md)**: Technical metrics and talking points
- **[API Documentation](docs/API.md)**: Code structure and operator interface

---

## Benchmarking

### One Command

```bash
./scripts/run_all.sh
```

This builds all components, runs benchmarks, and saves results to:
- `results/raw/` — Raw TXT outputs
- `results/json/` — Structured JSON data
- `results/figures/` — Generated plots (after running plotting script)

### Custom Benchmarks

```bash
# Single configuration with JSON export
mpirun -np 2 ./bin/cg_solver_mgpu_stencil matrix/stencil_1k.mtx --json=custom.json

# Extract timing from JSON
jq '.timing.median_ms' custom.json
```

### Profiling with Nsight Systems

```bash
# Generate timeline report (multi-rank MPI)
nsys profile \
  --trace=cuda,nvtx,osrt,mpi \
  --trace-fork-before-exec=true \
  --stats=true \
  --cuda-memory-usage=true \
  --output=cg_profile \
  mpirun -np 2 ./bin/cg_solver_mgpu_stencil matrix/stencil_1k.mtx

# View in GUI
nsys-ui cg_profile.nsys-rep
```

---

## Development

### Build System

**Dual build approach** for flexibility:
- **Makefile**: Primary build for CUDA/MPI binaries
- **CMake**: Testing framework with Google Test

```bash
# Release build (default)
make

# Debug build with GPU debugging (-g -G)
make BUILD_TYPE=debug

# Build specific targets
make cg_solver_mgpu_stencil
make generate_matrix

# Run tests
cd tests && mkdir build && cd build
cmake .. && make && ./test_runner
```

### Adding Features

1. **New SpMV kernel**: Implement in `src/spmv/`, register in `get_operator()`
2. **New solver**: Add to `src/solvers/`, create entry point in `src/main/`
3. **Performance metrics**: Extend `benchmark_stats_mgpu_partitioned.cu`

### Testing

```bash
# All tests
./test_runner

# Specific test suite
./test_runner --gtest_filter="PartitionedSolver*"
```

---

## Requirements

- **NVIDIA GPUs**: Compute Capability ≥ 7.0 (Volta, Turing, Ampere, Hopper)
- **CUDA Toolkit**: ≥ 11.0 with cuSPARSE and cuBLAS libraries
- **MPI Implementation**: OpenMPI ≥ 4.0 or MPICH ≥ 3.3
- **C++ Compiler**: Supporting C++11 (nvcc, g++, clang++)
- **Optional**: Nsight Systems/Compute for profiling

**Tested configurations:**
- NVIDIA A100-SXM4-80GB (8 GPUs) - Primary development
- NVIDIA RTX 3090 (2 GPUs) - Validation
- NVIDIA H100 NVL (single GPU) - Compatibility

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mgpu_cg_solver,
  author = {Bouhrour, Stephane},
  title = {Multi-GPU Conjugate Gradient Solver with MPI},
  year = {2026},
  url = {https://github.com/1fni/cuda-spmv-benchmark},
  note = {7.48× speedup on 400M unknowns with 8 A100 GPUs}
}
```

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Contact

**Stephane Bouhrour**
Email: bouhrour.stephane@gmail.com
GitHub: [@1fni](https://github.com/1fni)

For questions, issues, or collaboration opportunities, please open an issue on GitHub.
