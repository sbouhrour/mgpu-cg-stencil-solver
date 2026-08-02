# Profiling Analysis: Why Stencil Specialization Wins

This document explains **why** the custom CG solver outperforms NVIDIA AmgX, using profiling data from Nsight Systems and Nsight Compute. Both sides run unpreconditioned CG, so the speedup reflects implementation efficiency on the same algorithm, not an algorithmic difference.

> **Hardware note.** Performance numbers in this document (solver timings, kernel breakdowns, SpMV throughput) were measured on 8× NVIDIA A100-SXM4-80GB (NVLink NV12). The roofline analysis in [section 2](#2-spmv-kernel-analysis) was profiled on an RTX 4060 Laptop GPU due to NCU permission constraints on shared A100 hosts. Both kernels remain memory-bound on either architecture, so the relative comparison (95% vs 67% memory throughput) transfers; absolute GFLOP/s values reflect the RTX 4060 only.

## Executive Summary

| Finding | Impact |
|---------|--------|
| AmgX spends **48% of compute time** in generic CSR SpMV | Primary optimization target |
| Custom stencil kernel achieves **2× higher throughput** | Eliminates index indirection |
| Stencil-aware halo exchange: **one boundary row per neighbor** (N × 8 bytes) | Minimal communication overhead |
| Overall solver speedup: **1.40× single-GPU, 1.44× multi-GPU** | Consistent advantage at scale |

**Key insight**: By exploiting the known 5-point stencil structure, the custom solver removes the index indirection that dominates AmgX's SpMV (the primary source of the 2D solver speedup) and reduces halo communication to one boundary row per neighbor (a design property whose measurable payoff appears at scale — see [`profiling-3d.md`](profiling-3d.md)).

---

## 1. Kernel Distribution (Single-GPU)

### AmgX Kernel Breakdown (10k×10k, 1 GPU)

| Kernel Type | Time % | Notes |
|-------------|-------:|-------|
| cuSPARSE CSR SpMV | 48% | Generic sparse matrix-vector multiply |
| AXPY | 19% | Vector addition |
| Dot product | 10% | Inner product reductions |
| AXPBY | 9% | Scaled vector operations |
| Other | 14% | Setup, synchronization, etc. |

### Custom CG Kernel Breakdown (10k×10k, 1 GPU)

| Kernel Type | Time % | Notes |
|-------------|-------:|-------|
| Stencil SpMV | 41% | Stencil-aware kernel |
| AXPY | 29% | Vector addition |
| Dot product (cuBLAS) | 16% | cuBLAS ddot |
| AXPBY | 13% | Scaled vector operations |

*Both breakdowns measured on a single A100 to isolate kernel-level distribution from communication overhead. Multi-GPU scaling is analyzed separately in [section 3](#3-multi-gpu-scaling-analysis).*

### Observation

SpMV dominates in both implementations (~40-50% of total time), making it the primary optimization target. The custom kernel's 2× speedup on this operation drives the overall solver improvement.

---

## 2. SpMV Kernel Analysis

### Why Stencil Kernels Are Faster

The 5-point stencil discretization produces a sparse matrix with a **predictable structure**:

```
     [N]
      |
[W]--[C]--[E]
      |
     [S]
```

Each interior row has exactly 5 non-zeros at fixed offsets: `-grid_size`, `-1`, `0`, `+1`, `+grid_size`.

**Generic CSR (cuSPARSE)**:
- Must read `col_idx[]` array for every non-zero
- Indirect memory accesses → cache misses
- Cannot predict next memory location

**Stencil-aware kernel (custom)**:
- Column indices computed from row index (no lookup)
- Grouped memory accesses: W-C-E (stride-1) before N-S (stride grid_size)
- 95% of rows use fast path (interior points)

### Measured Performance (A100 80GB)

| Implementation | Time (20k×20k) | Bandwidth | Speedup |
|----------------|---------------:|-----------:|--------:|
| cuSPARSE CSR | 26.77 ms | 1195 GB/s | baseline |
| Stencil kernel | 12.86 ms | 2364 GB/s | **2.08×** |

### Roofline Analysis (Nsight Compute)

Profiled on RTX 4060 Laptop GPU (7k×7k matrix, same relative behavior):

![Roofline Comparison](figures/roofline_spmv_comparison.png)

| Kernel | Duration | Memory Throughput | Performance |
|--------|----------|-------------------|-------------|
| cuSPARSE CSR | 22.99 ms | 67% | 21.3 GFLOP/s |
| Custom Stencil | 11.25 ms | **95%** | **43.6 GFLOP/s** |

**Key observations:**
- Both kernels are **memory-bound** (positioned on the sloped part of the roofline)
- Stencil achieves **95% memory throughput** vs 67% for CSR
- The 2× speedup comes from better memory system utilization, not more compute
- CSR's index indirection creates irregular access patterns that reduce effective bandwidth

??? note "Raw Nsight Compute Screenshots"

    **cuSPARSE CSR:**

    ![cuSPARSE CSR Roofline](figures/profiling_roofline_cusparse_csr.png)

    **Custom Stencil:**

    ![Stencil Kernel Roofline](figures/profiling_roofline_stencil.png)

### Arithmetic Intensity Analysis

Both kernels are memory-bound, but the stencil kernel achieves higher effective bandwidth:

| Metric | CSR | Stencil |
|--------|----:|--------:|
| Bytes per row | 88 B | **56 B** |
| (5 values + 5 indices + 1 x + 1 y) | | (5 values + 1 x + 1 y, no indices) |
| Arithmetic intensity | 0.11 FLOP/B | **0.179 FLOP/B** |

The stencil kernel moves **36% less data** per row by eliminating index storage and lookups.

The stencil figures are measured at the DRAM boundary with Nsight Compute (5000 × 5000 grid, RTX 4060
Laptop): 1.40 GB moved for 25.0 M rows = 56.0 B/row, of which 48 B read (40 B coefficients + 8 B
amortised `x`) and 7.9 B written (`y`). An earlier revision of this table listed 48 B/row and
0.21 FLOP/B; that count omitted the `y` write. The corrected value is consistent with the 96 % DRAM
throughput reported above — 48 B/row would imply 82 %.

A full roofline treatment, including the 3D 27-point kernel, the distance to the A100 ridge point, and
what reduced precision or a matrix-free operator would change, is in
[Roofline Limits and the Precision Question](roofline-precision-limits.md).

---

## 3. Multi-GPU Scaling Analysis

### Communication Pattern Comparison

| Aspect | Custom CG | AmgX |
|--------|-----------|------|
| Halo exchange | One boundary row per neighbor (N × 8 bytes) | Generic CSR pattern |
| Method | MPI explicit staging | Internal NCCL/MPI |
| Overlap | None (synchronous) | Internal optimization |

### Halo volume in practice (10k×10k on 8 GPUs)

Applying the formula above to a concrete configuration:
- Each GPU owns ~12.5M rows
- Halo zone = 1 row = 10,000 doubles = 80 KB
- Two neighbors (top + bottom) = 160 KB total

Compare to naive AllGather: 100M doubles × 8 bytes = 800 MB (5000× more data).

### Scaling Efficiency

At 8 GPUs, the custom CG achieves a 6.94× speedup vs AmgX's 6.99× — similar parallel efficiency. The custom solver's **single-GPU advantage (1.40× at 20k×20k) is maintained at scale**, reaching 1.44× at 8 GPUs (also 20k×20k — see [`results.md`](results.md#2d-custom-cg-vs-nvidia-amgx) for the per-size table).

Full Custom CG vs AmgX comparison table (10k/15k/20k, 1 GPU and 8 GPUs) in [`results.md`](results.md#2d-custom-cg-vs-nvidia-amgx).

### Timeline Comparison (Nsight Systems)

**Custom CG Solver** (4k×4k, 2 GPUs):

![Custom CG Timeline](figures/custom_cg_nsys_profile_4k_2n.png)

**NVIDIA AmgX** (4k×4k, 2 GPUs):

![AmgX Timeline](figures/amgx_cg_nsys_profile_4k_2n.png)

**Figure** — Nsight Systems timeline of one Conjugate Gradient iteration (2 MPI ranks, A100 GPU). Top: custom CG using stencil-optimized CSR SpMV; bottom: NVIDIA AmgX under the same configuration. CUDA HW tracks show actual GPU kernel execution; MPI tracks highlight halo exchange phases. Annotations (green arrows, red rectangles) mark key phases: SpMV, halo exchange (DtoH → MPI → HtoD), and one full CG iteration. The AmgX iteration is approximately twice as long as the Custom CG, driven primarily by the longer cuSPARSE CSR SpMV kernel.

*NVTX ranges denote algorithmic phases and do not necessarily correspond to exact GPU kernel execution time; CUDA HW tracks provide the authoritative timing.*

**Key observation**: Performance gains come from a more efficient SpMV kernel and faster BLAS1 operations, not from compute-communication overlap (MPI halo exchange is synchronous in both implementations). The reduced communication volume of the stencil-aware exchange is a design property; its measurable impact appears at larger scale than the 4k×4k timeline shown here.

---

## Speedup Attribution

The custom CG's single-GPU advantage over AmgX (**1.41× at 10k×10k**, the size of the kernel breakdowns above; the headline **1.40×** refers to 20k×20k — see [`results.md`](results.md#2d-custom-cg-vs-nvidia-amgx)) comes from two measurable sources, not one:

- **SpMV specialization (primary)** — The custom stencil SpMV runs **1.65× faster in-solver** than AmgX's cuSPARSE CSR SpMV (derived from the kernel breakdowns: 41% of custom time vs 48% of AmgX time, normalized by the 1.41× overall speedup). The isolated microbenchmark shows a larger 2.08× gain; the in-solver figure is lower because cache state, launch patterns, and co-running operations differ from the isolated case.
- **A faster rest-of-solver (secondary)** — The non-SpMV operations (AXPY, dot, AXPBY) are collectively **1.24× faster in-solver**. This is consistent with operating on partitioned local vectors with coalesced access rather than AmgX's library-level operations on global vectors, though this contribution is not isolated to a single mechanism in the current measurements.

Communication volume is a design property of the stencil-aware halo exchange (one boundary row per neighbor vs generic patterns), but at the single-GPU and small-multi-GPU sizes profiled here it is not a measurable driver of the 2D speedup. Its impact appears at larger scale and is the central mechanism of the 3D overlap solver (see [`profiling-3d.md`](profiling-3d.md)).

### Theoretical vs Observed

Using Amdahl's Law with SpMV at 48% of AmgX time and the isolated 2× SpMV speedup, the predicted solver speedup if SpMV were the only optimization would be:

```
Theoretical (SpMV-only) speedup = 1 / (0.48/2 + 0.52) = 1.32×
```

The observed 1.41× exceeds this SpMV-only prediction. The gap is not measurement noise: it reflects the faster rest-of-solver quantified above (1.24× in-solver). In other words, the speedup has two contributors — a large gain on SpMV and a smaller but real gain on the BLAS1 operations — and the simple SpMV-only Amdahl model captures only the first.

---

## Methodology

### Profiling Tools

**Nsight Systems** (timeline analysis):
```bash
# Custom CG (1 GPU)
nsys profile --trace=cuda,nvtx -o custom_1gpu \
    ./bin/cg_solver_mgpu_stencil matrix/stencil_10000x10000.mtx

# Custom CG (multi-GPU)
nsys profile --trace=cuda,mpi,nvtx -o custom_mgpu \
    mpirun -np 4 ./bin/cg_solver_mgpu_stencil matrix/stencil_10000x10000.mtx

# AmgX (1 GPU)
nsys profile --trace=cuda,nvtx -o amgx_1gpu \
    ./external/benchmarks/amgx/amgx_cg_solver matrix/stencil_10000x10000.mtx
```

**Nsight Compute** (kernel analysis):
```bash
# cuSPARSE CSR roofline
ncu --set roofline -o roofline_cusparse \
    ./bin/spmv_bench matrix/stencil_10000x10000.mtx --mode=cusparse-csr

# Stencil kernel roofline
ncu --set roofline -o roofline_stencil \
    ./bin/spmv_bench matrix/stencil_10000x10000.mtx --mode=stencil5-csr
```

These commands document the profiling of this specific analysis. For general reproduction of the published numbers, see the [Reproducing](reproducing.md#profiling) page.

### Available Profile Data

| Profile | Location | Hardware |
|---------|----------|----------|
| Custom 1 GPU (10k) | `profiling/nsys/mpi_1ranks_profile_10000.nsys-rep` | A100 |
| Custom 2 GPUs (10k) | `profiling/nsys/mpi_2ranks_profile_10000.nsys-rep` | A100 |
| AmgX 1 GPU (10k) | `profiling/nsys/amgx_1ranks_profile_10000.nsys-rep` | A100 |
| AmgX 2 GPUs (10k) | `profiling/nsys/amgx_2ranks_profile_10000.nsys-rep` | A100 |
| CSR roofline | `profiling/ncu/roofline_cusparse_csr_7000_rtx4060.ncu-rep` | RTX 4060 Laptop |
| Stencil roofline | `profiling/ncu/roofline_stencil_7000_rtx4060.ncu-rep` | RTX 4060 Laptop |

---

## Conclusions

1. **SpMV is the bottleneck**: 48% of AmgX time, making kernel optimization high-impact

2. **Structure exploitation works**: Eliminating index indirection yields 2× SpMV speedup

3. **Gains compound at scale**: Single-GPU advantage (1.40×) maintained through 8 GPUs (1.44×)

4. **Not a limitation of AmgX**: AmgX correctly handles arbitrary sparse matrices; the performance gap reflects the value of specialization when problem structure is known
