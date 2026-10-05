# Results

All benchmark results for the multi-GPU CG stencil solver. Measured on 8× NVIDIA A100-SXM4-80GB (NVLink NV12).

For the analysis behind these numbers, see [`profiling-2d.md`](profiling-2d.md) (2D, kernel breakdown, roofline) and [`profiling-3d.md`](profiling-3d.md) (3D, compute-communication overlap). For measurement methodology, see [`methodology.md`](methodology.md).

For the halo exchange backends (host staging, CUDA-aware MPI, NCCL, NVSHMEM), see [`communication.md`](communication.md).

## 2D: Strong Scaling (Custom CG)

**Multi-GPU Strong Scaling** on 8× NVIDIA A100-SXM4-80GB

| Problem Size | 1 GPU | 8 GPUs | Speedup | Efficiency |
|--------------|-------|--------|---------|------------|
| **100M unknowns** (10k×10k stencil) | 133.9 ms | 19.3 ms | 6.94× | 86.8% |
| **225M unknowns** (15k×15k stencil) | 300.1 ms | 40.4 ms | 7.43× | 92.9% |
| **400M unknowns** (20k×20k stencil) | 531.4 ms | 71.0 ms | **7.48×** | **93.5%** |

## 2D: Detailed Scaling (1 / 2 / 4 / 8 GPUs)

### 10000×10000 stencil (100M unknowns)

| GPUs | Time (ms) | Speedup | Efficiency |
|------|-----------|---------|------------|
| 1    | 133.9     | 1.00×   | 100.0%     |
| 2    | 68.7      | 1.95×   | 97.5%      |
| 4    | 35.7      | 3.76×   | 93.9%      |
| 8    | 19.3      | **6.94×** | **86.8%**  |

### 15000×15000 stencil (225M unknowns)

| GPUs | Time (ms) | Speedup | Efficiency |
|------|-----------|---------|------------|
| 1    | 300.1     | 1.00×   | 100.0%     |
| 2    | 152.5     | 1.97×   | 98.4%      |
| 4    | 77.7      | 3.86×   | 96.5%      |
| 8    | 40.4      | **7.43×** | **92.9%**  |

### 20000×20000 stencil (400M unknowns)

| GPUs | Time (ms) | Speedup | Efficiency |
|------|-----------|---------|------------|
| 1    | 531.4     | 1.00×   | 100.0%     |
| 2    | 269.3     | 1.97×   | 98.7%      |
| 4    | 136.3     | 3.90×   | 97.5%      |
| 8    | 71.0      | **7.48×** | **93.5%**  |

<sub>Convergence: 14 iterations in every configuration. The 2D matrix is well conditioned, so the iteration count
does not grow with the grid; the 3D matrices are plain Laplacians, and their iteration counts grow with the grid.</sub>

## 2D: SpMV Format Comparison

**Format comparison** on NVIDIA A100-SXM4-80GB

| Matrix Size | cuSPARSE | CSR (cuSPARSE) | STENCIL5 (Custom) | Speedup |
|-------------|----------|---------------:|------------------:|--------:|
| **10k×10k** (100M unknowns) | CUDA 12.8 | 6.77 ms | 3.25 ms | **2.08×** |
| **15k×15k** (225M unknowns) | CUDA 12.8 | 15.00 ms | 7.29 ms | **2.06×** |
| **20k×20k** (400M unknowns) | CUDA 12.8 | 26.77 ms | 12.86 ms | **2.08×** |
| **10k×10k** (100M unknowns) | CUDA 13.0 | 6.10 ms | 3.31 ms | **1.84×** |

<sub>Median of 10 runs. The CUDA 13.0 row comes from a later session on the same GPU model. DRAM traffic measured with Nsight Compute: 56.0 B/row for the stencil kernel
(1.69-1.74 TB/s, 83-85% of peak), 83.3-83.8 B/row for cuSPARSE. Analysis in
[`profiling-2d.md`](profiling-2d.md#2-spmv-kernel-analysis).</sub>

## 2D: Custom CG vs NVIDIA AmgX

Both solvers run unpreconditioned CG (iso-algorithm): the speedups reflect implementation efficiency on the same algorithm, not an algorithmic difference.

**Hardware**: 8× NVIDIA A100-SXM4-80GB · CUDA 12.8 · Driver 575.57 (same configuration for both solvers)

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

## 3D: 7-Point Stencil (Sync vs Overlap)

**Hardware**: 8× NVIDIA A100-SXM4-80GB

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

## 3D: 27-Point Stencil (Sync vs Overlap)

| Grid | GPUs | Sync (ms) | Overlap (ms) | Overlap Gain | Iterations |
|------|------|-----------|--------------|--------------|------------|
| 128³ | 1 | 89.2 | 89.6 | — | 151 |
| 128³ | 2 | 57.3 | 51.1 | 1.12× | 151 |
| 128³ | 4 | 47.3 | 36.6 | 1.29× | 151 |
| 128³ | 8 | 40.5 | 33.6 | 1.21× | 151 |
| 256³ | 1 | 1315.4 | 1315.4 | — | 303 |
| 256³ | 2 | 718.9 | 680.3 | 1.06× | 303 |
| 256³ | 4 | 447.5 | 367.5 | 1.22× | 303 |
| 256³ | 8 | 294.0 | 203.5 | 1.45× | 303 |
| 512³ | 1 | 22016 | 21997 | — | 611 |
| 512³ | 2 | 11438 | 11142 | 1.03× | 611 |
| 512³ | 4 | 6461 | 5815 | 1.11× | 611 |
| 512³ | 8 | 3809 | 3110 | 1.23× | 611 |

## 3D: Strong Scaling Efficiency (overlap solver)

**7-point stencil**: speedup relative to 1-GPU sync baseline:

| Grid | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|------|-------|--------|--------|--------|
| 128³ | 1.00× | 1.69× | 1.59× | 1.49× |
| 256³ | 1.00× | 1.88× | 3.06× | 3.66× |
| 512³ | 1.00× | 1.97× | 3.84× | 6.17× |

<sub>512³ at 8 GPUs: 6.17× speedup (77% parallel efficiency).</sub>

**27-point stencil**: speedup relative to 1-GPU sync baseline:

| Grid | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
|------|-------|--------|--------|--------|
| 128³ | 1.00× | 1.75× | 2.44× | 2.66× |
| 256³ | 1.00× | 1.93× | 3.58× | 6.47× |
| 512³ | 1.00× | 1.98× | 3.79× | 7.08× |

<sub>512³ at 8 GPUs: 7.08× speedup (**88% parallel efficiency**).</sub>

## 3D: Custom CG vs NVIDIA AmgX (27-point)

At equal transport, the Custom CG synchronous solver solves the 3D 27-point system 1.09× to 1.31× faster than AmgX. The gap narrows as communication takes a larger share of the time: at 128³, 1.31× on 1 GPU and 1.09× on 8 GPUs. On 1 GPU, with no halo exchange, the gap is the computation alone (SpMV and vector operations); on several GPUs, both solvers send their halos through host memory. With compute-communication overlap, still through host memory, the Custom CG is 1.19× to 1.78× faster than AmgX. With NCCL, device dot products and a CUDA graph, a different transport from AmgX's, it is 1.27× to 1.69× faster. The fastest Custom configuration depends on the size: NCCL at 128³ on 8 GPUs (1.48×), overlap at 256³ on 8 GPUs (1.78×), the two within 0.3% at 512³ on 8 GPUs (1.37×).

All runs solve unpreconditioned CG to the same relative residual (1e-6, L2 norm, from x0 = 0 with b = 1). The synchronous and overlap solvers and AmgX take the same number of iterations: 151 at 128³, 303 at 256³, 611 at 512³. The NCCL configuration tests convergence every 10 iterations (`--check-every=10`), so it stops after 160, 310 and 620; its time includes them. AmgX uses its `MPI` communicator (Open MPI 4.1.6 without CUDA support); its `MPI_DIRECT` communicator is slower on this node (see [Communication Backends](communication.md)). Time to solution, median of 10 solves.

**Hardware**: 8× NVIDIA A100-SXM4-80GB (NVLink NV12) · CUDA 12.8 · Driver 580.65.06 · AmgX v2.5.0 · ranks bound to the CPU cores local to their GPU · measured on 5 October 2026 at commit `ba170c8` (PR #23)

| Grid | GPUs | NVIDIA AmgX | Custom, synchronous | Custom, overlap | Custom, NCCL + CUDA graph |
|------|-----:|------------:|--------------------:|----------------:|--------------------------:|
| 128³ (2.1M unknowns) | 1 | 121.5 ms | 92.5 ms (1.31×) | | |
| | 2 | 83.3 ms | 65.2 ms (1.28×) | 59.6 ms (1.40×) | 59.7 ms (1.40×) |
| | 4 | 61.8 ms | 54.2 ms (1.14×) | 43.5 ms (1.42×) | 43.7 ms (1.41×) |
| | 8 | 53.3 ms | 48.7 ms (1.09×) | 44.7 ms (1.19×) | 36.0 ms (1.48×) |
| 256³ (16.8M unknowns) | 1 | 1716.1 ms | 1330.6 ms (1.29×) | | |
| | 2 | 968.5 ms | 755.6 ms (1.28×) | 708.3 ms (1.37×) | 718.1 ms (1.35×) |
| | 4 | 566.4 ms | 458.7 ms (1.23×) | 381.3 ms (1.49×) | 391.1 ms (1.45×) |
| | 8 | 382.0 ms | 324.8 ms (1.18×) | 214.6 ms (1.78×) | 226.2 ms (1.69×) |
| 512³ (134M unknowns) | 1 | n/a | 21987.4 ms | | |
| | 2 | n/a | 11757.4 ms | 11326.3 ms | 11513.9 ms |
| | 4 | 7639.5 ms | 6631.8 ms (1.15×) | 5947.5 ms (1.28×) | 5994.5 ms (1.27×) |
| | 8 | 4412.9 ms | 3985.1 ms (1.11×) | 3228.3 ms (1.37×) | 3218.6 ms (1.37×) |

<sub>AmgX / Custom in brackets. Overlap and NCCL run on 2 GPUs or more. 512³ on 1 rank: AmgX's distributed matrix indexes local entries with 32-bit integers, and the rank holds 3.6 × 10⁹ entries. On 2 ranks (1.8 × 10⁹ local entries), the AmgX matrix upload stopped with "CUDA kernel launch error". Raw data: [`data/amgx_3d_a100/`](https://github.com/sbouhrour/mgpu-cg-stencil-solver/tree/main/docs/data/amgx_3d_a100). Commands: [Reproducing](reproducing.md).</sub>

## 3D: 27-Point SpMV vs cuSPARSE CSR (single GPU)

The same CSR arrays for every variant (`bench/spmv_27pt/`): cuSPARSE ALG1 (32-bit indices, the faster of ALG1 and ALG2), the row-major kernel of the 3D solver (one thread per row), and `staged`, the same kernel with the values of each warp's 32 rows copied to shared memory by coalesced 16-byte `cp.async` before use. `staged` and the row-major kernel produce bitwise identical results. Kernel time, median of 30 launches, constant coefficients; variable symmetric coefficients give the same times within 0.2%.

**Hardware**: NVIDIA A100-SXM4-80GB · Driver 580.65.06 · default L2 fetch granularity (64 bytes) · measured on 5 October 2026, benchmark and kernel of PR #19

| Grid | CUDA (cuSPARSE) | cuSPARSE ALG1 | Row-major | Staged |
|------|-----------------|--------------:|----------:|-------:|
| 256³ | 12.8 (12.5.8) | 4.530 ms | 3.348 ms (1.35×) | 2.789 ms (1.62×) |
| 384³ | 12.8 (12.5.8) | 15.064 ms | 11.797 ms (1.28×) | 9.147 ms (1.65×) |
| 256³ | 13.0 (12.6.3) | 4.339 ms | 3.347 ms (1.30×) | 2.786 ms (1.56×) |
| 384³ | 13.0 (12.6.3) | 14.574 ms | 11.797 ms (1.24×) | 9.153 ms (1.59×) |

<sub>Speedups against cuSPARSE ALG1 of the same CUDA version. Raw output: `bench/spmv_27pt/run_session.sh` writes one text file per coefficient set and a CSV.</sub>

