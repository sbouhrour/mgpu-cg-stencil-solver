# Results

All benchmark results for the multi-GPU CG stencil solver. Measured on 8× NVIDIA A100-SXM4-80GB (NVLink NV12), median of 10 runs with 3 warmup runs discarded.

For the analysis behind these numbers, see [`profiling-2d.md`](profiling-2d.md) (2D, kernel breakdown, roofline) and [`profiling-3d.md`](profiling-3d.md) (3D, compute-communication overlap). For measurement methodology, see [`methodology.md`](methodology.md).

## 2D — Strong Scaling (Custom CG)

**Multi-GPU Strong Scaling** on 8× NVIDIA A100-SXM4-80GB

| Problem Size | 1 GPU | 8 GPUs | Speedup | Efficiency |
|--------------|-------|--------|---------|------------|
| **100M unknowns** (10k×10k stencil) | 133.9 ms | 19.3 ms | 6.94× | 86.8% |
| **225M unknowns** (15k×15k stencil) | 300.1 ms | 40.4 ms | 7.43× | 92.9% |
| **400M unknowns** (20k×20k stencil) | 531.4 ms | 71.0 ms | **7.48×** | **93.5%** |

<sub>*Median of 10 runs; 3 warmup runs discarded.*</sub>

## 2D — Detailed Scaling (1 / 2 / 4 / 8 GPUs)

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

<sub>Convergence: 14 iterations across all configurations. Source: `results_archive/results_problem_size_scaling_NVIDIAA100-SXM4-80GB_20260109_123920/`.</sub>

## 2D — SpMV Format Comparison

**Format Comparison** on NVIDIA A100 80GB PCIe

| Matrix Size | CSR (cuSPARSE) | STENCIL5 (Custom) | Speedup | Bandwidth Improvement |
|-------------|----------------|-------------------|---------|----------------------|
| **10k×10k** (100M unknowns) | 6.77 ms | 3.25 ms | **2.08×** | 1.98× (1182 → 2339 GB/s) |
| **15k×15k** (225M unknowns) | 15.00 ms | 7.29 ms | **2.06×** | 1.96× (1200 → 2346 GB/s) |
| **20k×20k** (400M unknowns) | 26.77 ms | 12.86 ms | **2.08×** | 1.98× (1195 → 2364 GB/s) |

<sub>*Median of 10 runs; 3 warmup runs discarded.*</sub>

## 2D — Custom CG vs NVIDIA AmgX

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

## 3D — 7-Point Stencil (Sync vs Overlap)

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

## 3D — 27-Point Stencil (Sync vs Overlap)

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

## 3D — Strong Scaling Efficiency (overlap solver)

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
