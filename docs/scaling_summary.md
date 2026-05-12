# Strong Scaling Performance - CG Multi-GPU Solver

**Date**: January 2026  
**Hardware**: NVIDIA A100-SXM4-80GB (8 GPUs)  
**Matrix**: 15000×15000 5-point stencil (225M unknowns, 1.125B nonzeros)  
**Algorithm**: Conjugate Gradient with CSR partitioning + MPI halo exchange  

## Results Summary

| GPUs | Total Time | Time/Iteration | Speedup | Efficiency |
|------|------------|----------------|---------|------------|
| 1    | 299.5 ms   | 21.4 ms       | 1.00×   | 100.0%     |
| 2    | 152.2 ms   | 10.9 ms       | 1.97×   | 98.4%      |
| 4    | 77.7 ms    | 5.5 ms        | 3.85×   | 96.3%      |
| 8    | 40.3 ms    | 2.9 ms        | 7.43×   | **92.9%**  |

**Convergence**: 14 iterations (consistent across all configurations)

## Key Metrics

- **Peak Speedup**: 7.43× on 8 GPUs
- **Scaling Efficiency**: 92.9% at 8 GPUs
- **Per-GPU Performance**: Nearly constant (< 4% degradation)
- **Parallelization Overhead**: ~7% at 8-way scaling (residual from ideal scaling, includes halo exchange, MPI reductions, and synchronization)

## Implementation Highlights

- **Partitioning**: Row-band CSR distribution
- **Communication**: MPI staging (D2H → MPI → H2D) for 160 KB halo per iteration
- **Synchronization**: Explicit barrier-based convergence checks
- **Reproducibility**: Deterministic convergence (14 iterations on all GPU counts)

## Technical Notes

**Strong Scaling**:  Fixed problem size (225M unknowns) distributed across increasing GPU count.

**Efficiency Calculation**: `Speedup / GPUs × 100`

**Baseline**: Single A100 GPU = 299.5 ms (14 iterations)
