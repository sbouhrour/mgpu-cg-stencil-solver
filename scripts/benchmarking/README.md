# Benchmark Scripts Guide

Four scripts to evaluate the performance of the multi-GPU CG solver and SpMV.

---

## 1. SpMV Comparison (Single-GPU)

**Script**: `benchmark_spmv_comparison.sh`

**Goal**: Compare cuSPARSE CSR vs Stencil CSR on a single GPU

**Test**:
- Kernels: cuSPARSE CSR, Stencil CSR
- Sizes: 10k, 15k, 20k (100M to 400M unknowns)
- Hardware: 1 GPU

**Usage**:
```bash
./scripts/benchmarking/benchmark_spmv_comparison.sh

# Results in: results_single_gpu_formats_[GPU]_[DATE]/
```

**Expected results (A100)**: **2.08×** speedup (Stencil vs cuSPARSE, 20k×20k)

---

## 2. Strong Scaling (Multi-GPU CG)

**Script**: `benchmark_problem_sizes.sh`

**Goal**: Test strong scaling (same problem, more GPUs = faster)

**Test**:
- GPU counts: 1, 2, 4, 8
- Sizes: 10k, 15k, 20k
- Metric: Speedup and parallel efficiency

**Usage**:
```bash
./scripts/benchmarking/benchmark_problem_sizes.sh

# Results in: results_problem_size_scaling_[GPU]_[DATE]/
```

**Expected results (8× A100)**:
- 20k×20k: **7.48× speedup, 93.5% efficiency**

---

## 3. Weak Scaling (Multi-GPU CG)

**Script**: `benchmark_weak_scaling.sh`

**Goal**: Test weak scaling (constant work per GPU, ideally constant time)

**Test**:
- 1 GPU: 5k×5k (25M unknowns)
- 2 GPUs: 7071×7071 (~50M unknowns)
- 4 GPUs: 10k×10k (100M unknowns)
- 8 GPUs: 14142×14142 (~200M unknowns)

**Usage**:
```bash
./scripts/benchmarking/benchmark_weak_scaling.sh

# Results in: results_weak_scaling_[GPU]_[DATE]/
```

---

## 4. AmgX Comparison

**Script**: `benchmark_amgx.sh`

**Goal**: Compare Custom CG vs NVIDIA AmgX

**Prerequisites**: AmgX installed (`./scripts/setup/full_setup.sh --amgx`)

**Usage**:
```bash
./scripts/benchmarking/benchmark_amgx.sh

# Results in: results_amgx_comparison_[GPU]_[DATE]/
```

**Expected results**: Custom CG **1.40× faster** (single-GPU, 20k×20k), **1.44× faster** (8 GPUs, 20k×20k)

---

## Showcase Workflow

```bash
# 1. SpMV comparison (for the hero section)
./scripts/benchmarking/benchmark_spmv_comparison.sh

# 2. Strong scaling CG (main showcase)
./scripts/benchmarking/benchmark_problem_sizes.sh

# 3. AmgX comparison
./scripts/benchmarking/benchmark_amgx.sh
```

---

## Configuration

All scripts share the same conventions:
- **RUNS=10**: Number of runs per config (median reported)
- **Auto-detection**: GPU name and date for result naming
- **Matrix generation**: Automatic if the file is missing

---

## Troubleshooting

**Build fails**:
```bash
make cg_solver_mgpu_stencil  # Multi-GPU
make spmv_bench              # Single-GPU
make generate_matrix         # Matrix generation
```

**Out of memory**: Reduce MATRIX_SIZES in the script

**MPI errors**:
```bash
nvidia-smi --list-gpus       # Check GPU count
mpirun -np 2 hostname        # Test MPI
```
