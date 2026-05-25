# Benchmark Scripts Guide

Quatre scripts pour évaluer les performances du solver CG multi-GPU et SpMV.

---

## 1. SpMV Comparison (Single-GPU)

**Script**: `benchmark_spmv_comparison.sh`

**Objectif**: Comparer cuSPARSE CSR vs Stencil CSR sur single-GPU

**Test**:
- Kernels: cuSPARSE CSR, Stencil CSR
- Tailles: 10k, 15k, 20k (100M à 400M unknowns)
- Hardware: 1 GPU

**Usage**:
```bash
./scripts/benchmarking/benchmark_spmv_comparison.sh

# Résultats dans: results_single_gpu_formats_[GPU]_[DATE]/
```

**Résultats attendus (A100)**: Speedup **2.08×** (Stencil vs cuSPARSE, 20k×20k)

---

## 2. Strong Scaling (Multi-GPU CG)

**Script**: `benchmark_problem_sizes.sh`

**Objectif**: Tester strong scaling (même problème, plus de GPUs = plus rapide)

**Test**:
- GPU counts: 1, 2, 4, 8
- Tailles: 10k, 15k, 20k
- Métrique: Speedup et parallel efficiency

**Usage**:
```bash
./scripts/benchmarking/benchmark_problem_sizes.sh

# Résultats dans: results_problem_size_scaling_[GPU]_[DATE]/
```

**Résultats attendus (8× A100)**:
- 20k×20k: **7.48× speedup, 93.5% efficiency**

---

## 3. Weak Scaling (Multi-GPU CG)

**Script**: `benchmark_weak_scaling.sh`

**Objectif**: Tester weak scaling (constant work per GPU, temps constant idéal)

**Test**:
- 1 GPU: 5k×5k (25M unknowns)
- 2 GPUs: 7071×7071 (~50M unknowns)
- 4 GPUs: 10k×10k (100M unknowns)
- 8 GPUs: 14142×14142 (~200M unknowns)

**Usage**:
```bash
./scripts/benchmarking/benchmark_weak_scaling.sh

# Résultats dans: results_weak_scaling_[GPU]_[DATE]/
```

---

## 4. AmgX Comparison

**Script**: `benchmark_amgx.sh`

**Objectif**: Comparer Custom CG vs NVIDIA AmgX

**Prérequis**: AmgX installé (`./scripts/setup/full_setup.sh --amgx`)

**Usage**:
```bash
./scripts/benchmarking/benchmark_amgx.sh

# Résultats dans: results_amgx_comparison_[GPU]_[DATE]/
```

**Résultats attendus**: Custom CG **1.40× faster** (single-GPU, 20k×20k), **1.44× faster** (8 GPUs, 20k×20k)

---

## Workflow Showcase

```bash
# 1. SpMV comparison (pour section hero)
./scripts/benchmarking/benchmark_spmv_comparison.sh

# 2. Strong scaling CG (showcase principal)
./scripts/benchmarking/benchmark_problem_sizes.sh

# 3. AmgX comparison
./scripts/benchmarking/benchmark_amgx.sh
```

---

## Configuration

Tous les scripts utilisent les mêmes conventions :
- **RUNS=10** : Nombre de runs per config (median reporté)
- **Auto-detection** : GPU name, date pour nommage résultats
- **Matrix generation** : Automatique si fichier manquant

---

## Troubleshooting

**Build fails**:
```bash
make cg_solver_mgpu_stencil  # Multi-GPU
make spmv_bench              # Single-GPU
make generate_matrix         # Génération matrices
```

**Out of memory**: Réduire MATRIX_SIZES dans le script

**MPI errors**:
```bash
nvidia-smi --list-gpus       # Vérifier GPU count
mpirun -np 2 hostname        # Test MPI
```
