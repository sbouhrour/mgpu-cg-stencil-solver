# Scripts

## Quick Start

```bash
# Full setup (auto-detects GPU, installs dependencies)
./scripts/setup/full_setup.sh

# With AmgX for comparison benchmarks
./scripts/setup/full_setup.sh --amgx
```

## Run All Benchmarks

```bash
# Full benchmarks (1000×1000 matrix, 10 runs)
./scripts/run_all.sh

# Quick verification (~2 min, 512×512)
./scripts/run_all.sh --quick

# Custom matrix size
./scripts/run_all.sh --size=10000
```

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| `setup/` | Installation scripts (dependencies, AmgX) |
| `benchmarking/` | Individual benchmark scripts ([README](benchmarking/README.md)) |
| `plotting/` | Result visualization (matplotlib) |
| `profiling/` | Nsight Systems/Compute profiling |
| `visualizations/` | Figure generation for docs |

## Individual Benchmarks

See [benchmarking/README.md](benchmarking/README.md) for detailed documentation on:
- `benchmark_spmv_comparison.sh` - cuSPARSE CSR vs Stencil CSR
- `benchmark_problem_sizes.sh` - Strong scaling (1→8 GPUs)
- `benchmark_weak_scaling.sh` - Weak scaling
- `benchmark_amgx.sh` - AmgX comparison
