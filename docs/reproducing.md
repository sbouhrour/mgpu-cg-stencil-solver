# Reproducing the Results

This document explains how to build the solver, run the benchmarks, and reproduce the results on your own hardware.

For methodology details (statistical approach, timing scope, profiling tools), see [`methodology.md`](methodology.md).

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

## Build and Run

### One Command

```bash
./scripts/run_all.sh
```

This builds all components, runs benchmarks, and saves results to:
- `results/raw/` — Raw TXT outputs
- `results/json/` — Structured JSON data
- `results/figures/` — Generated plots (after running plotting script)

For the AmgX comparison build (optional, ~15 min extra build time):

```bash
./scripts/setup/full_setup.sh --amgx     # Setup with AmgX support
./scripts/run_all.sh                     # Auto-detects AmgX if installed
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
