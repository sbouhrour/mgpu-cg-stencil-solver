# Profiling Data

Nsight Systems and Nsight Compute profiles behind the performance analysis.

## Directory Structure

```
profiling/
├── nsys/       # Nsight Systems timeline profiles
├── ncu/        # Nsight Compute kernel analysis (roofline)
└── images/     # Exported screenshots for documentation
```

## Contents

### `nsys/` - Nsight Systems Timelines

| Profile                              | Description                | Hardware |
|--------------------------------------|----------------------------|----------|
| `mpi_1ranks_profile_10000.nsys-rep`  | Custom CG, 1 GPU, 10k×10k  | A100-SXM4-80GB |
| `mpi_2ranks_profile_10000.nsys-rep`  | Custom CG, 2 GPUs, 10k×10k | A100-SXM4-80GB |
| `amgx_1ranks_profile_10000.nsys-rep` | AmgX CG, 1 GPU, 10k×10k    | A100-SXM4-80GB |
| `amgx_2ranks_profile_10000.nsys-rep` | AmgX CG, 2 GPUs, 10k×10k   | A100-SXM4-80GB |

### `ncu/` - Nsight Compute Roofline Analysis

| Profile                                      | Description        | Hardware |
|----------------------------------------------|--------------------|----------|
| `spmv_2d_10000_a100.ncu-rep` | cuSPARSE CSR and stencil SpMV, 10k×10k, roofline set + DRAM bytes | A100-SXM4-80GB |

### `images/` - Figures

| Image | Description |
|-------|-------------|
| `roofline_spmv_comparison.png` | SpMV roofline on A100-SXM4-80GB, measured DRAM bytes (`scripts/plotting/plot_roofline.py`) |

## Viewing Profiles

```bash
# Nsight Systems GUI
nsys-ui profiling/nsys/mpi_2ranks_profile_10000.nsys-rep

# Nsight Compute GUI
ncu-ui profiling/ncu/spmv_2d_10000_a100.ncu-rep
```

## Generating New Profiles

### Nsight Systems (Timeline)

```bash
# Profile custom CG (2 GPUs)
nsys profile --trace=cuda,mpi,nvtx -o profiling/nsys/custom_2gpu \
    mpirun -np 2 ./bin/cg_solver_mgpu_stencil matrix/stencil_10000x10000.mtx

# Profile AmgX (2 GPUs)
nsys profile --trace=cuda,mpi,nvtx -o profiling/nsys/amgx_2gpu \
    mpirun -np 2 ./external/benchmarks/amgx/amgx_cg_solver_mgpu matrix/stencil_10000x10000.mtx
```

### Nsight Compute (Roofline)

```bash
# Both SpMV implementations in one report, clocks left to the GPU
ncu --set roofline --metrics dram__bytes_read.sum,dram__bytes_write.sum --clock-control none \
    -k regex:"csrmv_v3|csr_partition|stencil5_csr_direct" -o profiling/ncu/spmv_2d_10000_a100 \
    ./bin/spmv_bench matrix/stencil_10000x10000.mtx --mode=cusparse-csr,stencil5-csr
```

## Key Observations

- **SpMV dominates**: ~48% of AmgX time, ~41% of custom CG time
- **DRAM bandwidth**: the stencil kernel reaches 83% of peak against 67-71% for cuSPARSE CSR (CUDA 13.0), moving 56 against 83 bytes per row (see roofline)
- **Scaling**: Both implementations show similar parallel efficiency
- **Communication**: MPI staging (D2H → MPI → H2D) visible in custom implementation

See [docs/profiling-2d.md](../docs/profiling-2d.md) for detailed analysis.
