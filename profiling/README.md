# Profiling Data

Nsight Systems and Nsight Compute profiles for performance analysis and showcase.

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
| `mpi_1ranks_profile_10000.nsys-rep`  | Custom CG, 1 GPU, 10k×10k  | A100     |
| `mpi_2ranks_profile_10000.nsys-rep`  | Custom CG, 2 GPUs, 10k×10k | A100     |
| `amgx_1ranks_profile_10000.nsys-rep` | AmgX CG, 1 GPU, 10k×10k    | A100     |
| `amgx_2ranks_profile_10000.nsys-rep` | AmgX CG, 2 GPUs, 10k×10k   | A100     |

### `ncu/` - Nsight Compute Roofline Analysis

| Profile                                      | Description        | Hardware |
|----------------------------------------------|--------------------|----------|
| `roofline_cusparse_csr_7000_rtx4060.ncu-rep` | cuSPARSE CSR SpMV  | RTX 4060 Laptop |
| `roofline_stencil_7000_rtx4060.ncu-rep`      | Stencil SpMV (7k)  | RTX 4060 Laptop |
| `roofline_stencil_5000_rtx4060.ncu-rep`      | Stencil SpMV (5k)  | RTX 4060 Laptop |
| `roofline_stencil_512_rtx4060.ncu-rep`       | Stencil SpMV (512) | RTX 4060 Laptop |

### `images/` - Exported Screenshots

| Image                              | Description                           |
|------------------------------------|---------------------------------------|
| `cusparse_csr_7000_image.png`      | Roofline cuSPARSE CSR (ncu-ui export) |
| `custom_stencil_csr_7000_image.png`| Roofline stencil kernel (ncu-ui export)|

## Viewing Profiles

```bash
# Nsight Systems GUI
nsys-ui profiling/nsys/mpi_2ranks_profile_10000.nsys-rep

# Nsight Compute GUI
ncu-ui profiling/ncu/roofline_stencil_7000_rtx4060.ncu-rep
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
# cuSPARSE CSR roofline
ncu --set roofline -o profiling/ncu/roofline_cusparse \
    ./bin/spmv_bench matrix/stencil_7000x7000.mtx --mode=cusparse-csr

# Stencil kernel roofline
ncu --set roofline -o profiling/ncu/roofline_stencil \
    ./bin/spmv_bench matrix/stencil_7000x7000.mtx --mode=stencil5-csr
```

## Key Observations

- **SpMV dominates**: ~48% of AmgX time, ~41% of custom CG time
- **Memory throughput**: Stencil achieves 95% vs 67% for CSR (see roofline)
- **Scaling**: Both implementations show similar parallel efficiency
- **Communication**: MPI staging (D2H → MPI → H2D) visible in custom implementation

See [docs/profiling-2d.md](../docs/profiling-2d.md) for detailed analysis.
