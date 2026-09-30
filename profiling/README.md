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

The commands are on the [Reproducing](../docs/reproducing.md#profiling) page.

## Key Observations

- **SpMV dominates**: ~48% of AmgX time, ~41% of custom CG time
- **DRAM bandwidth**: the stencil kernel reaches 83% of peak against 67-71% for cuSPARSE CSR (CUDA 13.0), moving 56 against 83 bytes per row (see roofline)
- **Scaling**: Both implementations show similar parallel efficiency
- **Communication**: MPI staging (D2H → MPI → H2D) visible in custom implementation

See [docs/profiling-2d.md](../docs/profiling-2d.md) for detailed analysis.
