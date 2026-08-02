# Development

This guide is for contributors extending the solver: build system, adding new kernels or solvers, and running the test suite.

### Build System

**Dual build approach** for flexibility:
- **Makefile**: Primary build for CUDA/MPI binaries
- **CMake**: Testing framework with Google Test

```bash
# Release build (default)
make

# Debug build with GPU debugging (-g -G)
make BUILD_TYPE=debug

# Build specific targets
make cg_solver_mgpu_stencil
make generate_matrix

# Run tests
cd tests && mkdir build && cd build
cmake .. && make && ./test_runner
```

### Adding Features

1. **New SpMV kernel**: Implement in `src/spmv/`, register in `get_operator()`
2. **New solver**: Add to `src/solvers/`, create entry point in `src/main/`
3. **Performance metrics**: Extend `benchmark_stats_mgpu_partitioned.cu`

### Testing

```bash
# All tests
./test_runner

# Specific test suite
./test_runner --gtest_filter="PartitionedSolver*"
```

## Approaches tried and set aside

`main` carries the retained implementation only. Several alternatives were built, measured, and dropped;
they live on their branches, which is where to look before concluding a technique was never tried.

### GPU-to-GPU communication

Three NCCL designs preceded the current MPI staging:

| Branch | What it contains |
|---|---|
| `feature/multi-gpu-cg-nccl` | NCCL `AllGather` with full vector replication (~800 MB per iteration at 10000×10000), then batched `ncclGroupStart/End` |
| `feature/csr-partition` | local CSR partitions, ending the full replication |
| `feature/halo-exchange` | NCCL P2P `ncclSend/Recv` on the boundary only — 160 KB instead of 800 MB |
| `feature/halo-cuda-aware-mpi` | the switch to MPI staging that produced the current solver |
| `archive/experimental` | the archived NCCL AllGather solver |

The decisive comparison was NCCL P2P against MPI staging at an identical 160 KB halo, on 2× RTX 3090
(PCIe, no NVLink): 56.78 ms of halo time for NCCL against 31.93 ms for MPI staging. On NVLink hardware
the ranking may well invert — the result is specific to small, repeated messages over PCIe.

### Other branches

Communication and halo variants (`feature/cuda-aware-mpi`, `feature/halo-cudamemcpypeer`,
`feature/overlap-cudaipc`, `feature/overlap-streams`), kernel work (`feature/kernel-opt-3d`,
`feature/stencil-27pt`, `feature/spmv-27pt-soa`), AmgX integration (`feature/amgx-*`), and profiling-only
explorations (`exploration/*`).
