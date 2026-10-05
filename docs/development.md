# Development

This guide is for contributors extending the solver: build system, adding new kernels or solvers, and checking a change.

### Build System

The Makefile builds every binary (CUDA, and MPI when `mpic++` is in the `PATH`):

```bash
# Release build (default)
make

# Debug build with GPU debugging (-g -G)
make BUILD_TYPE=debug

# Build specific targets
make cg_solver_mgpu_stencil
make generate_matrix
```

### Adding Features

1. **New SpMV kernel**: Implement in `src/spmv/`, register in `get_operator()`
2. **New solver**: Add to `src/solvers/`, create entry point in `src/main/`
3. **Performance metrics**: Extend `benchmark_stats_mgpu_partitioned.cu`

### Checking a change

```bash
make
./scripts/verify_reproduction.sh   # on a GPU node; AmgX checks run when AmgX is built
```

`verify_reproduction.sh` compares counts, not times: iteration counts of the published cases, the
in-memory operators against the Matrix Market files (`Sum(x)`), 1 rank against 2, and AmgX against the
Custom CG. Its exit status is 0 only when every check passes. The CI
(`.github/workflows/ci.yml`) has no GPU: it checks formatting, script syntax, the documentation build,
and that every binary compiles with CUDA 11.8, 12.8 and 13.0.
