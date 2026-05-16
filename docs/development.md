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
