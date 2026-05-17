# Reference Implementations

Standalone benchmarks comparing the custom SpMV and CG kernels against production-grade libraries.

## AmgX (NVIDIA Production Solver)

NVIDIA's production-grade linear algebra library, used as the reference for the Custom CG comparison.

See [`benchmarks/amgx/README.md`](benchmarks/amgx/README.md) for build instructions, usage, and implementation notes.

## Build Configuration

The build flags are **not** unified across the comparison:

- **Custom SpMV / CG kernels**: `-O2 --ptxas-options=-O2 -std=c++11` (root `Makefile`, release build)
- **AmgX benchmark driver**: `-O3 --ptxas-options=-O3 -std=c++17` (`benchmarks/amgx/Makefile`)
- **AmgX library (`libamgx`)**: built by AmgX's own CMake build system in `Release` mode (`-O3 -DNDEBUG`); not controlled by this repository

So the entire AmgX side (driver and library) is compiled at `-O3`, while the custom kernels — where the solver work being measured actually runs — are compiled at `-O2`. The custom solver is therefore the **less-optimized** side of the comparison: the reported speedups are conservative, not flattering to the custom implementation. Test methodology is otherwise consistent (identical matrices, same run protocol, median of 10 runs).
