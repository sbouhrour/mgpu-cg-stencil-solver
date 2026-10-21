# `--spmv=staged` in the 3D solver, 8x A100-SXM4-80GB

Measured on 6 October 2026 at commit `cd3486c` (PR #26, branch `wip/solver-27pt-staged`), solver built with `make ARCH=80`. CUDA 12.8, driver 580.65.06.

| File | Content |
|------|---------|
| `verify_reproduction.txt` | `scripts/verify_reproduction.sh`: 21 passed, including `--spmv=staged` against `rowmajor` on 1 rank and on 2 ranks with overlap |
| `bench_spmv_27pt.txt`, `.csv` | `bin/bench_spmv_27pt --sizes=128,256,384 --coeffs=const`; the `staged-solver` line is the kernel of `--spmv=staged`, compiled from the solver's source |
| `nsys_256_np8_sync_summary.txt` | Nsight Systems, 256³ on 8 GPUs, synchronous solver, rank 0: mean kernel durations per iteration over 7 solves of 20 iterations, `rowmajor` then `staged` (`scripts/benchmarking/comm_nsys_summary.py --iters 20`) |
| `time_to_solution/` | one JSON per solve configuration (median of 10 solves, iterations, `solution_sum`), `rowmajor` and `staged`; `r2_*`: second round at 256³ |
