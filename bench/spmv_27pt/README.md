# 27-point SpMV against cuSPARSE, on the same CSR arrays

Single-GPU benchmark of `y = A x` for the 3D 27-point operator stored as a standard CSR matrix
(`row_ptr`, `col_idx`, `values`, columns sorted in each row). Every variant reads the same device
arrays: none of them converts, copies or reorders the matrix.

| Variant | What it reads | How |
|---|---|---|
| `cusparse-alg1`, `cusparse-alg2` | `row_ptr`, `col_idx`, `values`, `x` | `cusparseSpMV`, 32-bit indices |
| `rowmajor` | `values`, `x` (`row_ptr` once per row) | kernel of the CG solver, one thread per row |
| `staged` | same bytes as `rowmajor` | each warp copies its 32 rows' values to shared memory with coalesced 16-byte `cp.async` |

Kernel: [`include/spmv_stencil27_fast.cuh`](../../include/spmv_stencil27_fast.cuh). The
`rowmajor` kernel is compiled from the solver's own source file.

## What `staged` changes

With one thread per row, the 32 threads of a warp read their 27 coefficients 216 bytes apart: each
load instruction touches 32 different sectors, and the coefficient stream evicts `x` from the
caches. In `staged`, the 32 rows of a warp occupy one contiguous span of `values`; the warp copies
that span to shared memory with 16-byte `cp.async` (each instruction moves 512 contiguous bytes,
bypassing L1), then each thread reads its own 27 coefficients from shared memory (stride 27,
odd, so free of bank conflicts). The arithmetic and the summation order are those of `rowmajor`,
so the result is bitwise identical; the benchmark counts the differing bits.

Condition, checked once on the device before any timing (`check_pattern_kernel`): every row
holds the stencil pattern (27 entries inside, truncated on faces), the columns of interior rows
being derived from the grid coordinates.

## Run

```bash
make bench_spmv_27pt SPMV27_ARCH=80          # sm_80 or newer (cp.async); SPMV27_ARCH=90 on H100
bin/bench_spmv_27pt --sizes=128,256,384 --coeffs=const
bin/bench_spmv_27pt --sizes=256 --coeffs=var  # variable symmetric coefficients
bench/spmv_27pt/run_session.sh                 # timings, L2 fetch granularity sweep, ncu if present
```

Options: `--reps=N` (default 30), `--l2fetch=32|64|128` (`cudaLimitMaxL2FetchGranularity`),
`--only=a,b`, `--csv=file`.

Each variant is first compared with a reference CSR kernel (one thread per row, entries in CSR
order); a relative error above 1e-12 stops the run before any timing. Timings are medians over
round-robin repetitions, each launch between its own pair of CUDA events. The `model B/r` column
is the per-row traffic model; the bytes actually moved come from `ncu_bytes.sh`.

The two coefficient sets: `const` is the solver's matrix (26 on the diagonal, -1 elsewhere);
`var` draws each coupling from a hash of its row pair, so every row differs, with the diagonal set
to 1 plus the sum of the magnitudes off it. The kernels do not depend on the values.
