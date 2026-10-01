# 27-point SpMV against cuSPARSE, on the same CSR arrays

Single-GPU benchmark of `y = A x` for the 3D 27-point operator stored as a standard CSR matrix
(`row_ptr`, `col_idx`, `values`, columns sorted in each row). Every variant reads the same device
arrays: none of them converts, copies or reorders the matrix.

| Variant | What it reads | How |
|---|---|---|
| `cusparse-alg1`, `cusparse-alg2` | `row_ptr`, `col_idx`, `values`, `x` | `cusparseSpMV`, 32-bit indices |
| `rowmajor` | `values`, `x` (`row_ptr` once per row) | kernel of the CG solver, one thread per row |
| `staged` | same bytes as `rowmajor` | each warp copies its 32 rows' values to shared memory with coalesced 16-byte `cp.async` |
| `sym-tj4/8/16` | entries 13..26 of each interior row, `x` | symmetric half-read, 2.5D tile marching along z |

Kernels: [`include/spmv_stencil27_fast.cuh`](../../include/spmv_stencil27_fast.cuh). The
`rowmajor` kernel is compiled from the solver's own source file.

## Why reading half of `values` is exact

In an interior row the 27 entries are sorted by column, so entry `p` couples the row to the
neighbour at offset `(p/9-1, (p/3)%3-1, p%3-1)` and entry `26-p` to the opposite one. For a
symmetric matrix the 13 entries before the diagonal duplicate couplings stored after the diagonal
of another row: `A(r, r-d) = A(r-d, r)`. The `sym` kernel reads entries 13..26 (the diagonal and
the 13 positive offsets, one contiguous run of 112 bytes per row) and obtains each negative-offset
term from the thread that owns the stored copy, which multiplies it by its own `x` and passes the
product on: warp shuffle along `k`, shared memory along `j`, a register carried to the next plane
along `i`. Interior rows adjacent to a face read the few entries whose partner is a truncated face
row from their own row. Face rows run a plain CSR loop.

Conditions, checked once on the device before any timing (`check_pattern_sym_kernel`): interior
rows hold the 27-entry pattern, and the two stored copies of every coupling between interior rows
are bitwise equal. CG requires a symmetric matrix in any case.

## Run

```bash
make bench_spmv_27pt SPMV27_ARCH=80          # sm_80 or newer (cp.async)
bin/bench_spmv_27pt --sizes=128,256,384 --coeffs=const
bin/bench_spmv_27pt --sizes=256 --coeffs=var  # variable symmetric coefficients
bench/spmv_27pt/run_session.sh                 # full session: timings, L2 fetch sweep, ncu
```

Options: `--reps=N` (default 30), `--zc=N` (planes per block of the `sym` kernel, default 32),
`--l2fetch=32|64|128` (`cudaLimitMaxL2FetchGranularity`), `--only=a,b`, `--csv=file`.

Each variant is first compared with a reference CSR kernel (one thread per row, entries in CSR
order); a relative error above 1e-12 stops the run before any timing. Timings are medians over
round-robin repetitions, each launch between its own pair of CUDA events. The `model B/r` column
is the per-row traffic model (`sym`: 136 B of values per interior row at 32-byte DRAM sectors);
the bytes actually moved come from `ncu_bytes.sh`.

The two coefficient sets: `const` is the solver's matrix (26 on the diagonal, -1 elsewhere);
`var` draws each coupling from a hash of its row pair, so the matrix stays exactly symmetric while
every row differs, with the diagonal set to 1 plus the sum of the magnitudes off it.
