# AmgX Benchmarks

NVIDIA AmgX benchmarks for sparse linear solvers (CG, SpMV) on single and
multi-GPU. This harness exists to provide an external solver baseline for the
project's custom CG: AmgX runs the **same** matrix with the **same algorithm**
(unpreconditioned CG), so the timing difference reflects implementation, not a
different numerical method.

This document is an internal technical reference for the harness itself: what
each binary does, how AmgX is configured, the input format, how to run it, and
the known limits. It is not showcase material.

---

## 1. Overview

AmgX is NVIDIA's open-source GPU library for sparse iterative linear solvers
(algebraic multigrid and Krylov methods, with preconditioners). Here it is used
only as a **reference implementation of conjugate gradient**, not for its
multigrid capabilities.

**Why AmgX is the chosen baseline.** The project's contribution is a custom
multi-GPU CG built on stencil SpMV kernels. To make a fair solver-vs-solver
comparison, the baseline must run the *same* algorithm on the *same* matrix:
unpreconditioned CG, identical RHS and initial guess, identical convergence
criterion. AmgX provides a production-grade CG that satisfies this, so a
wall-clock comparison at equal iteration count is meaningful.

**What this integration does NOT do:**

- **No pure-kernel comparison.** The CG binaries time the whole solve, not an
  isolated SpMV. (A separate `amgx_spmv_bench` micro-benchmark times SpMV alone,
  but it is not part of the CG comparison.)
- **No multigrid.** AmgX is configured as plain CG, not AMG. Comparing against
  AMG would compare different algorithms and would not be interpretable as a
  same-method speedup.
- **No preconditioning.** The custom CG has no preconditioner, so AmgX is run
  unpreconditioned to match (see §3, §8).
- **No compute/communication overlap inside AmgX.** AmgX handles its own halo
  exchange; there is no AmgX equivalent of the custom solver's `--overlap`
  mode. The two are compared as-is.

---

## 2. Harness architecture

Three standalone binaries, each built from a single source file:

| Binary               | Source                    | Role                                            | MPI |
|----------------------|---------------------------|-------------------------------------------------|-----|
| `amgx_cg_solver`     | `amgx_cg_solver.cpp`      | Single-GPU CG                                    | No  |
| `amgx_cg_solver_mgpu`| `amgx_cg_solver_mgpu.cpp` | Multi-GPU CG (one GPU per rank)                  | Yes |
| `amgx_spmv_bench`    | `amgx_spmv_stencil.cpp`   | Single-GPU SpMV micro-benchmark (reference only) | No  |

Shared helper: `amgx_benchmark.h` — JSON/CSV **result** export and the
`BenchmarkResults` / `MatrixInfo` structs (output formatting only; not solver
configuration).

### Dependencies

- **CUDA** (tested with 12.4+; the local source build reports CUDA Runtime 12.9).
- **MPI** (OpenMPI or MPICH) — only for `amgx_cg_solver_mgpu`.
- **AmgX library** — linked as `-lamgx` (the `libamgx.a` static archive in
  practice; the produced binaries carry AmgX statically, no `libamgx*.so` in
  `ldd`).
- AmgX version observed at build: **2.5.0** (printed at runtime).

### How AmgX is located (Makefile auto-detection)

The Makefile picks the first of these that contains `include/amgx_c.h`:

1. `../../amgx-src` (a source build; uses `../../amgx-src/build` as libdir)
2. `../../amgx` (an install tree; uses `../../amgx/lib`)
3. `/usr/local` (system install)

Check what was detected:

```bash
make help
```

> **To clarify / watch out:** the auto-detected libdir for the source-build case
> is `../../amgx-src/build`, but the static archive may actually live elsewhere
> (e.g. `/usr/local/lib/libamgx.a`). On the machine used here the link still
> succeeds because the linker finds `libamgx.a` on its default search path. If a
> rebuild fails to find `-lamgx`, add `-L/usr/local/lib` (or wherever
> `libamgx.a` is) to `LDFLAGS`.

### Building

```bash
make                       # all three
make amgx_cg_solver        # single-GPU CG
make amgx_cg_solver_mgpu   # multi-GPU MPI CG
make amgx_spmv_bench       # SpMV micro-benchmark
```

---

## 3. AmgX configuration

**There is no JSON configuration file in this harness.** AmgX accepts its
configuration as a string, and each binary builds that string inline and passes
it to `AMGX_config_create`. (AmgX ships many example `.json` configs under its
own `build/configs/`, but none are used here. The `--json=` CLI flag is for
*result export*, not input config.)

### CG configuration

**`amgx_cg_solver_mgpu` (multi-GPU)** — `amgx_cg_solver_mgpu.cpp`:

```
config_version=2,
solver=CG,
max_iters=<N>,
convergence=RELATIVE_INI,
tolerance=<tol>,
norm=L2,
print_solve_stats=0,
monitor_residual=1,
obtain_timings=0
```

**`amgx_cg_solver` (single-GPU)** — `amgx_cg_solver.cpp`:

```
config_version=2,
solver=PCG,
preconditioner=NOSOLVER,
max_iters=<N>,
convergence=RELATIVE_INI,
tolerance=<tol>,
norm=L2,
print_solve_stats=0,
monitor_residual=1,
obtain_timings=0
```

Parameter meaning:

- `solver=CG` / `solver=PCG, preconditioner=NOSOLVER` — conjugate gradient.
  `PCG` with `preconditioner=NOSOLVER` is preconditioned CG with the identity
  preconditioner, i.e. mathematically the same as plain CG. The two binaries use
  different spellings of the same unpreconditioned method.
- `convergence=RELATIVE_INI` — stop when the residual norm has dropped by a
  factor `tolerance` **relative to the initial residual**.
- `tolerance` — the relative threshold (default `1e-6`, overridable via `--tol`).
- `norm=L2` — convergence measured in the L2 norm.
- `monitor_residual=1` — required so AmgX computes the residual each iteration
  (without it the convergence check is unavailable).
- `print_solve_stats=0`, `obtain_timings=0` — AmgX's own logging/timing off;
  the harness does its own timing.

### Why this is "unpreconditioned CG" and why it matches the custom solver

The custom CG carries no preconditioner. With `x0 = 0` the initial residual is
`r0 = b - A·x0 = b`, so `RELATIVE_INI` reduction by `tolerance` is exactly the
custom solver's stop test `||r|| / ||b|| < tol` in the L2 norm. Same algorithm,
same RHS (`b = ones`), same initial guess (`x0 = 0`), same stop criterion. See
§7 and §8.

### SpMV configuration

`amgx_spmv_stencil.cpp` uses `config_version=2, determinism_flag=1` and calls
`AMGX_matrix_vector_multiply` directly — no solver, no convergence. It is a
standalone SpMV timing, not part of the CG comparison.

---

## 4. Input format and matrix loading

### `.mtx` format

Standard Matrix Market coordinate format, 1-based indices:

```
%%MatrixMarket matrix coordinate real general
% STENCIL_GRID_SIZE 192          <- optional comment, see below
7077888 7077888 189119224        <- rows cols nnz
<row> <col> <value>              <- nnz lines, 1-based
...
```

The `% STENCIL_GRID_SIZE N` comment is emitted by the project's
`generate_matrix_3d_27pt` / stencil generators. The AmgX harness reads it only
to print a grid label (`NxNxN` for a 3D stencil where `rows == N^3`, otherwise
the 2D `sqrt(rows)` label); it does not affect the solve. All `%` comment lines
are otherwise skipped.

### Loading

Every rank loads the **entire** matrix file independently:

1. `read_matrix_market` reads all `nnz` triples into COO arrays, then converts
   to CSR (`row_ptr`, `col_idx`, `values`).
2. **Single-GPU** (`amgx_cg_solver`): uploads the full CSR with
   `AMGX_matrix_upload_all` (local `int` column indices).
3. **Multi-GPU** (`amgx_cg_solver_mgpu`): computes a 1D row-band partition
   (`rows / world_size`, last rank absorbs the remainder), extracts this rank's
   rows into a local CSR with **global** `int64_t` column indices, and uploads
   with `AMGX_matrix_upload_all_global`. AmgX then auto-detects halos and sets
   up communication. The number of halo rings comes from
   `AMGX_config_get_default_number_of_rings`.

### Memory implications

Because each rank reads the whole file, peak host memory does **not** shrink
with rank count. For `nnz` nonzeros and `R` rows, per-rank host peak is roughly:

- COO scratch: `nnz · (4 + 4 + 8)` bytes (held during conversion), plus
- CSR: `nnz · (4 + 8)` + `(R+1)·4` bytes, plus
- multi-GPU only: a local `int64_t` column copy `nnz_local · 8` + values.

Example (192³ 27-point, `nnz = 189,119,224`): host peak ≈ 5 GB for single-GPU.
This is a property of the file-based loader, not of AmgX (see §6).

---

## 5. Running

### Single-GPU CG

```bash
./amgx_cg_solver matrix/stencil3d_27pt_192.mtx --tol=1e-6 --runs=10
```

Options: `--tol=<f>` (default `1e-6`), `--max-iters=<n>` (default `5000`),
`--runs=<n>` (default `10`), `--json=<file>`, `--csv=<file>`.

> **To clarify:** the usage string prints `[--max-iters=1000]`, but the actual
> default in the code is `5000`. The usage text is stale; the effective default
> is 5000.

### Multi-GPU CG

```bash
mpirun --allow-run-as-root -np 2 ./amgx_cg_solver_mgpu \
    matrix/stencil3d_27pt_192.mtx --tol=1e-6 --runs=10
mpirun --allow-run-as-root -np 4 ./amgx_cg_solver_mgpu \
    matrix/stencil_5000x5000.mtx --runs=10
```

Options: as above, plus `--timers` (per-rank upload/solve/download breakdown and
load-imbalance table). One GPU is assigned per rank via `rank % device_count`.

> **Known issue:** `amgx_cg_solver_mgpu` aborts at `-np 1` with
> `AMGX ERROR: Incorrect parameters` on `AMGX_matrix_upload_all_global`. The
> distributed upload path does not accept a single-rank job. Use
> `amgx_cg_solver` for the 1-GPU data point (this is the documented split, see
> the 3D campaign README and `docs/results.md`).
>
> **Stale comment:** the file header of `amgx_cg_solver_mgpu.cpp` says it uses
> `upload_all()` "with local data, AmgX detects MPI context". The code actually
> uses `AMGX_matrix_upload_all_global` with an explicit `MPI_COMM_WORLD` passed
> to `AMGX_resources_create`. The comment predates the current approach.

### SpMV micro-benchmark

```bash
./amgx_spmv_bench matrix/stencil3d_27pt_192.mtx
```

Prints rows, nnz, mean SpMV time (ms), and a checksum.

### Output

The CG binaries print: convergence (`YES/NO` and iteration count), median time,
min/max/std, a GFLOPS figure (`2·nnz·iters / (median·1e6)`), and a solution
checksum (`Sum(x)`, `Norm2(x)`). With `--json`/`--csv` the same data is exported
via `amgx_benchmark.h`.

---

## 6. Known limitations

### `int` ceiling on nnz (~2.14e9)

`MatrixMarket.nnz` and `row_ptr` are `int`. The global number of nonzeros must
stay below `INT_MAX ≈ 2.147e9`. For a 3D 27-point stencil the nonzero count is

```
nnz = (3N - 2)^3
```

(the 27-point stencil is the tensor product of a 3-point 1D stencil; verified:
N=192 → 574³ = 189,119,224, matching the generated file). The ceiling gives:

| Stencil        | nnz formula        | grid where nnz ≈ 2.14e9 |
|----------------|--------------------|-------------------------|
| 3D 27-point    | `(3N-2)^3`         | N ≈ **430** (512³ overflows: 3.6e9) |
| 2D 5-point     | `5N² - 4N`         | N ≈ **20000** (20k² ≈ 2.0e9) |

So the 3D campaign is capped near 420³ with this harness; 512³ would require
changing the harness to 64-bit indices (and avoiding the full-file load below).
The 2D 5-point campaign at 20k² sits just under the same ceiling.

> Note: `(3N-2)^d` is the formula for the dense `3^d`-point stencil (9-point in
> 2D, 27-point in 3D). The 2D campaign used the **5-point** stencil, which has a
> different (lower) count, `5N² - 4N`.

### Whole-file load per rank

Each rank reads the entire `.mtx` (§4). There is no banded/streaming read, so:

- host memory per rank does not decrease with more ranks;
- startup cost grows with file size (line-by-line `fscanf` over all nnz);
- large 3D grids produce very large files (192³ ≈ 3.9 GB; a 512³ file would be
  ~100 GB), independent of the `int` ceiling above.

This is a harness property; AmgX's own per-rank (local) nnz easily fits in
`int` at higher rank counts.

### No overlap mode

AmgX manages its halo exchange internally; there is no knob here equivalent to
the custom 3D solver's `--overlap`. AmgX is compared as a single configuration.

### Multi-GPU checksum varies slightly with rank count

Expected for distributed iterative solvers — see §9.

### Minor / cosmetic

- Single-GPU usage text default for `--max-iters` is stale (says 1000, is 5000).
- `amgx_cg_solver_mgpu.cpp` header comment describes a non-distributed upload
  that the code no longer uses (§5).
- The grid label is cosmetic only; it never feeds the solve.

---

## 7. Comparison protocol

To make AmgX-vs-custom timings comparable, both sides use the same protocol:

| Aspect              | AmgX CG binaries                  | Custom 3D CG                      |
|---------------------|-----------------------------------|-----------------------------------|
| Warmup runs         | 3 (discarded)                     | 3 (discarded)                     |
| Timed runs          | 10 (default, `--runs`)            | 10                                |
| Reported statistic  | median                            | median                            |
| Outlier handling    | drop runs >2σ from mean, then median | median over converged runs     |
| Timed region        | `AMGX_solver_solve` only (vector reset and download excluded) | CG iteration loop |
| RHS / initial guess | `b = ones`, `x0 = 0`              | `b = ones`, `x0 = 0` (default mode) |
| Convergence         | `RELATIVE_INI`, L2, tol `1e-6`    | `||r||/||b|| < 1e-6`, L2          |

**Iteration count must match.** Comparing absolute solve time only makes sense
if both solvers take the same number of iterations to converge; otherwise the
comparison conflates per-iteration cost with iteration count. This is checked on
every run. In the local 192³, np=1 validation all three (AmgX, custom Sync,
custom Overlap) converged in **227** iterations with matching solutions
(`Sum(x)` agreeing to ~13 significant digits). See
`exploration_amgx_3d/data/phase3_rtx4060_192.md`.

> The outlier rule differs slightly: the AmgX binaries drop runs more than 2σ
> from the mean before taking the median; the custom solver takes the median
> over its valid (converged) runs without the 2σ filter. With 10 runs and low
> variance this rarely changes the median, but it is not bit-identical
> bookkeeping.

---

## 8. Design decisions

**Unpreconditioned CG, not preconditioned CG.** The custom solver has no
preconditioner, so adding one to AmgX would change the algorithm and break the
same-method comparison (different iteration counts, different convergence path).
AmgX is therefore run as plain CG (`solver=CG`, or `PCG`+`NOSOLVER` on the
single-GPU path, which is equivalent).

**No AmgX multigrid.** AmgX's strength is algebraic multigrid, which converges
in far fewer iterations than CG but is a different method. Comparing custom CG
against AMG would not be interpretable as an implementation speedup, so AMG is
deliberately not used.

**Single-GPU via `amgx_cg_solver`, multi-GPU via `amgx_cg_solver_mgpu`.** The
distributed upload path rejects single-rank jobs (§5), so the 1-GPU point uses
the non-distributed binary. Both are unpreconditioned CG with the same
tolerance/norm, so the data points are consistent across the rank sweep. This
matches the 2D showcase methodology.

**Global column indices as `int64_t` on the multi-GPU path.** Even though the
*global* nnz must fit in `int` for the file loader (§6), the multi-GPU upload
uses `int64_t` global column indices as required by
`AMGX_matrix_upload_all_global`.

---

## 9. Expected behavior: multi-GPU checksum variation (~0.15%)

With multiple MPI ranks the solution checksum varies slightly from the
single-rank run. **This is expected** for distributed iterative solvers.

Example (512×512 2D stencil, tol 1e-6):

```
1 rank:  sum=2.608806e+05, norm=509.87, 17 iterations
2 ranks: sum=2.612679e+05, norm=510.88, 17 iterations  (0.15% diff)
```

Why: floating-point addition is non-associative; `MPI_Allreduce` for the dot
products sums in an implementation-dependent order; domain decomposition changes
how rounding error accumulates in the distributed matrix-vector products. The
iteration count and the met tolerance are unchanged; variation appears at the
8th+ significant digit and is reproducible. Bit-exact reproducibility would
require deterministic reductions, at a performance cost.

---

## 10. Results

Benchmark results (2D 5-point, single and multi-GPU, 10k/15k/20k) are in the
consolidated [results page](../../../docs/results.md). The 3D 27-point campaign
(AmgX vs custom Sync/Overlap) and its local validation live under
`exploration_amgx_3d/` with the launcher `scripts/bench_amgx_3d_27pt.sh`.
