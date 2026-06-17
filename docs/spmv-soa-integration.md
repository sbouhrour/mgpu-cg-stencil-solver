# Coefficient-Major (SoA) SpMV: Integration & Applicability

This page documents how the coefficient-major (SoA) 27-point SpMV kernel is wired into the multi-GPU CG solver, why it is *not* a drop-in CSR replacement, and when the layout is worth its cost in a real application. The kernel-level analysis that motivates it — the Nsight Compute case study taking the kernel from 71% to 96% of DRAM peak — lives in [`kernel-optimization-ncu.md`](kernel-optimization-ncu.md); this page is about the solver integration and the engineering tradeoff.

> **Hardware note.** The kernel speedups cited here are static NCU metrics on an RTX 4060 Laptop GPU (DRAM throughput, sector counts) and SpMV-level wall-clock on A100-SXM4-80GB (1.43–1.51× across 128³–384³). The **solver-level** numbers (full CG, multi-GPU, vs the CSR path) have not yet been measured on data-center hardware — the table in [Validation status](#validation-status) is a placeholder pending an A100/H100/cluster benchmark run. Output correctness, by contrast, is verified here (bitwise-identical to the CSR solver).

## Executive Summary

| Aspect | CSR path (default) | SoA path (`--spmv=soa`) |
|--------|--------------------|--------------------------|
| SpMV DRAM throughput (RTX 4060, NCU) | 71% of peak | **96% of peak** |
| SpMV wall-clock (A100, 128³–384³) | baseline | **1.43–1.51×** |
| Matrix footprint per unknown | 332 B | **216 B** (−35%) |
| CG output | reference | **bitwise-identical** |
| Matrix format available elsewhere | yes (standard CSR) | no (SpMV-specific) |

The SoA path is faster *and* lighter on the device, because the index array it removes (`col_idx`) was both a bandwidth cost and a storage cost. The catch is interoperability, addressed in [Applicability](#applicability-and-integration-cost).

## The `--spmv` option

```bash
# Default — unchanged behavior, standard CSR stencil kernel
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27

# Coefficient-major (SoA) SpMV
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27 --spmv=soa
```

`--spmv=csr` (default) leaves every existing result reproducible. `--spmv=soa` selects the coefficient-major kernel; it requires `--stencil=27` and the synchronous solver (it is rejected with `--overlap` for now — see [Validation status](#validation-status)). The JSON export tags SoA runs as `3d-stencil-27pt-soa` so benchmark tooling can separate them.

## What changes under the hood

The CSR stencil kernel is drop-in because it consumes the structures the solver already maintains. The SoA kernel changes **two data contracts**, which is why integration is a deliberate step rather than a kernel swap.

**Contract 1 — matrix layout.** CSR stores a row's coefficients contiguously (`values[row·27 + c]`), so warp-adjacent threads read 216 bytes apart — uncoalesced. SoA stores 27 coefficient streams (`values_soa[c·n_local + row]`), so each coefficient load is stride-1 across the warp. Because the matrix is a structured stencil, neighbor columns are implicit (`row + δ_c`), so `row_ptr` and `col_idx` are never read; boundary rows carry explicit 0.0 padding so every row takes one branch-free path. The transform runs once at setup (`build_values_soa_27pt_3d`) and validates that every entry maps to a stencil offset.

**Contract 2 — input-vector layout.** The CSR kernel takes the halo as three separate pointers `(x_local, x_halo_prev, x_halo_next)` and routes each column at run time. The SoA kernel takes one contiguous **ghost-layer** buffer:

```
x_ext = [  prev halo plane (N²)  |   local rows (n_local)   |  next halo plane (N²)  ]
```

so every neighbor is `x_ext[base + δ]` — uniform, branch-free, always in-bounds. The integration realizes this without touching the communication code: the local pointer is offset into the buffer (`d_p_local = d_p_ext + N²`), and the halo pointers **alias** the ghost planes (`d_p_halo_prev = d_p_ext`), so the existing halo exchange deposits received planes exactly where the kernel reads them. Only the two SpMV-input vectors (`x`, `p`) use the ghost layout; `r`, `b`, `Ap` and all BLAS-1 kernels are untouched.

The ghost-layer is a *design choice* that also eliminates boundary divergence — an SoA-values kernel could instead keep the three-pointer halo and accept a small boundary branch. The coefficient transpose (Contract 1) is the irreducible change; the vector layout (Contract 2) is optional.

## Correctness

The SoA kernel preserves the baseline's accumulation order and only inserts `+0.0` terms for absent boundary neighbors, which leaves IEEE-754 sums unchanged. The result is **bitwise-identical** to the CSR solver, so the CG trajectory is identical: same iteration count, residual, and solution checksums.

| Configuration | Result |
|---------------|--------|
| np = 1, 64³, `--verify` | PASS, identical to all printed digits (76 iters) |
| np = 2 and np = 4, 64³ | PASS, csr ↔ soa identical at each rank count |
| np = 2, 128³ | PASS, identical (147 iters) |

Cross-rank-count drift (in the last digits) comes from MPI reduction order and is present identically in both modes — confirming it originates in the reductions, not the SpMV.

## Memory and capacity

The SoA layout removes `col_idx` (an `int` per nonzero) and `row_ptr`, at the cost of ~1% padding on the values array:

| Per unknown (27-point) | CSR | SoA |
|------------------------|-----|-----|
| values | 216 B | 216 B (+~1% padding) |
| col_idx | 108 B | 0 |
| row_ptr | 8 B | 0 |
| **matrix subtotal** | **332 B** | **216 B (−35%)** |
| 5 CG vectors | 40 B | 40 B (ghost layer adds ≈0) |
| **total** | **372 B** | **256 B (−31%)** |

Measured at 64³, the per-rank matrix footprint drops 84.4 → 56.6 MB. On the device this *raises* the maximum problem size per GPU by ≈1.45× in unknowns (≈+13% in linear grid dimension), since `col_idx` is a large array.

Two caveats:

- **Host vs device.** The current loader assembles COO then CSR on the host before upload, so the host-side setup peak (≈16 B/nonzero for COO) is the same in both modes and is what bounds the largest in-memory-generated grid. SoA's device headroom is fully usable only with a lighter assembly path — for a stencil, `values_soa` can be synthesized directly from the grid, skipping COO/CSR entirely (a future improvement).
- **No runtime switching.** The mode is fixed at setup; exactly one matrix representation is device-resident. There is no per-iteration cost and no second device array.

## Applicability and integration cost

The device memory win above holds when the matrix has a single consumer — which is the case inside this stencil-CG solver (CG touches the matrix only through SpMV). In a larger application the picture is different, and stating that honestly is part of the result.

**CSR is the interchange standard.** Assembly (FEM/FVM), preconditioners (ILU, AMG), and external libraries (cuSPARSE, PETSc, Trilinos, AMGx) all speak CSR. As soon as the matrix is shared beyond the SpMV, SoA becomes an *additional*, SpMV-specific copy — and the device accounting flips: holding both is ≈1.65× the CSR matrix footprint.

Two things bound that cost in practice:

- **The surplus usually lives on the host.** CSR is typically needed for CPU-side assembly and interop; the device can still be SoA-only. The VRAM penalty appears only when a *second GPU-resident* consumer (a GPU preconditioner, a GPU AMG hierarchy) also needs CSR at the same time.
- **The SoA-stencil trick is structured-grid-specific.** The implicit-column property only exists for structured stencils — and structured-grid codes often do not carry CSR at all (they are matrix-free or stencil-coefficient based), so the "CSR is standard" premise is weakest exactly where this layout applies. General unstructured matrices would instead use SELL-C-σ (sliced ELLPACK), the coalesced-but-general analog.

The layout sits on a generality ↔ performance spectrum:

| Format | Sparsity | Indices | Memory | Interop |
|--------|----------|---------|--------|---------|
| CSR | any | explicit | baseline | standard |
| SELL-C-σ / ELLPACK | structured-ish | explicit, coalesced | +padding | moderate |
| **SoA-stencil (here)** | structured stencil | implicit | **−35% (no col_idx)** | SpMV-only |
| matrix-free | constant-coefficient stencil | implicit | no values stored | none |

**The decision rule** the integration follows, and the one to apply when reusing the kernel elsewhere:

- Specialize the hot inner loop (SpMV); keep the standard format at the interfaces; convert once.
- Matrix reused across many solves (fixed operator, multiple RHS) → conversion amortizes → use SoA.
- Matrix changes every solve (nonlinear, adaptive) → conversion recurs → reconsider.
- Capacity-bound with a GPU-side CSR consumer → keep CSR, skip SoA.

For this solver — a structured-stencil CG where SpMV is the only matrix consumer and dominates runtime — SoA is a net win on both speed and capacity. The caveat above is the honest boundary of that claim.

## Validation status

| Level | Hardware | Status |
|-------|----------|--------|
| SpMV kernel, DRAM throughput | RTX 4060 (NCU, fixed clocks) | measured — 71% → 96% of peak |
| SpMV kernel, wall-clock | A100-SXM4-80GB | measured — 1.43–1.51× (128³–384³) |
| CG output correctness | RTX 4060 / multi-rank | verified — bitwise-identical |
| **Full CG solver, wall-clock** | **A100 / H100 / cluster** | **pending** |
| Overlap solver + SoA | — | not yet supported |

The pending row is the next step before this feature merges to `main`: rerun the 3D benchmark flow with `--spmv=csr` and `--spmv=soa` across the np = 1…8 configurations already used for the [3D results](results.md), with clocks locked, and report the solver-level speedup and any change in per-GPU capacity headroom.

## Reproducing

```bash
make cg_solver_mgpu_stencil_3d

# Correctness: SoA must match CSR exactly (header-only matrix file is enough)
echo "% STENCIL_GRID_SIZE 64"  > matrix/stencil3d_27pt_64.mtx
echo "262144 262144 6859000"  >> matrix/stencil3d_27pt_64.mtx
mpirun -np 2 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_64.mtx --stencil=27 --spmv=csr --verify
mpirun -np 2 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_64.mtx --stencil=27 --spmv=soa --verify

# A/B timing at a larger size (use locked clocks on data-center GPUs)
echo "% STENCIL_GRID_SIZE 256" > matrix/stencil3d_27pt_256.mtx
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27 --spmv=csr
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27 --spmv=soa
```
