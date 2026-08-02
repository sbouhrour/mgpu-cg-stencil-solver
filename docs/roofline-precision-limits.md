# Roofline Limits and the Precision Question

> **What this page answers.** Where does the stencil SpMV actually sit on the roofline, what is the
> ceiling of the current data layout, and would tensor cores or reduced precision move it? Every number
> below is measured with Nsight Compute, not modelled.

**Hardware.** RTX 4060 Laptop GPU (AD107, CC 8.9, 24 SMs, 32 MB L2). Peak DRAM bandwidth **256.0 GB/s**
(8001 MHz × 128-bit, from `cudaDeviceProp`). Peak SM clock 3105 MHz → FP32 19.1 TFLOP/s → FP64
**≈ 298 GFLOP/s** (consumer Ada runs FP64 at 1/64 of FP32).

**Method.** `ncu --metrics dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,
smsp__sass_thread_inst_executed_op_{dfma,dadd,dmul}_pred_on.sum`. FLOP counted from the SASS instruction
mix (FMA = 2 FLOP), bytes counted at the DRAM boundary — so cache reuse is included in the measurement
rather than assumed.

---

## 1. Measured arithmetic intensity

| | 2D 5-point | 3D 27-point |
|---|---:|---:|
| Kernel | `stencil5_csr_direct_kernel` | `stencil27_csr_partitioned_halo_kernel_3d` |
| Problem | 5000 × 5000 grid (25.0 M rows) | 192³ grid (7.08 M rows, 189 M nnz) |
| Kernel time | 5.69 ms | 9.44 ms |
| DRAM traffic | 1.40 GB | 1.70 GB |
| **Bytes per row** | **56.0 B** | **240.2 B** |
| FLOP per row (from SASS) | 10.0 | 52.5 |
| **Arithmetic intensity** | **0.179 FLOP/B** | **0.218 FLOP/B** |
| DRAM throughput | 96.0 % of peak | 80.5 % of peak |

### Where the bytes go

The measured read/write split matches the kernel source term by term:

| Term | 5-point | 27-point | Reused across rows? |
|---|---:|---:|---|
| `values[]` (stencil coefficients) | 40 B | 216 B | **no** — each row reads its own coefficients |
| `x[]` (input vector) | 8 B | 8 B | yes — each element serves 5 (resp. 27) rows, amortised to one read |
| `y[]` (output vector, write) | 8 B | 8 B | — |
| `row_ptr[]` | 0 B — offset computed from grid coordinates | 8 B | — |
| `col_idx[]` | 0 B — column indices derived from stencil structure | 0 B | — |
| **Total** | **56 B** (measured: read 48 + write 7.9) | **240 B** (measured: read 231.7 + write 8.05) | |

**The coefficient array is 71 % of 2D traffic and 90 % of 3D traffic.** Index elimination — the
optimisation this solver is built on — already removed everything else that could be removed.

---

## 2. Distance to the ridge point

| Platform | FP64 peak | Bandwidth | Ridge point |
|---|---:|---:|---:|
| A100-SXM4-80GB, FP64 vector | 9.7 TFLOP/s | 2039 GB/s | **4.76 FLOP/B** |
| A100-SXM4-80GB, FP64 tensor core | 19.5 TFLOP/s | 2039 GB/s | **9.56 FLOP/B** |
| RTX 4060 Laptop, FP64 | ≈ 298 GFLOP/s | 256 GB/s | ≈ 1.17 FLOP/B |

At AI 0.18–0.22, the kernels sit **22–27× below the A100 FP64 ridge**.

### Why tensor cores cannot help here

A tensor core raises the *compute* ceiling. These kernels are limited by the *bandwidth* floor. Enabling
FP64 tensor cores on an A100 doubles peak FLOP/s, which moves the ridge point from 4.76 to 9.56 FLOP/B —
i.e. it moves the target **further away**, from 27× to 44×. The hardware unit is real, the bottleneck it
addresses is not the one we have.

This is not a statement about tensor cores being unimportant; it is a statement about this operator with
this data layout.

### The ceiling of the current layout

Each row performs a fixed 10 (resp. 52.5) FLOP and must read 5 (resp. 27) coefficients that are used
exactly once. Even with an infinite cache — `x` and `y` traffic driven to zero — the intensity is bounded by:

$$\text{AI}_{\max} = \frac{10\ \text{FLOP}}{40\ \text{B}} = 0.25\ \text{(2D)}, \qquad
\frac{52.5\ \text{FLOP}}{216\ \text{B}} = 0.24\ \text{(3D)}$$

No blocking, fusion, or scheduling change gets past 0.25 FLOP/B while the coefficients are stored and
read once per row. Reaching the A100 ridge would require a 20× increase. **The coefficient traffic is
the only lever that matters.**

---

## 3. The two levers on coefficient traffic

### Lever 1 — reduce coefficient precision

Storing `values[]` in FP32 while accumulating in FP64 halves the dominant term.

| Variant | Bytes/row (2D) | Bytes/row (3D) | Traffic reduction |
|---|---:|---:|---:|
| Baseline (all FP64) | 56 | 240 | 1.00× |
| `values` FP32 | 36 | 132 | 1.56× / 1.82× |
| `values`, `x`, `y` FP32 | 28 | 124 | 2.00× / 1.94× |

Ceiling after FP32 coefficients: AI 0.5 (2D) — still an order of magnitude below the ridge. Reduced
precision buys **bandwidth**, not a change of regime.

### Lever 2 — do not store the coefficients at all

Both matrices are **constant-coefficient**: the 5-point operator emits `5.0` on the diagonal and `-1.0`
on all four neighbours for every row; the 27-point operator emits `26.0` and `-1.0`. The kernel currently
loads 40 B (resp. 216 B) per row from DRAM to retrieve the same two numbers, 25 million times.

A matrix-free operator with the coefficients as compile-time constants moves only `x` and `y`:

| | Current | Matrix-free | |
|---|---:|---:|---:|
| Bytes/row, 2D | 56 | **16** | 3.5× less traffic |
| Bytes/row, 3D | 240 | **16** | 15× less traffic |
| AI, 2D | 0.179 | **0.625 FLOP/B** | |
| AI, 3D | 0.218 | **3.28 FLOP/B** | |

**The 3D case crosses over.** At AI 3.28 the 27-point operator sits within 1.45× of the A100 FP64 ridge
(4.76), and *above* the RTX 4060's FP64 ridge (≈ 1.17) — it becomes compute-bound on consumer hardware.
Projected on the 4060 at 192³: the memory floor drops to 0.63 ms while the FP64 compute floor is
1.25–1.9 ms, so compute becomes the binding constraint and the realistic gain is ≈ 5–7× rather than the
15× that traffic alone would suggest.

That crossover is the honest answer to "where could tensor cores ever matter in this solver": only after
the coefficient traffic is eliminated, and even then an A100 would still be at 69 % of the ridge in FP64.

**Scope note.** A matrix-free operator is specialised to one discretisation, whereas cuSPARSE and AmgX
consume a general sparse matrix. Comparing them requires saying so explicitly: it is a data-structure
result, not a faster SpMV.

---

## 4. Reproducing

```bash
make spmv_bench
ncu -k regex:stencil5_csr_direct_kernel \
    --metrics dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum \
    ./bin/spmv_bench matrix/stencil_5000x5000.mtx --mode=stencil5-csr
```

Peak bandwidth and clocks are read from `cudaDeviceProp` and `nvidia-smi -q -d CLOCK` on the same machine,
so the percentages above are not taken from a datasheet.
