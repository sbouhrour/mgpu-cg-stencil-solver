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

It leaves the comparison intact: storing the same operator at lower precision is a precision choice, not
knowledge of the operator's values, and AmgX offers mixed-precision modes of its own.

#### Measured: the traffic halves and the time does not

The 27-point kernel was templated on coefficient storage width and both variants run on the same matrix
(`bench_27pt_precision`, 192³, RTX 4060 Laptop, alternating A/B timing):

| | `values` double | `values` float |
|---|---:|---:|
| DRAM traffic (`dram__bytes.sum`) | 1.70 GB | **942 MB** (÷1.80) |
| L2 sectors (`lts__t_sectors.sum`) | 104 756 049 | 51 791 212 (÷2.02) |
| **L1 load sectors** | **246 976 440** | **246 976 440** — identical |
| DRAM throughput | 70.9 % of peak | 38.4 % of peak |
| Kernel time | 9.10 ms | 9.47 ms |
| **Speedup** | — | **0.96×** |

The predicted 1.82× does not appear **in this layout**. DRAM traffic drops exactly as modelled, but **the
number of 32-byte sectors the L1 must serve does not change at all**, and that is the binding constraint.
It does appear once the layout is changed — see [§ 3, coefficient-major layout](#measured-again-on-a-coalesced-layout).

The reason is the layout, not the precision. In row-major CSR a thread reads its own 27 contiguous
coefficients, so within one load instruction warp-adjacent threads are 27 × 8 = 216 B apart in double and
27 × 4 = **108 B** apart in float. Both strides exceed the 32 B sector, so every thread lands in its own
sector either way: 32 sectors per warp-level load, at both widths. Narrowing the element narrows the DRAM
footprint and the L2 traffic, but not the sector count — so the kernel simply moves from 71 % of DRAM peak
to 38 %, at the same speed.

Reduced precision therefore pays only once the coefficient loads are **coalesced**, which requires a
coefficient-major (SoA) layout where each of the 27 coefficient streams is contiguous across rows. The
point here is that a bandwidth model predicts the traffic correctly and the runtime not at all, whenever
the access is uncoalesced.

#### Measured again, on a coalesced layout

Same operator, coefficient-major storage, all six variants timed round-robin (192³, RTX 4060 Laptop,
median of the four cleanest runs):

| Variant | Time | vs CSR double |
|---|---:|---:|
| CSR double (production kernel) | 8.83 ms | 1.00× |
| SoA double | 7.65 ms | 1.15× |
| **SoA float** | **6.59 ms** | **1.34×** |
| SoA half / bfloat16 | 6.6 ms | 1.34× — no further gain |

Narrowing the coefficients is worth **1.16×** on top of the layout change, and the two compose to 1.34×
against the production kernel. Below float nothing more is gained: once the coefficient stream is 4 bytes
wide, the 27 vector loads dominate the sector count and further narrowing has nothing left to remove.
That is what the sector model predicted, and it is why the narrowest format is not the optimum.

The layout change alone is reproducible to 0.03 % across runs; the precision gain measures between 1.11×
and 1.21× depending on the run, because this is a laptop part whose SM clock swings between 26 MHz and
2025 MHz under power management. The direction is solid, the magnitude is not — it wants a card with
stable clocks.

#### The full precision ladder, in sectors

The sector argument generalises, and it is what decides whether a narrower format is worth anything:

| Coefficient format | Bytes/row | Sectors per warp load, **row-major CSR** | Sectors per warp load, **SoA** |
|---|---:|---:|---:|
| FP64 | 216 | 32 (stride 216 B) | 8 |
| FP32 | 108 | **32** (stride 108 B) — measured | 4 |
| FP16 / BF16 | 54 | **32** (stride 54 B) | 2 |
| FP8 | 27 | 27 B < 32 B, sharing begins | 1 |

In row-major CSR the per-thread stride only falls below a 32 B sector at FP8; in SoA every step down pays
immediately. Total traffic per row, SoA: 232 B at FP64 → 124 → **70** at FP16 → 43 at FP8, against 240 B
for the current CSR kernel.

These are the formats the tensor-core generation made native. Note what is and is not being claimed:
storing coefficients in FP16 or FP8 and accumulating in FP64 uses **conversion instructions on the CUDA
cores**, not the MMA pipeline — a stencil has no matmul shape to feed it. The value of those formats here
is bandwidth, not throughput. They are worth using for exactly that reason, and it is worth saying which
reason.

#### Why this matrix cannot measure the numerical cost

The zero error above is a property of the test matrix, not of the method. Checked against hardware
conversion on sm_89:

| | FP32 | FP16 | BF16 | FP8 E4M3 |
|---|---|---|---|---|
| Coefficients `26.0`, `-1.0` (3D) and `5.0` (2D) | exact | exact | exact | **exact** |
| Generic values (`1 + 0.5 sin`, 100 000 samples) | 99 999 inexact | 99 999 inexact, max rel. err. **4.9e-4** | 99 999 inexact, **3.9e-3** | — |

`26.0` is `1.101` × 2⁴ — four significand bits, and FP8 E4M3 carries three plus the implicit one, so the
constant-coefficient Laplacian survives every format intact.

**This is a limitation of the benchmark, not a result to build on.** A solver that consumes a general
sparse matrix cannot assume anything about its coefficients; designing the precision strategy around the
fact that *these particular* coefficients happen to be exact would be the same mistake as hard-coding
them. The honest figure is the second row: storing arbitrary values at width *w* costs roughly 2⁻⁽ᵖ⁺¹⁾
relative per element — about 6e-8 in FP32, 4.9e-4 in FP16, 3.9e-3 in BF16 — and that applies to
coefficients and vector alike.

The consequence is a prerequisite rather than a conclusion: **quantifying the accuracy cost of reduced
precision requires a variable-coefficient operator**, such as ∇·(a(**x**)∇u) with a spatially varying
diffusion field, which is also the problem class where the question actually arises. The constant-
coefficient Poisson matrix remains the right benchmark for bandwidth and scaling, and the wrong one for
precision.

#### Measured on a variable-coefficient operator

A `% STENCIL_CONTRAST <c>` line in the matrix header selects a discretisation of ∇·(a(**x**)∇u) with
`a` in [1, 1+c]. The field is a polynomial evaluated in a fixed order — no `libm` call, so the matrix is
bit-identical on every platform — scaled by a non-dyadic constant so that every coefficient occupies a
full 53-bit significand. The assembly is symmetric by construction and remains an M-matrix, so conjugate
gradient still applies, and contrast 0 reproduces the constant-coefficient operator bit for bit.

At 128³, contrast 0.7, with an input vector smooth over the three grid coordinates:

| Coefficient storage | Rounding of one coefficient | Error in `y`, normwise | Error in `y`, worst interior element |
|---|---:|---:|---:|
| float | 5.96e-8 | 5.04e-7 | **2.46e+02** |
| half | 4.88e-4 | 4.13e-3 | **3.05e+06** |
| bfloat16 | 3.91e-3 | 3.39e-2 | **8.71e+07** |

⭐ **The choice of metric changes the answer by nine orders of magnitude, and both metrics are correct.**

On an interior row the coefficients of a Laplacian sum to exactly zero, so applied to a smooth field the
result is a near-total cancellation: `y` is tiny while the individual terms are of order 26. Any
perturbation of the coefficients therefore produces a relative error of order one or worse *at that
element*. The normwise figure does not show this because it is dominated by boundary rows, which have
fewer neighbours, do not cancel, and carry a numerically large result.

Both readings are needed, because they answer different questions:

- **Normwise** answers *does the solver still work?* Conjugate gradient operates on norms, so 5e-7 for
  float storage is the relevant figure, and the answer is yes.
- **Worst interior element** answers *can I trust an individual value of `y`?* No. A reduced-precision
  SpMV result cannot be used as a pointwise-accurate Laplacian — not for an error estimator, not for
  adaptivity, not for a residual read element by element.

**The error also depends on the input, not only on the operator.** With a vector smooth in the *linear
row index* rather than over the grid, the normwise amplification falls from 8.5× to 1.05×: the stencil
reaches neighbours at index offsets of N and N², so such a vector jumps between neighbours in two of the
three directions and never cancels. Reporting an accuracy number without stating which vector produced
it reports half a result, which is why the benchmark prints the choice it used.

**One caveat on the layout comparison.** With constant coefficients the coefficient-major kernel
reproduces the row-major one bit for bit. With variable coefficients 1995 boundary rows out of 2 097 152
differ, at a normwise 4.4e-17. The cause is floating-point contraction, not a difference of operator: the
row-major boundary path is a loop and the coefficient-major path is unrolled, so the compiler contracts
multiply-adds differently. Building with `-fmad=false` makes the two bitwise identical again. The earlier
bitwise claim held only because `26.0 × x` and `−1.0 × x` are exact products.

### Lever 2 — specialise the operator, and why this benchmark does not

The test matrices are constant-coefficient: the 5-point operator emits `5.0` on the diagonal and `-1.0`
on all four neighbours for every row; the 27-point operator emits `26.0` and `-1.0`. A specialised
operator could therefore carry those coefficients as compile-time constants and move only `x` and `y` —
16 B/row, giving AI 0.625 in 2D and 3.28 in 3D, the latter above the RTX 4060's FP64 ridge.

**This solver deliberately does not do that**, and the reason is the point of the comparison rather than
a missed optimisation. The solver is benchmarked against cuSPARSE and AmgX, which consume a general
sparse matrix; the comparison is only meaningful while every implementation reads the same operator from
memory. A kernel that knows its coefficients is no longer solving the same problem — it is answering a
question the others were not asked. The measured 2.08× against cuSPARSE and 1.44× against AmgX rest on
that equal footing, and it is worth more than a bandwidth factor.

The distinction is worth stating precisely, because the two are easy to conflate:

| | What it removes | Effect on DRAM traffic |
|---|---|---|
| Generating the matrix in memory instead of reading a `.mtx` | file I/O and host memory (a 5000² grid is a 2.7 GB text file) | **none** — the assembled CSR still lives in device memory and is still read |
| Hard-coding the coefficients in the kernel | the coefficient array itself | 3.5× in 2D, 15× in 3D — at the cost of value-agnosticism |

The first is a scalability measure and is already used for the larger 3D cases. The second is what
geometric multigrid and specialised PDE codes do, and it is the right choice *for those codes*; it is the
wrong choice for a benchmark whose claim is a like-for-like comparison.

So the ceiling of this solver, by construction, is the 0.25 FLOP/B of § 2 — and reduced precision is the
lever that operates within it.

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
