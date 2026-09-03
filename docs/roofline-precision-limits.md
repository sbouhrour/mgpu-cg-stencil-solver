# Roofline limits and the precision question

> **What this page answers.** Where the stencil SpMV sits on the roofline, what bounds it, and whether
> tensor cores or reduced precision can move that bound. Every figure is measured with Nsight Compute at
> the DRAM boundary, so cache reuse is observed rather than assumed.
>
> **Two things this page is careful about.** Absolute speedups measured on a consumer GPU are marked as
> such and are **not** presented as results for datacenter hardware — the section [What a consumer part
> cannot settle](#7-what-a-consumer-part-cannot-settle) says exactly why, and which figures are still
> pending. And no conclusion here depends on the particular values in the test matrix; where that
> distinction matters it is stated explicitly.

**Hardware.** RTX 4060 Laptop GPU (AD107, CC 8.9, 24 SMs, 32 MB L2). Peak DRAM bandwidth **256.0 GB/s**
(8001 MHz × 128-bit, from `cudaDeviceProp`). Peak SM clock 3105 MHz → FP32 19.1 TFLOP/s, and **FP64
≈ 298 GFLOP/s** because consumer Ada carries two FP64 units per multiprocessor against 128 FP32 ones.
That ratio matters later.

**Reference points.** A100-SXM4-80GB: FP64 9.7 TFLOP/s vector, 19.5 TFLOP/s tensor core, 2039 GB/s.

---

## 1. Where the operator sits

| | 2D 5-point | 3D 27-point |
|---|---:|---:|
| Kernel | `stencil5_csr_direct_kernel` | `stencil27_csr_partitioned_halo_kernel_3d` |
| Problem | 5000 × 5000 grid, 25.0 M rows | 192³ grid, 7.08 M rows, 189 M nonzeros |
| Kernel time | 5.69 ms | 9.44 ms |
| DRAM traffic | 1.40 GB | 1.70 GB |
| **Bytes per row** | **56.0 B** | **240.2 B** |
| FLOP per row, from the SASS instruction mix | 10.0 | 52.5 |
| **Arithmetic intensity** | **0.179 FLOP/B** | **0.218 FLOP/B** |
| DRAM throughput | 96.0 % of peak | 80.5 % of peak |

The measured read/write split matches the kernel source term by term:

| Term | 5-point | 27-point | Reused across rows? |
|---|---:|---:|---|
| `values[]`, the stencil coefficients | 40 B | 216 B | **no** — each row reads its own |
| `x[]`, the input vector | 8 B | 8 B | yes — each element serves 5 (resp. 27) rows, amortised to one read |
| `y[]`, the output, written | 8 B | 8 B | — |
| `row_ptr[]` | 0 B — the offset is computed from grid coordinates | 8 B | — |
| `col_idx[]` | 0 B — column indices follow from the stencil structure | 0 B | — |
| **Total** | **56 B** (measured: 48 read + 7.9 written) | **240 B** (measured: 231.7 read + 8.05 written) | |

**The coefficient array is 71 % of 2D traffic and 90 % of 3D traffic.** Index elimination — the
optimisation this solver is built on — already removed everything else that could be removed.

---

## 2. Why tensor cores cannot help

A tensor core raises the *compute* ceiling. These kernels are held at the *bandwidth* floor, 22 to 27×
below the A100's FP64 ridge point of 4.76 FLOP/B. Enabling FP64 tensor cores doubles peak FLOP/s and
therefore moves the ridge to 9.56 FLOP/B: the gap widens from 27× to 44×. The unit is real; the
bottleneck it addresses is not the one this operator has.

**And the layout has a hard ceiling.** Each row performs a fixed 10 (resp. 52.5) FLOP and must read 5
(resp. 27) coefficients that are each used exactly once. Even with an infinite cache — `x` and `y` traffic
driven to zero — intensity is bounded by

$$\text{AI}_{\max} = \frac{10}{5 \times 8} = 0.25\ \text{FLOP/B (2D)}, \qquad \frac{52.5}{27 \times 8} = 0.24\ \text{(3D)}$$

No blocking, fusion or scheduling change passes 0.25 FLOP/B while the coefficients are stored and read
once per row. Reaching the ridge would need a twentyfold increase. **Coefficient traffic is the only lever
that matters**, and there are exactly two ways to act on it.

---

## 3. Lever one: specialise the operator — and why this solver does not

The test matrices are constant-coefficient: the 5-point operator emits `5.0` on the diagonal and `-1.0`
on its four neighbours, the 27-point one `26.0` and `-1.0`. An operator carrying those as compile-time
constants would move only `x` and `y` — 16 B/row, giving AI 0.625 in 2D and 3.28 in 3D, the latter above
this card's FP64 ridge.

**That is deliberately not done.** The solver is benchmarked against cuSPARSE and AmgX, which consume a
general sparse matrix, and the comparison holds only while every implementation reads the same operator
from memory. A kernel that knows its coefficients answers a question the others were not asked. The 2.08×
against cuSPARSE and the 1.44× against AmgX rest on that equal footing, and it is worth more than a
bandwidth factor. Geometric multigrid and specialised PDE codes go matrix-free and are right to; a general
sparse benchmark is not.

Two things are easy to conflate here, and only one of them is about bandwidth:

| | What it removes | Effect on DRAM traffic |
|---|---|---|
| Generating the matrix in memory instead of reading a `.mtx` — already done on the 3D path | file I/O and host memory; a 5000² grid is a 2.7 GB text file | **none** — the assembled CSR still lives in device memory and is still read |
| Hard-coding the coefficients in the kernel | the coefficient array itself | 3.5× in 2D, 15× in 3D, at the cost of value-agnosticism |

The first is a scalability measure. The second changes what is being compared.

---

## 4. Lever two: reduce coefficient precision

Storing `values[]` at a narrower width while accumulating in double halves or quarters the dominant term,
and leaves the comparison intact: storing the same operator at lower precision is a precision choice, not
knowledge of its values, and AmgX offers mixed-precision modes of its own.

**Whether it pays is decided by sector counts, not byte counts.** DRAM moves in 32-byte sectors, and a
warp-level load fetches whole sectors whether or not all of each is used:

| Coefficient format | Bytes/row (3D) | Sectors per warp load, **row-major CSR** | Sectors per warp load, **coefficient-major** |
|---|---:|---:|---:|
| double | 216 | 32 (per-thread stride 216 B) | 8 |
| float | 108 | **32** (stride 108 B) | 4 |
| half / bfloat16 | 54 | **32** (stride 54 B) | 2 |

In row-major CSR each thread reads its own 27 contiguous coefficients, so within one load instruction
warp-adjacent threads are 216 B apart in double and 108 B in float. Both exceed a sector, so each thread
occupies a sector of its own at either width — **the sector count is invariant under the change**.
Measured: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` is 246 976 440 for both, to the digit, while
DRAM traffic falls from 1.70 GB to 942 MB. Throughput drops from 71 % of DRAM peak to 38 % at constant
runtime.

**A byte count predicts traffic; only a sector count predicts time.**

A coefficient-major layout — 27 contiguous streams, one per stencil offset — makes each load stride-1
across the warp, and then the sectors halve with the width. It moves 99.5 M load sectors against 246.9 M
for row-major, within 4 % of the 95.6 M predicted from the stride argument, and reproduces the row-major
result bit for bit in double precision.

---

## 5. What actually limits the kernel

Once the coefficients are four bytes wide the vector loads account for 63 % of the sector count, which
points at staging `x` in shared memory as the next step. **An ablation says otherwise.** A probe keeping
every coefficient load and every multiply-add but reading the input vector **once** instead of 27 times —
and therefore bounding from above anything shared-memory staging could achieve — runs in 6.246 ms against
6.304 ms for the real kernel. **0.9 %.**

The vector loads dominate the sector count and are not the constraint. *A counter can be correct about
what a kernel does and silent about what limits it.* Ablating a term costs a few dozen lines; the
optimisation it would have justified costs a kernel.

**Two further probes identify the real limit.** Accumulating in float rather than double, leaving every
memory access identical, runs in 4.100 ms and lifts the kernel from 54 % to 84 % of DRAM peak. Splitting
the double accumulation across four independent partial sums, which allows four multiply-adds in flight
instead of one, gains nothing. So the constraint is **double-precision arithmetic throughput**, not the
latency of the dependency chain — and the 2.26 ms between the two probes is that arithmetic.

That single mechanism explains the whole ladder on this hardware:

| Coefficient storage | DRAM floor | + FP64 arithmetic | Measured | Regime |
|---|---:|---:|---:|---|
| double | 6.67 ms | — | 7.65 ms | memory-bound; the arithmetic hides behind memory |
| float | 3.57 ms | 5.83 ms | 6.36 ms | the arithmetic emerges and the two costs add |
| half / bfloat16 | 2.01 ms | 4.27 ms | 6.68 ms | the arithmetic dominates; nothing is gained |

Narrowing the coefficients removes enough traffic to make the double-precision accumulation visible. The
gain stops exactly there, which is why the narrowest format is not the optimum **on this part**.

⚠️ **This is a property of a consumer GPU.** Its FP64 rate is a sixty-fourth of FP32; a datacenter part
runs FP64 at half. The same 191 million multiply-adds cost 0.039 ms on an A100 against a 0.43 ms memory
floor — nine percent of the runtime rather than thirty-five. The regime is different there, so the
absolute speedups are not transferable; see § 7.

---

## 6. The numerical cost of reduced precision

**The constant-coefficient operator cannot measure it.** `26.0` is `1.101 × 2⁴` — four significand bits —
and FP8 E4M3 carries three plus the implicit one, so `26.0`, `-1.0` and `5.0` are exact in FP32, FP16,
BF16 and FP8 alike. Narrowing their storage produces exactly zero error, which measures nothing. That is a
limitation of the benchmark, not a result: a solver consuming a general matrix can assume nothing about
its coefficients.

A `% STENCIL_CONTRAST <c>` line in the matrix header therefore selects a discretisation of
∇·(a(**x**)∇u) with `a` in [1, 1+c]. Three properties are deliberate: the field is a polynomial evaluated
in a fixed order, with **no `libm` call**, so the matrix is bit-identical on every platform where `sin()`
would not be; it is scaled by a non-dyadic constant, so every coefficient occupies a full 53-bit
significand; and contrast 0 reproduces the constant-coefficient operator bit for bit, so existing results
are unaffected.

At 128³, contrast 0.7, with an input vector smooth over the three grid coordinates:

| Coefficient storage | Rounding of one coefficient | Error in `y`, normwise | Error in `y`, worst interior element |
|---|---:|---:|---:|
| float | 5.96e-8 | 5.04e-7 | **2.46e+02** |
| half | 4.88e-4 | 4.13e-3 | **3.05e+06** |
| bfloat16 | 3.91e-3 | 3.39e-2 | **8.71e+07** |

**The choice of metric changes the answer by nine orders of magnitude, and both metrics are correct.**

On an interior row the coefficients of a Laplacian sum to exactly zero, so applied to a smooth field the
result is a near-total cancellation: `y` is tiny while the individual terms are of order 26. Any
perturbation of the coefficients therefore dominates that element. The normwise figure does not show this
because it is dominated by boundary rows, which have fewer neighbours, do not cancel, and carry a
numerically large result.

Both readings are needed because they answer different questions:

- **Normwise** answers *does the solver still work?* Conjugate gradient operates on norms, so 5e-7 for
  float storage is the relevant figure, and the answer is yes.
- **Worst interior element** answers *can I trust an individual value of `y`?* No. A reduced-precision
  SpMV result is not a pointwise-accurate Laplacian — not for an error estimator, not for adaptivity, not
  for a residual read element by element.

**The error depends on the input, not only on the operator.** With a vector smooth in the *linear row
index* rather than over the grid, the normwise amplification falls from 8.5× to 1.05×: the stencil reaches
neighbours at index offsets of N and N², so such a vector jumps between neighbours in two of the three
directions and never cancels. An accuracy number without its input vector is half a result, which is why
the benchmark prints which one it used.

**One caveat on comparing the two layouts.** With constant coefficients the coefficient-major kernel
reproduces the row-major one bit for bit. With variable coefficients 1995 boundary rows out of 2 097 152
differ, at 4.4e-17 normwise. The cause is floating-point contraction, not a difference of operator: the
row-major boundary path is a loop and the coefficient-major path is unrolled, so the compiler contracts
multiply-adds differently. Building with `-fmad=false` restores bitwise equality. The earlier bitwise
claim held only because `26.0 × x` and `−1.0 × x` are exact products.

---

## 7. What a consumer part cannot settle

Everything above is either a property of the algorithm, a mechanism, or a numerical result, and all of it
transfers. **Absolute speedups do not**, and this section states plainly what is still open.

The layout and precision changes measure 1.15× and 1.34× against the production kernel on the RTX 4060,
with the layout change reproducible to 0.03 % across runs. Those figures are **not** offered as results
for datacenter hardware, for three independent reasons:

1. **The regime differs.** § 5 shows the kernel is limited by FP64 arithmetic throughput once the
   coefficients narrow. On a part whose FP64 rate is half of FP32 rather than a sixty-fourth, that term
   is nine percent of the runtime instead of thirty-five, so each ratio should approach the traffic
   ceiling it is bounded by. **Each figure must be read against its own base**, and there are two:

   | ratio | traffic ceiling | measured on the RTX 4060, 192³ |
   |---|---:|---:|
   | coefficient-major double → float — precision alone | 232 / 124 = **1.87×** | 1.16 – 1.23× |
   | row-major double → coefficient-major float — the production kernel | 240 / 124 = **1.93×** | 1.31 – 1.41× |

   Those are predictions, not measurements. Pairing the 1.87× ceiling with the 1.37× figure would
   compare a coefficient-major base against a row-major one; the two differ by the eight bytes of
   `row_ptr` that the coefficient-major layout does not read.
2. **The ridge points differ by a factor of four.** 1.17 FLOP/B here against 4.76 on an A100, so this
   card is the one *least* able to show a memory-side gain.
3. **Local variance approaches the effect, and at small sizes it exceeds it.** The ranges above are
   across five separate processes, not five repetitions inside one: the median over eleven timed
   repetitions is reproducible to a thousandth *within* a process and moves by up to a tenth *between*
   processes, so repeating inside one process measures the wrong dispersion. At 192³ that leaves a
   6 % spread against an effect of 20 %. **At 128³ it does not**: the same ratio comes out either
   ≈0.78 or ≈1.88 depending on the run — two sharp, stable states rather than a noisy one, and the
   spread is then four times the effect. It is not thermal, and it is not the clock: the fastest
   ratio was recorded at the *lowest* observed SM frequency. Whatever selects the state, a figure
   from a single 128³ process means nothing, which is why the measurements here start at 192³. The
   direction is solid, the magnitude is not.

Still pending, on hardware with stable clocks: the precision ladder on a datacenter GPU, the layout
comparison at solver level across several ranks, and a rebuild at `-O3` — the published figures are built
at `-O2` while AmgX is built at `-O3`, so they are conservative.

---

## 8. Reproducing

```bash
make bench_27pt_precision

# Layout and precision sweep, all variants timed round-robin
./bin/bench_27pt_precision matrix/stencil3d_27pt_192.mtx --reps=10 --csv=out.csv

# The numerical cost needs a variable-coefficient operator
./bin/bench_27pt_precision matrix/stencil3d_27pt_128_var0.7.mtx --reps=10 --x=grid

# Arithmetic intensity of the 2D kernel, at the DRAM boundary
ncu -k regex:stencil5_csr_direct_kernel \
    --metrics dram__bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum \
    ./bin/spmv_bench matrix/stencil_5000x5000.mtx --mode=stencil5-csr
```

Variants are timed **round-robin**, one launch each per repetition, rather than one variant to completion
and then the next: timing them in blocks lets the die heat under the early ones and clock down under the
late ones, which on this part was worth 27 % — more than the effect being measured. Three warmup rounds
are discarded and the median of ten is reported. Peak bandwidth and clocks come from `cudaDeviceProp` and
`nvidia-smi` on the same machine, so the percentages are not datasheet figures.
