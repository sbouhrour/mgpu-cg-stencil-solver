# NCU Case Study: Optimizing the 3D 27-Point SpMV

This document walks through a complete Nsight Compute (NCU) optimization of the 3D 27-point stencil SpMV kernel — from baseline profile to a variant running at **96% of DRAM peak bandwidth** (from 71%), with **−28% elapsed cycles** at fixed clocks, subsequently validated at **1.43–1.51× wall-clock speedup on A100-SXM4-80GB** up to 1.5 billion nonzeros. The emphasis is on the *process*: how each metric was read, which hypotheses it eliminated, and how every optimization was predicted quantitatively before being measured. Experiments that returned null results are documented alongside the one that worked, because they are part of the evidence.

> **Hardware note.** All NCU metrics in this document were collected on an RTX 4060 Laptop GPU (Ada, CC 8.9, 24 SMs, GDDR6 ≈256 GB/s peak, FP64 at 1/64 of FP32 rate), CUDA 12.9, NCU 2025.2.1, with clocks fixed by the profiler (SM 1.14 GHz, DRAM 7.99 GHz). On this laptop GPU, DVFS makes absolute wall-clock times unreliable, so the analysis uses **static metrics and ratios only**; wall-clock validation is performed separately on A100 ([section 6](#6-hardware-validation-on-a100)). This mirrors the approach already used for the [2D roofline analysis](profiling-2d.md).

**Problem setup**: 192³ grid (7,077,888 unknowns, 189,119,224 nonzeros), double precision, single-GPU configuration of the partitioned kernel (1 rank, no halos). Kernel: `stencil27_csr_partitioned_halo_kernel_3d` (`src/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu`), launched as 1 thread per row, 256-thread blocks.

## Executive Summary

| Finding | Evidence |
|---------|----------|
| The baseline already moves the **minimum possible DRAM bytes** | Measured 1.700 GB vs analytic minimum 1.698 GB (+0.06%) |
| The bottleneck is **how** bytes are requested, not how many | 59% of L1 sectors excessive; LSU queue stalls 43% of cycles |
| All excess traffic comes from **one stream**: the CSR coefficients | 31.7 cache lines touched per warp-load vs 2 ideal (measured per-instruction) |
| Fix: **coefficient-major (SoA) storage** with implicit indices | DRAM 71.3 → **96.1%** of peak, cycles **−28.2%**, bitwise-identical output |
| Remaining headroom ≤ 4% → stop | Block-size sweep: null (±0.15%); `__ldcs`: −0.9%; further rewrites bounded out |
| A100 validation: **1.43–1.51× wall-clock**, 128³–384³ | Baseline 55–59% vs SoA 80–82% of HBM peak; 15/15 runs bitwise-identical |

---

## 1. The Baseline Kernel and Its Byte Budget

The production kernel already exploits the stencil structure aggressively. For interior rows (96.9% of rows at 192³), it performs a geometric interior test, reads a single `row_ptr` entry, and issues 27 unrolled fused multiply-adds with **computed** neighbor offsets — `col_idx` is never read on this path:

```c
// interior fast path (excerpt): offsets are arithmetic, not loaded
long long csr_offset = row_ptr[local_row];
sum  = values[csr_offset + 0] * x_local[local_row - N*N - N - 1];
sum += values[csr_offset + 1] * x_local[local_row - N*N - N    ];
// ... 25 more, ascending column order
```

Boundary rows (3.1%) fall back to a generic CSR loop with halo routing. An earlier branch-cost study (`exploration/spmv-27pt-boundary-cost`) bounded this path as negligible: register allocation is identical with or without it (40/40), and the whole-kernel timing effect is sub-millisecond on a ~10 ms kernel.

Before profiling, it pays to write down what the kernel *must* move per interior row — the **compulsory traffic** (bytes that have to cross DRAM at least once):

| Stream | Bytes per interior row | Share |
|--------|------------------------|-------|
| `values` (27 × 8 B coefficients) | 216 | **89%** |
| `x` (27 reads, but 27× reuse → 1 effective load) | 8 | 3.3% |
| `row_ptr` | 8 | 3.3% |
| `y` (store) | 8 | 3.3% |
| `col_idx` (boundary rows only) | ~0 | 0.9% overall |

That is ~240 bytes for 54 FLOP: an **arithmetic intensity of 0.22 FLOP/byte**. Ridge points (the intensity where a GPU becomes compute-bound) sit at ≈0.4 for this RTX 4060 at NCU clocks, ≈5 for A100-SXM4-80GB, ≈10 for H100-SXM — the kernel is **memory-bound by construction on every relevant architecture**, so optimization must target bytes and memory-pipe efficiency, not FLOPs.

## 2. Reading the Baseline Profile

`ncu --set full`, one launch, native `-arch=sm_89 -lineinfo` build. A guard run confirmed codegen equivalence with the project's canonical PTX/JIT build (all headline metrics within 0.3%).

### 2.1 Volume audit: is the kernel wasting bytes?

| | Analytic compulsory model | Measured (`dram__bytes_*`) |
|---|---|---|
| DRAM read | 1,641.9 MB | 1,643.3 MB (+0.06%) |
| DRAM write | 56.6 MB | 56.8 MB |

**The kernel moves the minimum possible bytes for this data structure.** The 27× reuse of `x` is fully absorbed by the caches. This single check eliminates an entire family of optimizations — anything aimed at "saving traffic" (shared-memory tiling of `x`, cache blocking) has nothing to save. A shared-memory tiling experiment had in fact been run earlier (branch `exploration/spmv-27pt-shared-tiling`): 2.48× *slower*, dominated by MIO-queue stalls — consistent with this audit.

The question becomes: if the volume is optimal, why does the kernel reach only **71.3% of DRAM peak** (182.5 GB/s)?

### 2.2 Rate audit: where do the cycles go?

| Metric | Value | Reading |
|--------|-------|---------|
| Issued IPC | 0.28 | schedulers idle most cycles |
| Cycles with no eligible warp | 92.9% | warps present (occupancy 79.5%) but stalled |
| Stall: Long Scoreboard | 48.2% of stall cycles | waiting on global-load data |
| Stall: **LG Throttle** | 43.3% | **cannot even issue loads — the LSU queue is full** |
| Occupancy limiter | none (100% theoretical) | 40 registers, no shared memory |

Occupancy is not the problem: the theoretical limit is 100% and no resource caps it. The pairing of *Long Scoreboard* (memory latency) with *LG Throttle* (load/store-unit queue saturation) points at transaction count, not parallelism. NCU's access-pattern rule quantifies it: *"only 12.6 of the 32 bytes transmitted per sector are utilized"* and *"145,980,736 excessive sectors (59% of the total)"*.

### 2.3 Attribution: which loads, exactly?

Per-SASS-instruction counters (`-lineinfo` build, Source page) attribute the excess precisely. Per warp-level load:

| Stream | Loads | L1 tag requests / warp | Sectors / warp (ideal 8) | Excessive |
|--------|-------|------------------------|--------------------------|-----------|
| `values` | 27 | **31.7** | **31.7** | **74.7% each** — 185.2 M sectors, 74.5% of kernel total |
| `x`, k-aligned neighbors | 9 | 2.0 | 8.0 | **0.0%** |
| `x`, k±1 neighbors | 18 | 2.8 | 8.8 | 9.4% (sector-boundary crossing) |
| `row_ptr`, `y` | 1+1 | 2.0 | 8.0 | 0.0% |

The geometric signature doubles as a model check: exactly 9 of the 27 `x` loads are k-aligned (the stencil offsets with Δk = 0) and load perfectly, while the 18 Δk = ±1 loads each cross one extra 32-byte sector. Model and measurement lock together.

### 2.4 The causal chain

CSR row-major storage means thread *t*'s 27 coefficients are contiguous *for that thread* but **216 bytes away from thread t+1's**. Each warp-level coefficient load therefore touches 32 distinct 128-byte cache lines (31.7 measured) instead of 2:

> stride 216 B across threads → 32 cache-line tag lookups per load (×16 the LSU front-end work on 89% of the traffic) → LSU queue fills (*LG Throttle*, with backpressure concentrating on each warp's first loads) → loads complete only after 32 sector fills (*Long Scoreboard*) → schedulers starve (0.14 eligible warps/cycle) → DRAM is fed at only 71% of peak.

DRAM traffic still stays compulsory because the sectors of successive loads overlap and L1 absorbs the redundancy — the waste is in *transactions*, not *bytes*. This distinction is what the fix exploits.

## 3. The Fix: Coefficient-Major (SoA) Storage

### Mechanism

Transpose the coefficients once at setup into 27 streams — `values_soa[c·n + row]` — the classic ELLPACK column-major idea (an ELLPACK operator also exists in this project's 2D SpMV suite). Three things compose:

1. **Coalescing**: for each coefficient *c*, warp-adjacent threads now read adjacent doubles — every coefficient load becomes stride-1 (2 tag lookups instead of 32).
2. **Implicit indices**: for a stencil operator on a structured grid, the column of coefficient *c* is always `row + δ_c` — `row_ptr` and `col_idx` leave the hot path entirely, which also removes the address dependency that gated all 27 coefficient loads behind the `row_ptr` fetch.
3. **Zero-padding + clamp**: boundary rows store 0.0 for absent neighbors and indices are clamped into `[0, n−1]`; the clamped load is in-bounds and multiplied by exactly 0.0, so *every row takes the same branch-free path* — warp divergence is eliminated by construction. (Assumes finite `x`, which CG guarantees.)

This is **not** a matrix-free kernel: coefficients are still read from memory, so variable-coefficient stencil operators remain supported. Only the *sparsity structure* is encoded in the layout — exactly the contract of this operator family. Memory cost: the SoA buffer equals the CSR values array (+16 MB padding at 192³), while `col_idx` (756 MB) and `row_ptr` (57 MB) become unnecessary — a net **−0.80 GB** of device memory if adopted standalone.

Correctness gate: **bitwise-identical output on all 7,077,888 rows** (max abs diff 0.0), boundaries included — the accumulation order matches the baseline, and inserting +0.0 terms leaves IEEE-754 sums unchanged.

### Predictions vs. measurements

Each effect was quantified from the baseline tables *before* running:

| Predicted | Measured |
|-----------|----------|
| Total sectors −59% | **−59.3%** (248.7 M → 101.1 M) |
| Coefficient loads: 31.7 → 2 tags/warp | **2.0** |
| Excessive sectors ≈ 0 (residual: the 18 k±1 `x` loads) | **4%** (was 59%) |
| LG Throttle collapses | **43.3% → 7.4%** |
| DRAM read −3.4% (no `row_ptr`/`col_idx`, +padding) | 1.643 → 1.586 GB (model: 1.585) |
| DRAM ≥ 85% of peak (target) | **96.05%** |

### Before / after

| Metric | Baseline (CSR) | SoA | |
|--------|---------------|-----|---|
| Elapsed cycles (fixed clocks) | 10,621,670 | 7,621,322 | **−28.2%** |
| DRAM throughput | 71.3% of peak (182.5 GB/s) | **96.1%** (245.7 GB/s) | at the wall |
| L1 tag requests | 204.7 M | 28.3 M | −86% |
| Executed instructions | 72.5 M | 56.6 M | −22% |
| Active threads per warp | 22.7 | **32.0** | divergence eliminated |
| Achieved occupancy | 79.5% | 92.0% | |
| Stall profile | LG Throttle 43% + Long Scoreboard 48% | Long Scoreboard 88% | now pure DRAM latency at saturation |

Two readings that *look* like regressions but are the expected signature of the fix:

- **Cache hit rates dropped** (L1 57.9 → 37.8%, L2 50.8 → 21.2%). The vanished "hits" were an artifact of the strided pattern: the baseline requested 4× the sectors and re-hit the overlap between successive redundant requests. Stream-once data with a near-zero hit rate is *correct*; the remaining hits are `x`'s genuine 27× reuse. Cache hit rate is a diagnostic, not an objective.
- **Warp latency rose** (134 → 143 cycles per issued instruction). Warps now wait almost exclusively on DRAM at 96% saturation — that *is* the roofline, not waste. Useful issue rate went up (IPC 0.28 → 0.31 on 22% fewer instructions).

## 4. Knowing When to Stop

At 96.1% of DRAM peak with compulsory traffic, the best possible remaining gain is 1/0.961 − 1 ≈ **4%**. Three cheap experiments closed the loop:

| Experiment | Hypothesis | Result | Lesson |
|------------|-----------|--------|--------|
| Block size 128 / 256 / 512 | at the DRAM wall, launch shape is irrelevant | cycles within **0.15%** while occupancy varies 89–95% | null result *confirming* the diagnosis: occupancy is not binding |
| `__ldcs` on coefficients | stream-once data evicts `x`'s reuse window from L2; evict-first hint protects it | **−0.9%** cycles | real but marginal; kept as optional |
| Thread coarsening + 128-bit loads | fewer, wider transactions | **not run** — ceiling arithmetic caps any gain at 4.1%, instruction issue is 7.7% busy, register pressure would rise | estimate the ceiling *before* writing the kernel |

Further speedup requires changing the *problem*, not the kernel: a constant-coefficient (matrix-free) variant would delete the entire 1.53 GB coefficient stream — a different operator contract, out of scope for a general stencil-coefficient SpMV.

## 5. Method Summary

The transferable checklist this case study follows:

1. **Write the byte model first** — per-stream compulsory traffic and arithmetic intensity, before any profiling.
2. **Audit volume against the model** (`dram__bytes` vs analytic minimum). If measured ≈ compulsory, stop optimizing traffic and start optimizing *transactions*.
3. **Classify the rate limiter** — stall reasons + scheduler statistics distinguish latency (long scoreboard), front-end saturation (LG/MIO throttle), and genuine bandwidth saturation; check whether occupancy is actually binding before touching it.
4. **Attribute to streams at the SASS level** — per-instruction sectors/tags name the culprit load, and structural signatures (here: 9 aligned vs 18 crossing `x` loads) cross-validate the model.
5. **Predict quantitatively, then measure** — every optimization came with falsifiable numbers (sector counts, stall shares, SOL targets) written down in advance.
6. **Bound the ceiling before coding** — 1/SOL − 1 is the most code you can ever save.
7. **Keep a blocking correctness gate** — bitwise comparison against the reference on every variant, boundaries included.
8. **Document null results** — the block-size sweep and the rejected tiling attempt are evidence about *why* the winning fix wins.

## 6. Hardware Validation on A100

NCU metrics above are static/ratio measurements on the RTX 4060 at profiler-fixed clocks. The wall-clock claim was validated separately on an **A100-SXM4-80GB** (driver 580.126.09, `-arch=sm_80` build): blocking correctness gates plus paired timing medians (3 × 10-run medians per variant) across 128³–384³ grids. GPU performance counters were blocked on the rented instance (no NCU there) and clock locking was not permitted; run-to-run stability is evidenced by the per-rep spread of **0.03–1.6%** across all configurations.

All 15 runs passed the bitwise gate (4/4 variants, up to 1.52 billion nonzeros). Implied bandwidth = analytic byte model / measured time, against the 2,039 GB/s HBM2e peak:

| Grid | Nonzeros | Baseline (CSR) | SoA | Speedup | Baseline → SoA bandwidth |
|------|----------|---------------|-----|---------|--------------------------|
| 128³ | 55.7 M | 0.436 ms | 0.298 ms | 1.46× | 57% → 80% of peak |
| 192³ | 189.1 M | 1.445 ms | 0.989 ms | 1.46× | 58% → 81% |
| 256³ | 449.5 M | 3.362 ms | 2.351 ms | 1.43× | 59% → 81% |
| 320³ | 879.2 M | 6.644 ms | 4.562 ms | 1.46× | 58% → 82% |
| 384³ | 1.52 B | 12.046 ms | 7.970 ms | **1.51×** | 55% → 81% |

Three observations close the loop on the RTX 4060 analysis:

- **The mechanism transfers, amplified.** The speedup is *larger* on A100 (1.43–1.51×) than the local fixed-clock −28% (1.39×). A100 offers more memory bandwidth per SM-cycle, so the baseline's LSU front-end saturation binds harder there: the CSR layout reaches only 55–59% of HBM peak, while the SoA layout holds a flat ~80–82% across a 27× range of problem sizes — bandwidth-limited and size-insensitive, as a stencil SpMV should be.
- **Block size remains a non-factor** (block 512 is consistently 1–2% faster on A100, which allows 64 resident warps/SM vs Ada's 48 — within the predicted "flat" band).
- **`__ldcs` does not travel**: −0.9% on the RTX 4060 but +0.5–2.6% on A100. Cache-policy hints are microarchitecture-dependent; the default policy is the right production choice.

Raw data (medians, per-run logs, summary) is archived on the exploration branch under `exploration/data/remote/`.

## Glossary

| Term | Meaning |
|------|---------|
| **Warp** | Group of 32 threads scheduled in lockstep; the unit of memory-request generation |
| **Coalescing** | Merging a warp's 32 addresses into few cache-line transactions; stride-1 across threads is ideal |
| **Sector** | 32-byte unit of memory transfer (4 per 128 B cache line); "excessive" = transferred beyond what used bytes require |
| **L1 tag request** | Cache-line lookup; each distinct line touched by a warp request costs one |
| **LSU** | Load/Store Unit — the SM front-end turning memory instructions into cache transactions |
| **LG Throttle** | Stall: the load/global instruction queue is full; signature of too many or too fragmented memory transactions |
| **MIO Throttle** | Same for the memory-IO queue (shared memory and special ops) |
| **Long / short scoreboard** | Stall waiting on data from global memory / on-chip dependencies |
| **SOL (Speed of Light)** | NCU's percentage of a subsystem's theoretical peak (DRAM, SM, L1, L2) |
| **Compulsory traffic** | Bytes that must cross DRAM at least once for the algorithm and data structure |
| **AoS / SoA** | Array-of-Structures vs Structure-of-Arrays; CSR row-major is AoS from the warp's perspective |
| **`__ldg` / `__ldcs`** | Load intrinsics with cache-policy hints: read-only path / streaming evict-first |
| **Occupancy** | Active warps per SM vs hardware maximum; buys latency hiding, irrelevant once bandwidth-bound |
| **Arithmetic intensity / ridge point** | FLOPs per byte; the intensity where compute peak equals bandwidth × intensity — below it, memory-bound |
| **DVFS** | Dynamic voltage/frequency scaling; why wall-clock is untrusted on laptop GPUs here |

## Artifacts

Branch `exploration/spmv-27pt-3d-ncu-analysis`: bench harness with blocking correctness gate (`exploration/bench_27pt_variants.cu`), SoA kernels (`exploration/stencil27_soa_kernel.cu`), full NCU reports (`exploration/data/*.ncu-rep`), baseline analysis and per-experiment log (`exploration/baseline_analysis.md`, `exploration/optimization_log.md`), and the A100/H100 validation runner (`exploration/run_remote.sh`). Core profiling command:

```bash
./exploration/build.sh   # release flags + native arch + -lineinfo
ncu --set full -k regex:stencil27_soa_k --launch-count 1 \
    -o report -f ./bin/bench_27pt_variants --profile
```
