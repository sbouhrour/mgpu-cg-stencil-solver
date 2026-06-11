# NCU baseline analysis — 27-point SpMV (production kernel), RTX 4060

Kernel: `stencil27_csr_partitioned_halo_kernel_3d` (unmodified production kernel,
linked from `src/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu`).
Config: single-GPU, 1 rank, `x_halo_prev = x_halo_next = NULL`, `row_offset = 0`.
Matrix: 192³ 27-point stencil (7 077 888 rows, 189 119 224 nnz), generated in
memory by `load_matrix_stencil27_3d_from_grid`. Input `x[i] = sin(i*0.001)`.
Launch: 27 648 blocks × 256 threads (1D, 1 thread = 1 row).

Harness: `exploration/bench_27pt_variants.cu` (`--profile` mode, one launch).
Report: `data/ncu_baseline_27pt_192_rtx4060.ncu-rep` (`ncu --set full`, NCU 2025.2.1,
40 passes). Build: project release flags + `-arch=sm_89 -lineinfo`.

This report extends the May 2026 profiling (`exploration/spmv-27pt-profiling`
branch) with: (a) a build-equivalence guard, (b) a DRAM-traffic model checked
against measurement, (c) per-SASS-instruction attribution of the uncoalesced
sectors and stalls (the May report had no source/SASS correlation).

All numbers below are static/ratio NCU metrics at NCU-fixed clocks
(SM 1.14 GHz, DRAM 7.99 GHz). Absolute wall-clock times are meaningless on this
laptop GPU (DVFS, see the May boundary-cost study) and are never used.

## 0. Build-equivalence guard (native sm_89 + lineinfo vs canonical JIT)

The project Makefile ships sm_52 PTX and JIT-compiles at runtime. Profiling uses
a native `-arch=sm_89 -lineinfo` build. One guard run of the canonical JIT binary
(`bin/cg_solver_mgpu_stencil_3d --stencil=27`, report
`data/ncu_baseline_jit_guard_192.ncu-rep`) confirms equivalence:

| Metric | native sm_89 | canonical JIT | delta |
|---|---|---|---|
| Elapsed cycles | 10 621 670 | 10 635 228 | +0.13 % |
| DRAM throughput (% peak) | 71.34 % | 71.26 % | −0.08 pt |
| Compute (SM) throughput | 45.38 % | 45.64 % | +0.26 pt |
| Registers / thread | 40 | 40 | = |
| Achieved occupancy | 79.47 % | 79.53 % | +0.06 pt |
| Avg active threads / warp | 22.72 | 22.72 | = |

Same codegen behavior for this kernel; `-lineinfo` does not affect optimization.
The May numbers (SOL 69.98 %, 9.50 ms) also reproduce within ~2 % (71.34 %, 9.32 ms).

## 1. Speed of light and roofline position

| Metric | Value |
|---|---|
| DRAM throughput | **71.34 %** of peak (182.5 GB/s) |
| Compute (SM) throughput | 45.38 % (FP64 pipe is the highest-utilized) |
| L1/TEX throughput | 48.33 % |
| L2 throughput | 58.58 % |
| Duration / cycles | 9.32 ms / 10 621 670 (fixed clocks) |
| FP64 utilization | 17.16 of 48 peak thread-DFMA/cycle = 35.8 % of FP64 peak |

Arithmetic intensity (measured): 2·nnz = 378.2 MFLOP over 1.700 GB DRAM traffic
→ **AI = 0.222 FLOP/byte**. Ridge points: ≈0.43 FLOP/B on this RTX 4060 at NCU
clocks (FP64 1:64), ≈6.3 on A100, ≈10 on H100. The kernel sits left of the ridge
everywhere — **memory-bound by construction**, and far more so on the data-center
targets than on this consumer GPU (whose 1:64 FP64 rate compresses, but does not
invert, the hierarchy). Optimizations must target bytes and memory-pipe
efficiency, not FLOPs.

## 2. DRAM traffic: measured = compulsory model to 0.06 %

Per-stream compulsory model at 192³ (interior row: 27 values + ~1 x + 1 row_ptr
read, 1 y write; col_idx only on the 3.1 % boundary rows):

| Stream | Model | Measured (NCU) |
|---|---|---|
| values (8 B/nnz) | 1 512.95 MB | — |
| x (8 B/row, ×27 reuse served by caches) | 56.62 MB | — |
| row_ptr (8 B/row) | 56.62 MB | — |
| col_idx (4 B/boundary nnz) | 15.70 MB | — |
| **Total read** | **1 641.9 MB** | **1 643.3 MB** (`dram__bytes_read.sum`) |
| **Total write (y)** | **56.62 MB** | **56.81 MB** (`dram__bytes_write.sum`) |

**The kernel already moves the minimum possible DRAM bytes for this data
structure** (+0.06 % read, +0.3 % write). There is no over-fetch to recover:
x's 27× reuse is fully absorbed by L1/L2 (L1 hit 57.9 %, L2 hit 50.8 %), and the
values stream's strided sectors are coalesced before reaching DRAM. Conclusion:
the optimization question is not *volume* but *rate* — why 71 % of peak
bandwidth and not >90 %?

## 3. Why the rate stalls: LSU front-end saturation + memory latency

Issue starvation: 0.28 IPC; schedulers issue every ~14 cycles; 92.9 % of cycles
have **no eligible warp** (0.14 eligible of 9.55 active warps/scheduler).
Occupancy is not the cause (theoretical 100 %, achieved 79.5 %, no hard limiter:
40 regs → register and warp block-limits both 6). Warps are present but stalled:

| Stall reason | Cycles per issued instruction | Share of 134.1 |
|---|---|---|
| Long Scoreboard (waiting for global-load data) | 64.6 | 48.2 % |
| **LG Throttle (L1 load/global instruction queue full)** | 58.1 | 43.3 % |
| All others | ≤3.8 each | 8.5 % total |

NCU's access-pattern rules quantify the cause: *"only 12.6 of the 32 bytes
transmitted per sector are utilized"* and *"uncoalesced global accesses
resulting in 145 980 736 excessive sectors (59 % of the total 248 745 912)"*
(Est. Speedup 58.6 %).

## 4. SASS-level attribution: the values stream is 100 % of the problem

Per-instruction source counters (native build, `-lineinfo`), interior fast path
(27 values loads = `LDG.E.64.CONSTANT [R8.64+0x0 … +0xd0]`, executed by 216 600
warps each):

| Stream | #LDG | L1 tag requests / warp | Sectors / warp (ideal) | Excessive sectors | Total sectors |
|---|---|---|---|---|---|
| values | 27 | **31.7** | **31.7 (8.0)** | **74.7 % each** | 185.2 M (74.5 % of kernel) |
| x, k-aligned neighbors (9) | 9 | 2.0 | 8.0 (8.0) | **0.0 %** | 15.6 M |
| x, k±1 neighbors (18) | 18 | 2.8 | 8.8 (8.0) | 9.4 % (sector-boundary crossing) | 34.4 M |
| row_ptr | 1 | 2.0 | 8.0 (8.0) | 0.0 % | 1.8 M |
| y (STG) | 1 | 2.0 | 8.0 (8.0) | 0.0 % | 1.8 M |
| boundary CSR loop (col_idx 32-bit + values + x) | — | — | — | 50–58 % but tiny counts | ~10 M |

Reading: with CSR row-major storage, thread *t* owns row *t*, so its 27 values
are contiguous **per thread** but **216 B apart across threads**. Each warp-level
values load touches 32 distinct 128 B cache lines (31.7 measured tag requests vs
2 for a coalesced 8 B load) and 32 sectors where 8 carry useful data. The effects
compose:

1. **LSU front-end work ×16 on the dominant stream** — 185.2 M of ~205 M total
   L1 tag requests come from values loads. The L1 instruction queue fills,
   warps cannot even issue their loads → LG Throttle 43.3 %. The backpressure
   concentrates on each warp's first global instructions (the `row_ptr` load
   alone carries 17 070 of the LG-throttle samples).
2. **Latency amplification** — each values load completes only after 32 sector
   fills; consumers (DFMA chain) wait → Long Scoreboard 48.2 %. The `row_ptr`
   load gates all 27 values addresses (address dependency), lengthening the
   critical chain.
3. DRAM itself stays at 71 % — fed too slowly by (1)+(2), not short of work.

Secondary observation: avg active threads/warp 22.72 is dragged down by the
divergent boundary-CSR loop (2.0 threads/warp on its instructions), but the May
boundary-cost study already bounded the whole-kernel effect as negligible
(3.1 % of rows). It is not a target by itself.

The compiler already emits `.CONSTANT` (read-only path) for all loads — there is
nothing to gain from `__ldg`. No 128-bit loads are present (all 8 B); the AoS
layout's 27·8 B per-thread run cannot be vectorized across threads.

## 5. Hypotheses for Phase 2, ranked

**H3 — coefficient-major (SoA) values layout** (primary candidate; structural
decision, to be validated before implementation). Transpose values into 27
streams `values_c[row]` (ELLPACK-style column-major — the same trick the
project's 2D ELLPACK operator already uses), pad boundary rows with
0-coefficients, neighbor indices become implicit (`row + δ_c`, clamped; padded
slots contribute 0·x[clamped] = 0 exactly). Effects predicted from the tables
above, to be falsified by measurement:

- values loads become stride-1 across threads: 31.7 → 2 tag requests, 31.7 → 8
  sectors per warp; kernel-total sectors −59 %; LG Throttle should collapse.
- row_ptr and col_idx leave the hot path entirely: DRAM traffic −56 MB (−3.3 %),
  one latency link removed; boundary divergence disappears (all rows fast path).
- Device memory net −0.80 GB (drop col_idx 756 MB + row_ptr 57 MB, padding +16 MB).
- Expected duration at fixed clocks: 1.6436 GB / (SOL·256 GB/s) → −10 % (SOL
  0.75) to −25 % (SOL 0.92). Falsifiable target: **DRAM SOL ≥ 85 %**.

**H4 — micro-tuning after H3** (cheap, one cycle each): block size 128/512,
`__launch_bounds__`, `__ldcs` streaming hint on values (L2-pollution test),
thread-coarsening ×2 + 16 B vectorized loads (SoA makes alignment workable).

**Rejected (May 2026 evidence, not retried)**: shared-memory tiling of x
(2.48× slower, MIO throttle — x reuse was never the problem, see §2/§4);
boundary-branch restructuring in isolation (cost bounded negligible).

## 6. Methodological caveats

- NCU fixes SM ≈ 1.14 GHz but DRAM ≈ 7.99 GHz during collection: the SM:DRAM
  capacity ratio is skewed vs free-running clocks. Comparisons between kernels
  profiled under identical NCU conditions remain valid; absolute durations are
  report-internal references only.
- Real-time validation (wall-clock speedups) is out of scope here and belongs
  to A100/H100 runs with locked clocks.
