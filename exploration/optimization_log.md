# Optimization log — 27-point SpMV, NCU-driven cycles (RTX 4060)

Method: one cycle = hypothesis → mechanism → implementation → NCU measurement
→ verdict. All NCU numbers at NCU-fixed clocks (SM 1.14 GHz, DRAM 7.99 GHz),
matrix 192³ (7 077 888 rows, 189 119 224 nnz), 1 launch, harness
`bench_27pt_variants --profile`. Wall-clock numbers are paired medians from the
same process and are quoted as sign/rough-magnitude corroboration only (DVFS;
see the May 2026 boundary-cost study).

Baseline reference: `baseline_analysis.md` — DRAM 71.3 % of peak, traffic =
compulsory model to 0.06 %, LSU front-end saturated by the 216 B inter-thread
stride of the values stream (31.7 sectors and 31.7 L1 tag requests per warp
per values load; 59 % of all sectors excessive; LG-Throttle 43.3 % +
Long-Scoreboard 48.2 % of stall cycles).

## Cycle 1 — Coefficient-major (SoA) values layout

**Hypothesis (H3).** The values stream causes 100 % of the excessive sectors.
Storing values coefficient-major (`values_soa[c·n + row]`, 27 streams) makes
every values load stride-1 across threads; row_ptr/col_idx leave the hot path
(indices become implicit `row + δ_c`); boundary rows get 0.0 padding with
clamped indices (`0.0 · x[clamped] = 0.0`, exact); no branch → no divergence.

**Implementation.** `stencil27_soa_kernel_3d` (exploration/stencil27_soa_kernel.cu)
+ host transform `build_values_soa_host` (validates per-row ascending columns
and stencil-offset membership). Setup-time transform, production code untouched.
Memory: +27·n·8 B SoA buffer (1.529 GB at 192³) but the kernel no longer needs
col_idx (756 MB) nor row_ptr (57 MB) — net −0.80 GB if adopted standalone.
Correctness gate: **bitwise identical to the production kernel on all
7 077 888 rows** (max abs diff 0.0), boundaries included.

**Predictions (made before measuring) vs measured:**

| Prediction | Measured |
|---|---|
| Total sectors −59 % (248.7 M → ~103 M) | 101.1 M (**−59.3 %**) |
| values L1 tags 31.7 → 2.0 per warp-load | 2.0 (15.9 M tags over 36+ coalesced loads) |
| Excessive sectors ≈ 0 (residual: 18 k±1 x loads) | 3.97 M = 4 % (was 146.0 M = 59 %); all on k±1 x loads (9.0 sectors/warp, 11 %) |
| LG Throttle collapses | 43.3 % → **7.4 %** of stall cycles |
| DRAM read −3.4 % (drop row_ptr+col_idx, add padding) | 1.6433 → 1.5858 GB (model 1.5854, +0.03 %) |
| DRAM SOL ≥ 85 % (target) | **96.05 %** (245.7 GB/s) |

**Full comparison (ncu --set full, both kernels, same conditions):**

| Metric | baseline (CSR fast path) | SoA | |
|---|---|---|---|
| Elapsed cycles | 10 621 670 | 7 621 322 | **−28.2 %** |
| DRAM throughput (% peak) | 71.34 % | 96.05 % | +24.7 pt |
| Compute (SM) throughput | 45.38 % | 52.24 % | FP64 pipe 35.8 → 50.3 % of peak |
| L1/TEX throughput | 48.33 % | 33.43 % | front-end relieved |
| L1 tag requests (global) | 204.7 M | 28.3 M | **−86 %** |
| Sectors total / excessive | 248.7 M / 146.0 M (59 %) | 101.1 M / 3.97 M (4 %) | |
| Stalls: LG Throttle | 58.1 cyc (43.3 %) | 10.7 cyc (7.4 %) | queue unblocked |
| Stalls: Long Scoreboard | 64.6 cyc (48.2 %) | 125.7 cyc (87.7 %) | now pure DRAM latency at saturation |
| Warp cycles / issued inst | 134.1 | 143.4 | higher per-warp wait, more useful issue overall |
| Executed instructions | 72.5 M | 56.6 M | −22 % (no row_ptr load, no idx decomposition, no branch; +clamps) |
| Global LD / ST per warp | 55 / 1 (+ boundary loop) | 54 / 1 exactly | |
| Avg active threads / warp | 22.72 | **32.0** (0 divergent branches) | |
| Achieved occupancy | 79.5 % | 92.0 % | |
| Registers / thread | 40 | 40 | block limits unchanged (6) |
| L1 / L2 hit rate | 57.9 % / 50.8 % | 37.8 % / 21.2 % | expected: the vanished "hits" were recovered waste of the strided pattern, not lost locality |
| DRAM read / write | 1.643 GB / 56.8 MB | 1.586 GB / 56.7 MB | both = compulsory model |

Wall-clock paired medians (same process, 10 runs, 3 warmups): −31 % and −20 %
across two sessions — sign and magnitude consistent with the fixed-clock −28 %,
spread illustrates why wall-clock is not the metric here.

**Verdict: confirmed.** Every mechanism-level prediction landed. The kernel is
now pinned at the DRAM roofline (96 % SOL); NCU's own guidance flips from
"check coalescing" to "shift work away from DRAM".

**Interpretation note (cache hit rates).** L1 hit 57.9 → 37.8 % and L2 50.8 →
21.2 % are *improvements in disguise*: in the baseline, the strided values
loads requested 4× the sectors and the overlap between successive loads was
"hit" in cache — hits that only existed because of the redundant requests.
SoA values stream once with no reuse (hit rate ≈ 0 is correct for them); the
remaining hits are x's genuine 27× reuse. Cache hit rate is a diagnostic, not
an objective.

## Cycle 2 — Block size 128 / 256 / 512 (falsification test)

**Hypothesis.** At 96 % DRAM SOL the kernel is wall-limited, not
latency-limited: block size should not matter (it matters when you need more
in-flight warps to hide latency, or when tail/imbalance effects bite).

**Measured** (same report `ncu_soa_tuning_192.ncu-rep`, 14-pass sections):

| Block | Elapsed cycles | DRAM SOL | Achieved occupancy |
|---|---|---|---|
| 128 | 7 623 327 | 96.00 % | 95.0 % |
| 256 | 7 629 829 | 95.95 % | 93.2 % |
| 512 | 7 618 528 | 96.06 % | 89.4 % |

**Verdict: null result, as predicted** — spread 0.15 % in cycles while
occupancy varies by 5.6 pt. Documented because it is informative: it proves
occupancy is not the binding resource, which is exactly what "at the memory
wall" means. `__launch_bounds__` tuning is pointless here for the same reason.

## Cycle 3 — `__ldcs` (cache-streaming hint) on the values streams

**Hypothesis.** values are stream-once data passing 1.53 GB through L2
(32 MB); their lines may evict x's reuse window (~0.6 MB per active plane
pair, but many planes in flight). `__ldcs` marks them evict-first, protecting
x reuse. Expected effect: small (x is only 3.6 % of read traffic and its
reuse is mostly L1-served).

**Measured.** Elapsed cycles 7 629 829 → 7 560 449 (**−0.9 %**), DRAM SOL
95.95 → 96.07 %, L2 hit 21.3 → 21.6 %. Wall-clock paired medians cannot
resolve it (−0.9 % is below the DVFS noise floor).

**Verdict: marginal positive, keep-optional.** A one-intrinsic change worth
~1 % at fixed clocks; included in the variant file for completeness.

## Cycle 4 — Thread coarsening ×2 + 128-bit loads: bounded out, not run

Ceiling arithmetic before writing code: at 96.06 % DRAM SOL with traffic
already equal to the compulsory model, a perfect kernel gains at most
1/0.9606 − 1 ≈ **4.1 %**. Coarsening ×2 (one thread = rows 2r, 2r+1) would
halve values-load instructions via LDG.128 and cut x loads ~33 % via
intra-thread neighbor sharing — but instruction count is not the limiter
(issue slots 7.7 % busy), register pressure would rise from 40 (risking the
6-block occupancy tier), and the cap is 4 %. Decision: **not pursued** —
estimate the ceiling before writing the kernel.

## Status after Phase 2

The SoA kernel is at the memory wall: DRAM 96 % of peak, traffic = compulsory
to 0.03 %, stalls = DRAM latency at saturation, zero divergence, occupancy
92 %. Locally (RTX 4060, FP64 1:64, GDDR6 256 GB/s) there is ≤4 % of headroom
left for this algorithm/data structure. Further gains require changing the
*problem*, not the kernel: fewer bytes per row (e.g., constant-coefficient /
matrix-free stencil: drops the 1.53 GB values stream entirely — a different
operator contract), or mixed precision for the coefficients.

Real-time validation (wall-clock speedup, CG integration) belongs to
A100/H100 with locked clocks: FP64 1:2 and HBM change the ratios, datacenter
parts honor `nvidia-smi -lgc`, and the showcase baseline is A100.
