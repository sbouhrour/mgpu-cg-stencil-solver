# NCU profile summary — 27-point SpMV (production kernel), RTX 4060

Kernel: `stencil27_csr_partitioned_halo_kernel_3d` (unmodified production kernel,
linked from `src/spmv/spmv_stencil_3d_27pt_partitioned_halo_kernel.cu`).
Single launch. Config: single-GPU, 1 rank, `x_halo_prev = x_halo_next = NULL`,
`row_offset = 0`. Matrix: `matrix/stencil3d_27pt_192.mtx` (192³, 7 077 888 rows,
189 119 224 nnz). Launch: grid 27 648 blocks × 256 threads.

Report: `27pt_profile_rtx4060.ncu-rep` (`ncu --set full`, NCU 2025.2.1).
All values read back from that report; decimals rendered with `.` (NCU emitted `,`).

## Speed Of Light (SOL)

| Metric | Value |
|---|---|
| Memory throughput (% of peak) | 69.98 % |
| DRAM throughput (% of peak) | 69.98 % |
| Compute (SM) throughput (% of peak) | 44.52 % |
| L1/TEX cache throughput | 47.42 % |
| L2 cache throughput | 58.37 % |
| Duration | 9.50 ms |
| Elapsed cycles | 10 828 278 |

NCU rule output: "Memory is more heavily utilized than Compute."

## Occupancy

| Metric | Value |
|---|---|
| Theoretical occupancy | 100 % |
| Achieved occupancy | 79.42 % |
| Theoretical active warps / SM | 48 |
| Achieved active warps / SM | 38.12 |
| Registers per thread | 40 |
| Block Limit — Registers | 6 blocks |
| Block Limit — Warps | 6 blocks |
| Block Limit — Shared Mem | 16 blocks |
| Block Limit — SM | 24 blocks |
| Static / dynamic shared mem per block | 0 / 0 byte |

Limiter as NCU reports it: the theoretical block limits are **Registers = 6** and
**Warps = 6** (equal and binding); shared memory (16) and SM (24) are not binding.
Both binding limits yield 48 warps/SM = 100 % theoretical occupancy. NCU attributes
the gap between theoretical (100 %) and achieved (79.42 %) occupancy verbatim to
"warp scheduling overheads or workload imbalances during the kernel execution"
(Occupancy-section rule, Est. Speedup 20.58 %). There is no register / block-size /
shared-memory hard cap below 100 % in the report.

## Top 3 warp stall reasons

Warp Cycles Per Issued Instruction = 136.33 (basis for the percentages below).
Stall cycles per issued instruction, from `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`:

| Rank | Stall reason | Cycles | % of 136.33 |
|---|---|---|---|
| 1 | Long Scoreboard (L1TEX data dependency) | 65.94 | 48.4 % |
| 2 | LG Throttle (local/global instr-queue) | 60.00 | 44.0 % |
| 3 | Short Scoreboard | 3.75 | 2.8 % |

(Next: mio_throttle 1.92, wait 1.69, drain 1.41, selected 1.00, dispatch_stall 0.13,
branch_resolving 0.12, no_instruction 0.11; barrier/membar/sleeping/tex_throttle = 0.)

## Memory Workload Analysis

| Metric | Value |
|---|---|
| Memory throughput | 178.99 GByte/s |
| Max bandwidth (% of peak) | 69.98 % |
| Mem Busy | 58.37 % |
| Mem Pipes Busy | 16.70 % |
| L1/TEX hit rate | 57.81 % |
| L2 hit rate | 51.15 % |

## Compute Workload Analysis

| Metric | Value |
|---|---|
| Highest-utilized pipeline | FP64, 44.5 % (active cycles) |
| SM Busy | 44.54 % |
| Executed IPC (active) | 0.28 inst/cycle |
| Issued IPC (active) | 0.28 inst/cycle |
| Issue Slots Busy | 6.98 % |

Warp execution efficiency (from Warp State Statistics): avg active threads/warp 22.72;
avg not-predicated-off threads/warp 20.94.

## Hardware and clocks

`nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`:

```
name, driver_version, compute_cap
NVIDIA GeForce RTX 4060 Laptop GPU, 575.57.08, 8.9
```

24 SMs, 12 TPCs (from NCU Launch Statistics).

Clocks during the NCU run (from the report's SOL section):
- SM Frequency: 1.14 GHz
- DRAM Frequency: 7.99 GHz

Idle clocks before the run (`nvidia-smi --query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem`):
1470 MHz SM / 8000 MHz mem; max 3105 MHz SM / 8001 MHz mem.
Clocks were not externally locked for this run (NCU fixes clocks during its own
profiling passes by default; the SOL SM frequency above is the value in effect
during measurement).
