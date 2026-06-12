# Raw data — NCU analysis of the 27-point SpMV baseline, 192³

Analysis and interpretation: `../baseline_analysis.md`.

| File | Content |
|---|---|
| `ncu_baseline_27pt_192_rtx4060.ncu-rep` | `ncu --set full` (40 passes), production kernel, 1 launch via `bench_27pt_variants --profile`, build = release flags + `-arch=sm_89 -lineinfo` |
| `ncu_baseline_jit_guard_192.ncu-rep` | Guard run: same kernel from the canonical JIT binary (`cg_solver_mgpu_stencil_3d --stencil=27 --max-iters=3`, launch-skip 2), headline sections only |
| `ncu_soa_27pt_192_rtx4060.ncu-rep` | `ncu --set full`, SoA variant (`stencil27_soa_kernel_3d`), 1 launch — Cycle 1 |
| `ncu_soa_tuning_192.ncu-rep` | Sections subset, 4 launches: soa@256 / @128 / @512 / soa_ldcs@256 — Cycles 2–3 |
| `hw_info.txt` | GPU, driver, clocks (idle), CUDA / NCU versions |

Reproduce:

```bash
./exploration/build.sh
ncu --set full -k regex:stencil27 --launch-count 1 \
    -o exploration/data/ncu_baseline_27pt_192_rtx4060 -f \
    ./bin/bench_27pt_variants --profile
```
