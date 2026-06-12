# Remote validation summary — NVIDIA A100-SXM4-80GB

Clock lock: none  |  NCU: present but counters blocked (expected on rented instances)
Median of per-rep medians (ms); ratio = variant / baseline.

| N | baseline | soa | soa_b128 | soa_b512 | soa_ldcs | soa/baseline |
|---|---|---|---|---|---|---|
| 128³ | 0.436 | 0.298 | 0.302 | 0.295 | 0.306 | **0.684** |
| 192³ | 1.445 | 0.989 | 0.999 | 0.977 | 1.015 | **0.685** |
| 256³ | 3.362 | 2.351 | 2.349 | 2.303 | 2.372 | **0.699** |
| 320³ | 6.644 | 4.562 | 4.585 | 4.505 | 4.621 | **0.687** |
| 384³ | 12.046 | 7.970 | 7.975 | 7.872 | 7.991 | **0.662** |

Per-rep spread is in medians.csv; clock traces per run alongside.
