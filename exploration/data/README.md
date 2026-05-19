# Données brutes — coût de la branche boundary, SpMV 27-point 192³

Synthèse et conclusion : `../CONCLUSION.md`.
Tous les chiffres sont relus depuis les fichiers de ce dossier (runs réels).

GPU : NVIDIA GeForce RTX 4060 Laptop, CC 8.9, 24 SMs (GeForce portable).
Matrice : `matrix/stencil3d_27pt_192.mtx` (3 913 442 176 o, 7 077 888 rows,
189 119 224 nnz). Build : flags release du projet.

## Fichiers par méthode testée

### Diagnostic DVFS (méthodes 1–2)
| Fichier | Contenu |
|---|---|
| `diag_run2_stableclock_1470MHz.log` | Run avec `nvidia-smi` polling serré en // → SM clock figé 1470 MHz (la stabilité était un artefact du polling) |
| `diag_clock_trace_run2.csv` | Trace horloge correspondante |

### Entrelacé apparié — méthode 3 (harnais intermédiaire, depuis réécrit)
| Fichier | Contenu |
|---|---|
| `timing_rep{1,2,3}.log` | full→A→B par round, 30 warmup + 30 rounds, deltas appariés par round |
| `clock_trace_rep{1,2,3}.csv` | SM clock échantillonné pendant chaque rep |

### Tentative verrou sudo — méthode 4
| Fichier | Contenu |
|---|---|
| `locktest_tightpoll_artifact_1800.csv` | Poll **serré** pendant run "verrouillé" → 1800 constant (artefact) |
| `locktest_during_lockcheck_interleaved.log` | Run de vérif du verrou (harnais entrelacé d'alors) |
| `locked_clock_trace_rep{1,2,3}.csv` | Poll **doux** pendant run verrouillé → gr **52 → 1800 MHz** : verrou NON honoré |

### Séparé verrouillé, N=100, trim — méthode 5 (harnais final)
| Fichier | Contenu |
|---|---|
| `locked_separated_rep{1,2,3}.log` | Par-kernel séparé, 20 warmup + 100 runs, médiane + trimmed-mean + sd |

### NCU — méthode 6 (fiable, indépendant horloge)
| Fichier | Contenu |
|---|---|
| `ncu_metrics_run0.csv`, `ncu_metrics_rep{A,B}.csv` | `launch__registers_per_thread` + `sm__warps_active.avg.pct_of_peak_sustained_active`, 1 lancement/kernel, 3 reps |
| `ncu_stderr_run0.log` | stderr NCU (vide = pas d'erreur de permission) |

## Valeurs clés (relues des fichiers)

NCU (séparateur décimal = virgule dans les CSV) :

| Kernel | regs/thread (run0/A/B) | occupancy % (run0/A/B) |
|---|---|---|
| `stencil27_full` | 40 / 40 / 40 | 79.79 / 79.91 / 79.74 |
| `stencil27_boundary_neutralized` | 40 / 40 / 40 | 82.32 / 82.12 / 82.06 |
| `stencil27_interior_pure` | 40 / 40 / 40 | 85.59 / 86.35 / 86.73 |

Temps — deltas sur médiane (ms), seuls semi-fiables sur ce HW :

| Méthode | full−A | A−B | full−B |
|---|---|---|---|
| Entrelacé apparié (rep1/2/3) | +0.47 / +0.29 / +0.27 | −0.78 / −0.14 / −0.06 | −0.27 / +0.02 / +0.23 |
| Séparé verrouillé (rep1/2/3) | +0.79 / +0.45 / +0.61 | −0.99 / −1.39 / −1.37 | −0.20 / −0.94 / −0.76 |

Médianes par kernel ≈ 9.5–11.1 ms (full ≈10.3, A ≈9.6, B ≈10.9).
Trimmed-mean / sd / max des runs verrouillés = inexploitables (queue
49–80 ms : DVFS non maîtrisé, voir `locked_separated_rep*.log`).
