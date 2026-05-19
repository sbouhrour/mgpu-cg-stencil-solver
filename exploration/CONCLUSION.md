# Exploration — Coût de la branche boundary, SpMV 27-point

Branche : `exploration/spmv-27pt-boundary-cost`
Kernel de production `stencil27_csr_partitioned_halo_kernel_3d` **non modifié**.
Tous les chiffres ci-dessous proviennent de runs réels (fichiers dans `data/`).

---

## 1. Objectif

Mesurer ce que coûte le chemin *boundary* du kernel 27-point, en séparant :

- **coût d'EXÉCUTION** du travail boundary (indirection `col_idx` + routage halo) ;
- **coût de PRÉSENCE** de la branche (registres alloués pour le chemin
  boundary, susceptibles de dégrader l'occupancy même des warps 100 % interior).

Micro-test exploratoire borné. Un petit delta est une conclusion valide.

## 2. Phase 1 — Repérage (validé)

- Loader : `load_matrix_stencil27_3d_from_grid` (Path B, rank 0 / world 1)
  puis `build_csr_struct` — réutilisés tels quels.
- Struct CSR : `include/spmv_csr.h` (`long long* row_ptr`, `int* col_indices`,
  `double* values`) — types identiques à la signature kernel.
- **Ordre des colonnes** : ni le `.mtx` ni les loaders ne trient. Le tri
  croissant (requis par le fast path `values[csr_offset+0..26]`) est produit
  **uniquement** par l'insertion sort interne de `build_csr_struct`. Vérifié
  cohérent avec le mapping des 27 voisins.
- Matrice 192³ : `matrix/stencil3d_27pt_192.mtx`, **3 913 442 176 octets**
  mesurés, 7 077 888 rows, 189 119 224 nnz.

## 3. Les trois kernels (option (a), `__global__` séparés)

`exploration/bench_27pt_boundary_cost.cu`, config single-GPU
(1 rang, `x_halo_prev=x_halo_next=NULL`, `row_offset=0`), même config de
lancement (blocks=27648, threads=256) pour les trois :

- `stencil27_full` : copie EXACTE du kernel de production.
- `stencil27_boundary_neutralized` (A) : bloc `else` conservé entièrement
  (row_start, row_end, boucle, registres) ; corps réduit à `sum += values[jj];`.
- `stencil27_interior_pure` (B) : bloc `else` physiquement supprimé.

Checksums anti-DCE constants : full & A = 1.983752e+06 (résultat correct),
B = 0.0 (else supprimé, attendu).

## 4. Tout ce qui a été testé (méthodologie + échecs)

GPU : **RTX 4060 Laptop (CC 8.9, 24 SMs)** — GeForce portable grand public.

| # | Méthode | Résultat | Verdict |
|---|---|---|---|
| 1 | Séquentiel naïf (10 runs/kernel, blocs séparés) | full 8.69 ms net ; A 17.89, B 17.60 **bimodaux** (≈8 ↔ ≈18 ms) | **Échec** : le noyau le plus lourd 2× plus *rapide* que les variantes allégées — impossible. Artefact DVFS. |
| 2 | Diagnostic : `nvidia-smi` polling **serré** en parallèle | SM clock figé 1470 MHz, résultats cohérents (full 9.04 / A 8.25 / B 8.97) | Confirme la cause = DVFS. La stabilité venait du polling serré qui maintenait le GPU réveillé (artefact). |
| 3 | Entrelacé apparié (full→A→B par round, deltas par round) | médianes 10.0–10.5 ms, deltas appariés sub-ms reproductibles sur 3 reps | **Le moins mauvais** — mais la stabilité vient surtout du *GPU maintenu occupé* (lancements continus), pas de la soustraction. N'annule pas la rampe intra-round ni le biais de position fixe. |
| 4 | `sudo nvidia-smi -pm 1 -lgc 1800 -lmc 8001` | Vérif poll serré → 1800 constant **(artefact)** ; poll doux pendant run → gr **52 → 1800 MHz** | **Verrou NON honoré** par ce GeForce portable sous charge intermittente. |
| 5 | Séparé verrouillé, N=100, trim 10/côté (méthode standard) | médianes stables (full ≈10.3, A ≈9.6, B ≈10.9) ; trimmed-mean/sd/max **inexploitables** (queue 49–80 ms) | Médiane seule semi-fiable ; le DVFS persiste malgré le « verrou ». |
| 6 | NCU `launch__registers_per_thread`, `sm__warps_active...` | voir §5 | **Fiable et définitif** (statique / ratio, indépendant de l'horloge). |

Leçon : sur ce GPU portable, **aucune méthode logicielle ni le `sudo` ne
contrôle réellement le DVFS** pour les *temps absolus*. Seuls restent
exploitables : (a) les **deltas sur médiane**, cohérents en signe entre
méthodes ; (b) les **métriques NCU statiques**.

## 5. Résultats retenus

### NCU — définitif, indépendant de l'horloge (3 fichiers, reproductible)

| Kernel | registres/thread | occupancy atteinte % (run0 / repA / repB) |
|---|---|---|
| `stencil27_full` | **40** | 79.79 / 79.91 / 79.74 |
| `stencil27_boundary_neutralized` | **40** | 82.32 / 82.12 / 82.06 |
| `stencil27_interior_pure` | **40** | 85.59 / 86.35 / 86.73 |

→ La **présence/absence** de la branche ne change **pas** l'allocation
registre (40 dans les 3 cas) ⇒ même plafond d'occupancy théorique. L'écart
d'occupancy *atteinte* (~80 → ~86 %) n'est donc pas piloté par les registres.

### Temps — deltas sur médiane (seuls semi-fiables ici)

| Méthode | full−A (ms) | A−B (ms) | full−B (ms) |
|---|---|---|---|
| Entrelacé apparié (3 reps) | +0.47 / +0.29 / +0.27 | −0.78 / −0.14 / −0.06 | −0.27 / +0.02 / +0.23 |
| Séparé verrouillé (3 reps) | +0.79 / +0.45 / +0.61 | −0.99 / −1.39 / −1.37 | −0.20 / −0.94 / −0.76 |

Sur un noyau ≈10 ms. `full` ≥ `A` (full fait le vrai travail boundary,
logique) ; `full ≈ B` (signe instable). Bornage **structurel** : à 192³ les
rows boundary = **3.1 %** du total (190³/192³ = 96.9 % interior, fast path
identique aux 3 kernels) ⇒ l'effet whole-kernel ne *peut pas* être grand.

## 6. Conclusion

1. **Coût de PRÉSENCE de la branche : nul côté registres.** 40/40/40,
   prouvé NCU, indépendant de l'horloge. L'argument « la branche dégrade
   l'occupancy via la pression registre » **n'est pas soutenu** : registres
   identiques, plafond théorique identique.
2. **Coût d'EXÉCUTION : faible.** Sub-ms à ~1.4 ms sur ~10 ms, `full` très
   légèrement au-dessus de `A`, `full ≈ B`. Borné petit par construction
   (3 % de rows boundary à 192³). Conclusion : **branche boundary
   négligeable à cette taille.**
3. **Temps absolus non fiables sur ce HW** (GeForce portable, DVFS non
   maîtrisable même avec `sudo` — verrou non honoré). Pour un chiffre précis
   publiable, rejouer le mode séparé (déjà écrit) sur **A100/H100 dédiée +
   clocks réellement verrouillés** (cohérent avec le showcase projet =
   A100). Le résultat NCU, lui, est déjà définitif et se retrouvera identique.

## 7. Fichiers

Bench : `bench_27pt_boundary_cost.cu` (mode timing séparé + `--profile`
pour NCU). Données brutes et leur index : voir `data/README.md`.
