# Campagne AmgX-CG vs custom CG — 3D 27-point, 256³

Ancrage externe du showcase 3D : compare le CG **non-préconditionné** de
NVIDIA AmgX à notre solveur CG custom (modes **Sync** et **Overlap**) sur une
matrice 3D 27-point, à `{1, 2, 4, 8}` GPUs.

**Campagne unique à 256³**, 12 cellules : 3 méthodes × 4 tailles de GPU,
toutes mesurées dans le même run (mêmes conditions, même protocole médiane).

> Cette table 256³ **complète** le showcase ; elle ne remplace rien. Les
> chiffres 512³ existants dans `docs/results.md` restent intacts et inchangés.

## Pourquoi 256³

- **Faisabilité harnais AmgX** : `nnz = (3N−2)³`. Pour N=256 →
  766³ = **449 455 096** nnz (~450M), `.mtx` ~12 GB. Tient dans `int`
  (INT_MAX ≈ 2.15e9) et chaque rang charge le fichier global sans déborder.
  Le harnais AmgX plafonne vers ~420³ (à 512³, nnz=3.6e9 déborde `int` et le
  fichier ferait ~108 GB) — 256³ est confortablement sous ce mur.
- **Régime favorable à l'Overlap** : 256³ est la taille où le gain Overlap
  custom sur 8 GPUs est le meilleur (1.45× pour le 27pt, cf. `docs/results.md`).

Aucune modification du harnais AmgX n'est requise : il accepte 256³ tel quel.

## Protocole (identique pour les trois méthodes)

- CG **non-préconditionné**, `b = ones`, `x0 = 0`.
- Convergence : `||r|| / ||b|| < 1e-6`, norme L2, relative au résidu initial.
- 3 warmups + 10 runs chronométrés, **médiane** rapportée.
- Région chronométrée : le solve seul (hors upload/download des vecteurs).
- Matrice strictement identique des deux côtés (center 26.0, voisins −1.0) ;
  validé en Phase 3 par concordance de `sum(x)`/`norm2(x)` à ~13 chiffres.

Binaire AmgX selon le nombre de rangs (méthodo identique au 2D du showcase) :

| np   | Binaire AmgX            | API d'upload                  | Config solveur                       |
|------|-------------------------|-------------------------------|--------------------------------------|
| 1    | `amgx_cg_solver`        | `AMGX_matrix_upload_all`      | `solver=PCG, preconditioner=NOSOLVER`|
| ≥ 2  | `amgx_cg_solver_mgpu`   | `AMGX_matrix_upload_all_global` | `solver=CG`                        |

`PCG + NOSOLVER` ≡ CG non-préconditionné (préconditionneur identité) — les
deux configs sont algorithmiquement équivalentes, `RELATIVE_INI`, tol 1e-6, L2.
Le binaire mgpu (API distribuée) ne fonctionne pas à un rang unique ; d'où le
binaire single-GPU à np=1.

## Cible matérielle

8× A100 SXM4-80GB (un GPU par rang MPI). 256³ tient largement sur 80 GB par
GPU ; la lecture du `.mtx` ~12 GB par rang demande des nœuds à grande RAM
(typique des nœuds A100 SXM, ≥ 256 GB).

## Build (sur le nœud A100)

```bash
# Binaires custom
make generate_matrix_3d_27pt
make cg_solver_mgpu_stencil_3d

# Binaires AmgX (auto-détection d'AmgX, cf. external/benchmarks/amgx/README.md)
cd external/benchmarks/amgx
make amgx_cg_solver        # single-GPU (np=1)
make amgx_cg_solver_mgpu   # multi-GPU  (np>=2)
cd ../../..
```

## Lancement — une commande

Le script génère la matrice si absente, lance les 12 mesures et écrit une
table Markdown horodatée sous `exploration_amgx_3d/data/`.

```bash
scripts/bench_amgx_3d_27pt.sh 256 1 2 4 8
```

Surcharges d'environnement : `TOL` (déf. 1e-6), `RUNS` (déf. 10),
`MPIRUN_FLAGS` (déf. `--allow-run-as-root`). Exemple :

```bash
RUNS=10 MPIRUN_FLAGS="--bind-to none" scripts/bench_amgx_3d_27pt.sh 256 1 2 4 8
```

## Lancement — commandes manuelles équivalentes

Génération de la matrice (une fois) :

```bash
./bin/generate_matrix_3d_27pt 256 matrix/stencil3d_27pt_256.mtx
```

Pour chaque `np ∈ {1,2,4,8}` :

```bash
# --- AmgX-CG ---
# np = 1 :
./external/benchmarks/amgx/amgx_cg_solver \
    matrix/stencil3d_27pt_256.mtx --tol=1e-6 --runs=10
# np >= 2 :
mpirun --allow-run-as-root -np <np> \
    ./external/benchmarks/amgx/amgx_cg_solver_mgpu \
    matrix/stencil3d_27pt_256.mtx --tol=1e-6 --runs=10

# --- Custom Sync ---
mpirun --allow-run-as-root -np <np> \
    ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27

# --- Custom Overlap ---
mpirun --allow-run-as-root -np <np> \
    ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27 --overlap
```

## Sortie attendue

`exploration_amgx_3d/data/campaign_256_<timestamp>.md` — table 12 cellules :

```
| np | Method         | Iterations | Median (ms) | Converged |
|---:|----------------|-----------:|------------:|:---------:|
|  1 | AmgX-CG        | ...        | ...         | YES       |
|  1 | Custom Sync    | ...        | ...         | YES       |
|  1 | Custom Overlap | ...        | ...         | YES       |
|  2 | AmgX-CG        | ...        | ...         | YES       |
| ...                                                          |
|  8 | Custom Overlap | ...        | ...         | YES       |
```

Logs bruts par mesure sous `exploration_amgx_3d/logs/`.

## Contrôles de validité à vérifier après la campagne

- Les trois méthodes convergent au **même nombre d'itérations** à chaque `np`
  (algorithme aligné). En Phase 3 locale (192³, np=1) : 227 itérations pour
  les trois.
- `sum(x)` / `norm2(x)` concordent entre AmgX et custom (à la non-associativité
  flottante près des réductions MPI distribuées, cf.
  `external/benchmarks/amgx/README.md`).
- L'Overlap ne se distingue du Sync qu'à np ≥ 2 (à np=1 il n'y a pas de halo
  à recouvrir, les deux temps coïncident).

## Référence locale (Phase 3, RTX 4060, 192³, np=1)

Validation préalable du pipeline — voir `data/phase3_rtx4060_192.md`.
Résumé : AmgX-CG 4768.7 ms / Custom Sync 3231.5 ms / Custom Overlap 3245.7 ms,
227 itérations chacun, solutions identiques.
