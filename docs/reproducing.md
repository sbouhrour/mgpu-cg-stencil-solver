# Reproducing the Results

This is the authoritative page for reproducing the benchmark results. It explains how to build the solver, run the benchmark suite, and verify individual published numbers on your own hardware.

For methodology details (statistical approach, timing scope, profiling tools), see [`methodology.md`](methodology.md). For the published numbers themselves, see [`results.md`](results.md).

## Requirements

- **NVIDIA GPUs**: Compute Capability ≥ 7.0 (Volta, Turing, Ampere, Hopper)
- **CUDA Toolkit**: ≥ 11.0 with cuSPARSE and cuBLAS libraries
- **MPI Implementation**: OpenMPI ≥ 4.0 or MPICH ≥ 3.3
- **C++ Compiler**: Supporting C++11 (nvcc, g++, clang++)
- **Optional**: Nsight Systems/Compute for profiling

**Notes on common setup gaps:**

If `nvcc` is not in your `PATH` (frequent on fresh cloud images):

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

If MPI is missing, install it on Ubuntu/Debian (run `apt update` first, or `apt install` fails with "Unable to locate package"):

```bash
apt update && apt install -y libopenmpi-dev openmpi-bin
```

Without MPI, only the SpMV benchmark runs — the CG (single- and multi-GPU) and 3D benchmarks are skipped silently.

**Tested configurations:**

- NVIDIA A100-SXM4-80GB (8 GPUs) — primary development
- NVIDIA RTX 3090 (2 GPUs) — validation
- NVIDIA H100 NVL (single GPU) — compatibility

**Toolchain that produced the published Key Numbers** (8× A100-SXM4-80GB):

- CUDA 12.8, Driver 575.57
- OpenMPI: version not recorded in repo — to be confirmed
- AmgX: release/tag not recorded in repo — to be confirmed

## Quick smoke test

Verifies that the build and the full pipeline work, without running the large showcase problem sizes.

Clone the repository and enter it:

```bash
git clone https://github.com/sbouhrour/mgpu-cg-stencil-solver.git
cd mgpu-cg-stencil-solver
```

Then run the smoke test:

```bash
./scripts/run_all.sh --quick
```

`--quick` differs from the full run in two ways (see `scripts/run_all.sh`):

- Matrix size: **512×512** instead of the default 1000×1000.
- Runs per benchmark: **3** instead of 10.

Both the default 1000×1000 and `--quick`'s 512×512 are small pipeline-validation sizes meant to confirm everything runs end to end; they are not the showcase problem. Reproducing the published Key Numbers requires a showcase size such as `--size=20000`.

`--quick` runs only the custom solver path — AmgX is **not** included. To compare against AmgX, run `./scripts/setup/full_setup.sh --amgx` first, then run with or without `--quick`.

It builds the binaries, generates the matrix, then runs SpMV (single-GPU), CG (single-GPU), and — if MPI and ≥2 GPUs are present — multi-GPU CG and the 3D overlap benchmark. Outputs land in `results/raw/` (TXT), `results/json/` (JSON), and `results/3d/` (3D benchmark).

**How to know it worked:**

- A `PERFORMANCE SUMMARY` block is printed at the end, listing SpMV and CG timings with their ratios.
- TXT files appear in `results/raw/` (e.g. `spmv_512_<timestamp>.txt`) and JSON files in `results/json/`.

Runtime not measured — to be confirmed on local hardware.

> Absolute timings and ratios depend on hardware; the headline numbers in
> this page are 8× A100-SXM4-80GB measurements. Expect different values on
> other GPUs.

## Full benchmark suite

```bash
./scripts/run_all.sh                 # default 1000×1000
./scripts/run_all.sh --size=10000    # custom matrix size (e.g. reproduce a showcase point)
```

The script auto-detects the environment (GPU count, MPI, AmgX) and runs the applicable subset of:

- **SpMV single-GPU** — cuSPARSE CSR vs custom stencil (`spmv_bench`).
- **CG single-GPU** — custom solver (`mpirun -np 1`).
- **CG multi-GPU** — custom solver on all detected GPUs (only if MPI is present and `NUM_GPUS ≥ 2`).
- **AmgX single-GPU and multi-GPU** — reference comparison, only if AmgX is installed (see [AmgX comparison setup](#amgx-comparison-setup)).
- **3D stencil overlap** — runs `scripts/benchmarking/benchmark_3d_overlap.sh --quick --gpus=1,<NUM_GPUS>` (only if MPI is present and `NUM_GPUS ≥ 2`).

Options:

- `--size=N` — use an `N×N` 2D stencil matrix (`matrix/stencil_NxN.mtx`, generated if missing).
- `--quick` — see [Quick smoke test](#quick-smoke-test).
- `--help` — prints the option summary.

Outputs:

- `results/raw/` — raw TXT outputs.
- `results/json/` — structured JSON (timings parsed by the summary table).
- `results/3d/` — 3D benchmark JSON, raw logs, and a `summary_<timestamp>.txt`.

AmgX auto-detection: if AmgX is not installed, benchmarks 4 and 5 are skipped and the summary shows custom-solver timings without the AmgX comparison.

## AmgX comparison setup

A reviewer comparing against NVIDIA AmgX needs the AmgX build, which is optional and adds build time:

```bash
./scripts/setup/full_setup.sh --amgx     # install main components + AmgX
./scripts/run_all.sh                      # auto-detects AmgX once installed
```

With AmgX present, `run_all.sh` adds the single-GPU and multi-GPU AmgX reference runs and reports the Custom-CG-vs-AmgX ratios in the `PERFORMANCE SUMMARY`.

To validate the AmgX comparison quickly (small problem, fewer runs) or to go
straight for the headline numbers:

```bash
./scripts/setup/full_setup.sh --amgx     # one-time setup (adds build time)
./scripts/run_all.sh --quick             # or --size=20000 for headline numbers
```

### Manual AmgX install

`full_setup.sh --amgx` delegates to `scripts/setup/install_amgx.sh`, which clones, builds, and installs AmgX, then builds the comparison binaries. The equivalent manual steps, run from the repository root:

```bash
REPO=$(pwd)

# 1. Clone AmgX (the setup script tracks the main branch — see version note below)
git clone --depth 1 --branch main https://github.com/NVIDIA/AMGX.git /tmp/AMGX
cd /tmp/AMGX
git submodule update --init --recursive        # Thrust dependency

# 2. Configure (CUDA archs: 80 = A100, 86 = RTX 30xx, 90 = H100)
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$REPO/external/amgx" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_NO_MPI=0 \
      -DCMAKE_CUDA_ARCHITECTURES="80;86;90" \
      ..

# 3. Build and install (10-15 min)
make -j$(nproc)
make install
```

This installs into `external/amgx/`: headers in `external/amgx/include/` (`amgx_c.h`) and the library in `external/amgx/lib/` (`libamgxsh.so` shared, `libamgx.a` static). Then build the comparison binaries against it from the repository root:

```bash
export AMGX_DIR="$REPO/external/amgx"
make -C external/benchmarks/amgx
```

`run_all.sh` auto-detects AmgX via `external/amgx/include/amgx_c.h` (it also accepts `external/amgx-src/include/amgx_c.h`) and enables the reference runs.

Notes:

- `-DCMAKE_NO_MPI=0` requires MPI and is needed for the multi-GPU distributed API; without MPI the setup script sets `-DCMAKE_NO_MPI=1` and only single-GPU AmgX works.
- `install_amgx.sh` only passes `-DCMAKE_CUDA_ARCHITECTURES` automatically in detected cloud/container environments, where it derives the list from the CUDA version (e.g. `70;75;80;86;89;90` for CUDA 12.x). Setting it explicitly as above keeps the build portable across A100/RTX/H100.
- **Version**: the script clones the moving `main` branch (`AMGX_VERSION="main"`), not a fixed tag or commit, so the exact AmgX revision behind the published Key Numbers is **not pinned** in the repo. The recorded toolchain is CUDA 12.8 / Driver 575.57.

## Reproducing specific results

Each row maps a published number to the exact command that produces it. Expected values are the published figures in [`results.md`](results.md); they were measured on 8× A100-SXM4-80GB and will differ on other hardware.

| Key Number | Source | Command | What to check |
|---|---|---|---|
| SpMV vs cuSPARSE CSR (A100, 20k×20k): **2.08×** | [Results](results.md#2d-spmv-format-comparison) | `./bin/generate_matrix 20000 matrix/stencil_20000x20000.mtx`<br>`./bin/spmv_bench matrix/stencil_20000x20000.mtx --mode=cusparse-csr,stencil5-csr` | `Execution time` of `stencil5-csr` (12.86 ms) vs `cusparse-csr` (26.77 ms) → 2.08×. `spmv_bench` prints no ratio; for a precise figure run each mode separately. |
| CG single-GPU vs AmgX (20k×20k): **1.40×** | [Results](results.md#2d-custom-cg-vs-nvidia-amgx) | `./scripts/run_all.sh --size=20000` (AmgX build required) | `PERFORMANCE SUMMARY`: Custom CG (1 GPU) 531.4 ms vs AmgX (1 GPU) 746.7 ms → 1.40× |
| CG 8-GPU vs AmgX (20k×20k): **1.44×** | [Results](results.md#2d-custom-cg-vs-nvidia-amgx) | `./scripts/run_all.sh --size=20000` on an 8-GPU node (AmgX build required) | `PERFORMANCE SUMMARY`: Custom CG (8 GPUs) 71.0 ms vs AmgX (8 GPUs) 102.3 ms → 1.44× |
| 27pt overlap gain (256³, 8 GPUs): **1.45×** | [Results](results.md#3d-27-point-stencil-sync-vs-overlap) | `echo "% STENCIL_GRID_SIZE 256" > matrix/stencil3d_27pt_256.mtx`<br>`mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27`<br>`mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27 --overlap` | Median of sync run (294.0 ms) ÷ median of overlap run (203.5 ms) → 1.45× |
| 27pt scaling efficiency (512³, 8 GPUs, overlap): **88%** | [Results](results.md#3d-strong-scaling-efficiency-overlap-solver) | `echo "% STENCIL_GRID_SIZE 512" > matrix/stencil3d_27pt_512.mtx`<br>`mpirun -np 1 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_512.mtx --stencil=27`<br>`mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_512.mtx --stencil=27 --overlap` | 1-GPU sync median (22016 ms) ÷ 8-GPU overlap median (3110 ms) = 7.08× → 7.08/8 = 88% |

### 27-point matrix files

The 27-point solver does **not** read matrix entries from disk: it generates them in memory per rank from the grid size in the file header. Only the `% STENCIL_GRID_SIZE N` line is read, so a one-line header file is sufficient:

```bash
echo "% STENCIL_GRID_SIZE 512" > matrix/stencil3d_27pt_512.mtx
```

Generating the full file via `./bin/generate_matrix_3d_27pt 512 matrix/stencil3d_27pt_512.mtx` also works, but writes ~54 GB to disk that the solver ignores. Note that in-memory generation of the 512³ grid requires ~54 GB of host RAM (held on a single rank for the 1-GPU baseline run). The 7-point solver, by contrast, reads the matrix from a full `.mtx` file produced by `generate_matrix_3d`.

## Manual build and run

For understanding what the script does under the hood:

```bash
# Build all (spmv_bench, cg_solver_mgpu_stencil) — requires MPI
make

# Build AmgX benchmarks (requires AmgX installed)
make -C external/benchmarks/amgx

# Generate a 5-point stencil matrix
./bin/generate_matrix 1000 matrix/stencil_1000x1000.mtx

# --- Run benchmarks ---

# SpMV benchmark (single-GPU)
./bin/spmv_bench matrix/stencil_1000x1000.mtx --mode=cusparse-csr,stencil5-csr

# CG solver (single-GPU)
mpirun -np 1 ./bin/cg_solver_mgpu_stencil matrix/stencil_1000x1000.mtx

# CG solver (multi-GPU)
mpirun -np 2 ./bin/cg_solver_mgpu_stencil matrix/stencil_1000x1000.mtx

# AmgX comparison (if installed)
./external/benchmarks/amgx/amgx_cg_solver matrix/stencil_1000x1000.mtx
mpirun -np 2 ./external/benchmarks/amgx/amgx_cg_solver_mgpu matrix/stencil_1000x1000.mtx
```

### Custom single configuration

```bash
# Single configuration with JSON export
mpirun -np 2 ./bin/cg_solver_mgpu_stencil matrix/stencil_1000x1000.mtx --json=custom.json

# Extract timing from JSON
jq '.timing.median_ms' custom.json
```

### 3D solver (manual)

```bash
# 7-point: generate the matrix file, then run
./bin/generate_matrix_3d 256 matrix/stencil3d_256.mtx
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_256.mtx --overlap

# 27-point: header-only file (see "27-point matrix files" above), then run
echo "% STENCIL_GRID_SIZE 256" > matrix/stencil3d_27pt_256.mtx
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27 --overlap

# Verify correctness (b = A·1, check residual)
mpirun -np 8 ./bin/cg_solver_mgpu_stencil_3d matrix/stencil3d_27pt_256.mtx --stencil=27 --overlap --verify
```

## Profiling

Profiling commands are centralized here. For the analysis behind these traces, see [`profiling-2d.md`](profiling-2d.md) (kernel breakdown, roofline) and [`profiling-3d.md`](profiling-3d.md) (compute-communication overlap).

### Nsight Systems (timeline)

```bash
# Custom CG (1 GPU)
nsys profile --trace=cuda,nvtx -o custom_1gpu \
    ./bin/cg_solver_mgpu_stencil matrix/stencil_10000x10000.mtx

# Custom CG (multi-rank MPI)
nsys profile \
  --trace=cuda,nvtx,osrt,mpi \
  --trace-fork-before-exec=true \
  --stats=true \
  --cuda-memory-usage=true \
  --output=custom_mgpu \
  mpirun -np 4 ./bin/cg_solver_mgpu_stencil matrix/stencil_10000x10000.mtx

# AmgX (1 GPU)
nsys profile --trace=cuda,nvtx -o amgx_1gpu \
    ./external/benchmarks/amgx/amgx_cg_solver matrix/stencil_10000x10000.mtx

# View in GUI
nsys-ui custom_mgpu.nsys-rep
```

### Nsight Compute (roofline)

Used for the SpMV roofline analysis in [`profiling-2d.md`](profiling-2d.md#2-spmv-kernel-analysis):

```bash
# cuSPARSE CSR roofline
ncu --set roofline -o roofline_cusparse \
    ./bin/spmv_bench matrix/stencil_10000x10000.mtx --mode=cusparse-csr

# Stencil kernel roofline
ncu --set roofline -o roofline_stencil \
    ./bin/spmv_bench matrix/stencil_10000x10000.mtx --mode=stencil5-csr
```
