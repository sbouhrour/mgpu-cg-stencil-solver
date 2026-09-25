# Communication Backends: Host Staging, CUDA-aware MPI, NCCL, NVSHMEM

> **Status: draft.** The method sections below describe the code on this branch and are complete.
> Every section marked *to be measured* is filled from the 8× A100-SXM4 campaign; nothing in it is
> estimated or carried over from another GPU.

The 3D solver moves two kinds of data between GPUs at every CG iteration: one XY-plane of `p` to each
Z-neighbour (the halo), and two scalars summed over all ranks (the dot products `p·Ap` and `r·r`,
on which α and β depend). This page compares four ways of doing it, on the same algorithm and the
same partition, and explains where each one spends its time.

## 1. The four backends

`--comm=MODE` on `cg_solver_mgpu_stencil_3d` selects how the halo moves; the solver and its numerics
are unchanged.

| Backend | Halo path | Host waits per halo |
|---|---|---|
| `staged` (default, the published results) | device → pinned host → MPI → pinned host → device | 2 |
| `gpuaware` | device pointers handed to a CUDA-aware MPI | 1 (MPI does not know CUDA streams) |
| `nccl` | `ncclSend`/`ncclRecv` enqueued on the solver's stream | 0 |
| `nvshmem` | one-sided puts into the neighbour's symmetric memory, ordered on the stream | 0 |

The host waits are counted with Nsight Systems, not inferred (section 5).

Options that change what the iteration synchronizes, independently of the halo backend:

- `--dots=device` keeps the CG scalars on the GPU: `cublasDdot` writes to device memory, the sum over
  ranks is enqueued on the stream (NCCL, NVSHMEM) and the BLAS1 kernels read α and β from device
  memory. The host reads `r·r` only to test convergence, every `--check-every=K` iterations.
- `--graph` (NCCL, device dots) replays K iterations as one CUDA graph.
- `--fused-halo` (NVSHMEM) removes the halo exchange as a separate step: the kernel that computes
  `p = r + βp` also stores its boundary planes directly into the neighbours' halo buffers, followed by
  one signal per neighbour.
- `--overlap` runs the interior/boundary split of [`profiling-3d.md`](profiling-3d.md) with any backend.

## 2. Correctness before timing

Every configuration must reproduce the residual history of the staged reference **bit for bit** at a
fixed iteration count (`--verbose=3` prints `r·r` in hexadecimal). A halo only moves bytes, so any
difference is a bug. The single legitimate exception is a device-side sum over three or more ranks by
NCCL or NVSHMEM, whose summation order differs from MPI's; there the relative deviation of `r·r` must
stay below 1e-10. `scripts/benchmarking/comm_preflight.sh` runs this check for every backend and mode
on the node before any timing, and a backend that fails it is not measured.

Library setup is never timed: communicators are created once per process, and each solve exercises
every send/receive buffer pair before its start event.

## 3. Measurement protocol

- Hardware: *to be measured* (8× A100-SXM4-80GB, NVLink NV12; topology checked with `nvidia-smi topo -m`).
- Software: CUDA 12.8 (as the rest of this repository's results), NCCL 2.31.2, NVSHMEM 3.7.2,
  Open MPI 5.0.8 with UCX 1.18.1, AmgX `cc1cebd` built with MPI. `scripts/benchmarking/comm_setup.sh`
  installs exactly this.
- Node check: an MPI ping-pong of 512 KB between two local ranks must exceed 3 GB/s
  (`rental_preflight.sh`, section 7), and the node must reproduce the published 27-point results of
  [`results.md`](results.md) before its multi-GPU numbers are used.
- Every backend converges in the same number of iterations, so configurations are timed at a fixed
  300 iterations and compared per iteration. Each point is the median of the timed solves.

## 4. Results

*To be measured.* Figure: time per iteration against GPU count, 27-point stencil, 128³
(communication-bound) and 512³ (compute-bound), one line per backend and AmgX with both of its
communicators. Full tables in [`results.md`](results.md).

## 5. What the timelines show

*To be measured.* One Nsight Systems timeline per path at 8 GPUs (staged, NCCL with device dots and a
CUDA graph, NVSHMEM with the fused halo), with per-iteration counts of CUDA API calls, host
synchronizations, kernels and copies (`scripts/benchmarking/comm_profile.sh`).

## 6. Calibration: what the links should give

*To be measured.* Latency and bandwidth of the node from `nccl-tests` (`comm_calibrate.sh`), the
α-β model fitted to them, and how far the solver's measured communication is from the model's
prediction.

## 7. Reproducing

```bash
./scripts/benchmarking/comm_setup.sh && source /opt/comm/env.sh
./scripts/benchmarking/rental_preflight.sh
./scripts/benchmarking/comm_preflight.sh
./scripts/benchmarking/comm_calibrate.sh
SET=core ./scripts/benchmarking/comm_matrix.sh
./scripts/benchmarking/comm_profile.sh
python3 scripts/visualizations/plot_comm_backends.py --input-dir=out/comm_matrix
```
