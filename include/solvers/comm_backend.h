/**
 * @file comm_backend.h
 * @brief Communication backends for the Z-slab multi-GPU CG solvers
 *
 * The solver communicates in exactly two ways per iteration:
 * - a halo exchange: one XY-plane to each Z-neighbour (point-to-point);
 * - scalar reductions: dot products summed over all ranks.
 *
 * A CommContext owns everything a backend needs (host staging buffers, library
 * communicators) and is created once per process, before any timed solve, so
 * that no connection setup or buffer allocation is ever charged to the solve.
 *
 * Backends:
 * - staged:   device -> pinned host -> MPI -> pinned host -> device
 * - gpuaware: device pointers passed straight to MPI (requires a CUDA-aware MPI)
 * - nccl:     ncclSend/ncclRecv enqueued on the solver stream (build with NCCL)
 * - nvshmem:  one-sided puts into the neighbour's symmetric memory, ordered on the solver
 *             stream (build with NVSHMEM; several ranks per GPU need CUDA MPS)
 *
 * The CUDA device must be selected before comm_create: a NCCL communicator
 * binds to the device that is current when it is created.
 */

#ifndef COMM_BACKEND_H
#define COMM_BACKEND_H

#include <mpi.h>
#include <stddef.h>
#include <cuda_runtime.h>

typedef enum {
    COMM_STAGED = 0,
    COMM_GPUAWARE = 1,
    COMM_NCCL = 2,
    COMM_NVSHMEM = 3,
} CommBackendKind;

typedef struct CommContext CommContext;

/** Parse "staged" / "gpuaware" / "nccl" / "nvshmem". Returns 0 on success. */
int comm_backend_parse(const char* name, CommBackendKind* kind);
const char* comm_backend_name(CommBackendKind kind);

/**
 * @brief Create the communication context (collective over mpi_comm)
 * @param max_halo_elems Largest halo (in doubles) any exchange will move
 * @return NULL if the backend is unavailable in this build or at runtime
 */
CommContext* comm_create(CommBackendKind kind, MPI_Comm mpi_comm, size_t max_halo_elems);
void comm_destroy(CommContext* ctx);
CommBackendKind comm_kind(const CommContext* ctx);

/**
 * @brief Exchange one plane with each Z-neighbour
 *
 * Sends d_send_prev to rank-1 and d_send_next to rank+1, receives into
 * d_recv_prev / d_recv_next. Pointers for a missing neighbour are ignored.
 * On return the received planes are visible to work later enqueued on stream.
 * staged and gpuaware block the host until the exchange completes; nccl and
 * nvshmem only enqueue it on stream and return.
 */
void comm_halo_exchange(CommContext* ctx, const double* d_send_prev, const double* d_send_next,
                        double* d_recv_prev, double* d_recv_next, int halo_elems,
                        cudaStream_t stream);

/**
 * @brief The same exchange split in three calls, to overlap it with interior work
 *
 * begin enqueues on stream what can be enqueued (staged: the device-to-host copies);
 * the caller then launches work that does not read the halo; post starts the transfer
 * (the MPI backends wait for stream here, nccl enqueues on it); end completes it. After
 * end, the received planes are visible to work enqueued on stream: a caller computing
 * on another stream records an event on stream and waits on it. One exchange at a time.
 */
void comm_halo_begin(CommContext* ctx, const double* d_send_prev, const double* d_send_next,
                     double* d_recv_prev, double* d_recv_next, int halo_elems, cudaStream_t stream);
void comm_halo_post(CommContext* ctx);
void comm_halo_end(CommContext* ctx);

/*
 * Fused halo (nvshmem only): the kernel that updates a vector stores its boundary planes straight
 * into the neighbours' halo buffers, then comm_fused_notify tells each neighbour; the neighbour's
 * comm_fused_wait orders its next read after those stores. There is no acknowledgement: the caller
 * must separate two updates by a global reduction (CG's p.Ap does), which cannot complete before
 * every neighbour has finished reading its halo.
 */

/** Device memory a neighbour can address: nvshmem_malloc for nvshmem (collective: every rank,
 *  same size, same order), cudaMalloc otherwise. */
double* comm_symmetric_alloc(CommContext* ctx, size_t elems);
void comm_symmetric_free(CommContext* ctx, double* p);

/**
 * @brief Direct pointers to the neighbours' copies of two symmetric halo buffers
 * @param d_halo_prev, d_halo_next  this rank's halo buffers (from comm_symmetric_alloc)
 * @param[out] into_prev  rank-1's d_halo_next (where my first plane goes), NULL without rank-1
 * @param[out] into_next  rank+1's d_halo_prev (where my last plane goes), NULL without rank+1
 * @return 0 if every existing neighbour is directly addressable (same GPU, NVLink, P2P)
 */
int comm_fused_peers(CommContext* ctx, double* d_halo_prev, double* d_halo_next, double** into_prev,
                     double** into_next);
void comm_fused_notify(CommContext* ctx, cudaStream_t stream);
void comm_fused_wait(CommContext* ctx, cudaStream_t stream);

/** Sum a host scalar over all ranks (MPI_Allreduce on the host for every backend). */
double comm_allreduce_sum(CommContext* ctx, double local);

/**
 * @brief Sum a device scalar over all ranks, in place, ordered on stream
 *
 * nccl enqueues an ncclAllReduce and returns without blocking. The MPI
 * backends must wait for the stream first (MPI is not stream-aware): gpuaware
 * then reduces the device value directly, staged goes through pinned memory.
 */
void comm_allreduce_sum_device(CommContext* ctx, double* d_value, cudaStream_t stream);

#endif  // COMM_BACKEND_H
