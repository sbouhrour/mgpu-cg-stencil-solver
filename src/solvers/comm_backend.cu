/**
 * @file comm_backend.cu
 * @brief Halo exchange and scalar reductions for the Z-slab CG solvers
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
    #include <mpi-ext.h>
#endif
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#ifdef HAS_NCCL
    #include <nccl.h>
#endif
#ifdef HAS_NVSHMEM
    #include <stdint.h>
    #include <nvshmem.h>
    #include <nvshmemx.h>
#endif

#include "spmv.h"
#include "solvers/comm_backend.h"

struct CommContext {
    CommBackendKind kind;
    MPI_Comm comm;
    int rank;
    int world_size;
    size_t max_halo_elems;

    // staged only: pinned host mirrors of the four halo planes
    double* h_send_prev;
    double* h_send_next;
    double* h_recv_prev;
    double* h_recv_next;
    double* h_scalar;

#ifdef HAS_NCCL
    ncclComm_t nccl;
#endif
#ifdef HAS_NVSHMEM
    // Symmetric memory, same addresses on every PE:
    //   land: [0, max) receives from rank-1, [max, 2 max) receives from rank+1
    //   sig:  [0] data from rank-1 landed, [1] data from rank+1 landed,
    //         [2] rank-1 consumed what I sent it, [3] rank+1 consumed what I sent it
    //   red:  [0] reduction input, [1] reduction output
    double* land;
    uint64_t* sig;
    double* red;
    uint64_t seq;  // exchanges so far; identical on every PE
#endif

    // Exchange in flight between comm_halo_begin and comm_halo_end
    struct {
        const double* send_prev;
        const double* send_next;
        double* recv_prev;
        double* recv_next;
        int elems;
        cudaStream_t stream;
        MPI_Request requests[4];
        int req_count;
    } pending;

    char halo_label[32];
    char allreduce_label[32];
    char allreduce_device_label[32];
};

int comm_backend_parse(const char* name, CommBackendKind* kind) {
    if (strcmp(name, "staged") == 0) {
        *kind = COMM_STAGED;
    } else if (strcmp(name, "gpuaware") == 0) {
        *kind = COMM_GPUAWARE;
    } else if (strcmp(name, "nccl") == 0) {
        *kind = COMM_NCCL;
    } else if (strcmp(name, "nvshmem") == 0) {
        *kind = COMM_NVSHMEM;
    } else {
        return 1;
    }
    return 0;
}

const char* comm_backend_name(CommBackendKind kind) {
    switch (kind) {
        case COMM_STAGED:
            return "staged";
        case COMM_GPUAWARE:
            return "gpuaware";
        case COMM_NCCL:
            return "nccl";
        case COMM_NVSHMEM:
            return "nvshmem";
    }
    return "unknown";
}

CommBackendKind comm_kind(const CommContext* ctx) {
    return ctx->kind;
}

/**
 * @brief Whether MPI accepts device pointers: 1 yes, 0 no, -1 cannot tell
 *
 * Passing a device pointer to an MPI built without CUDA support does not fail
 * cleanly: the library dereferences it on the host and the rank segfaults.
 */
static int mpi_cuda_support(void) {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    return MPIX_Query_cuda_support();
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
    return 0;
#else
    return -1;
#endif
}

#ifdef HAS_NCCL
static void halo_nccl(CommContext* ctx, const double* d_send_prev, const double* d_send_next,
                      double* d_recv_prev, double* d_recv_next, int halo_elems,
                      cudaStream_t stream);

/**
 * @brief Create the NCCL communicator and connect every neighbour pair
 *
 * NCCL connects a peer on the first operation that targets it (buffer
 * allocation, memory mapping, proxy setup). One exchange at full halo size
 * here keeps that cost out of every timed solve.
 */
static int nccl_init(CommContext* ctx) {
    ncclUniqueId id;
    if (ctx->rank == 0)
        ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, ctx->comm);

    ncclResult_t res = ncclCommInitRank(&ctx->nccl, ctx->world_size, id, ctx->rank);
    if (res != ncclSuccess) {
        fprintf(stderr, "[Rank %d] ncclCommInitRank: %s (%s)\n", ctx->rank, ncclGetErrorString(res),
                ncclGetLastError(NULL));
        return 1;
    }

    size_t bytes = ctx->max_halo_elems * sizeof(double);
    double *d_send, *d_recv_prev, *d_recv_next;
    CUDA_CHECK(cudaMalloc(&d_send, bytes));
    CUDA_CHECK(cudaMalloc(&d_recv_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_recv_next, bytes));
    CUDA_CHECK(cudaMemset(d_send, 0, bytes));
    halo_nccl(ctx, d_send, d_send, d_recv_prev, d_recv_next, (int)ctx->max_halo_elems, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
    cudaFree(d_send);
    cudaFree(d_recv_prev);
    cudaFree(d_recv_next);
    return 0;
}
#endif

#ifdef HAS_NVSHMEM
static int nvshmem_setup(CommContext* ctx) {
    nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;
    if (ctx->rank == 0)
        nvshmemx_get_uniqueid(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, ctx->comm);
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_uniqueid_args(ctx->rank, ctx->world_size, &id, &attr);
    if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0) {
        fprintf(stderr, "[Rank %d] nvshmemx_init_attr failed\n", ctx->rank);
        return 1;
    }
    // Collective allocations: same size on every PE
    ctx->land = (double*)nvshmem_malloc(2 * ctx->max_halo_elems * sizeof(double));
    ctx->sig = (uint64_t*)nvshmem_calloc(4, sizeof(uint64_t));
    ctx->red = (double*)nvshmem_malloc(2 * sizeof(double));
    ctx->seq = 0;
    if (!ctx->land || !ctx->sig || !ctx->red) {
        fprintf(stderr, "[Rank %d] nvshmem_malloc failed\n", ctx->rank);
        return 1;
    }
    return 0;
}

/**
 * @brief Halo exchange by one-sided puts, entirely ordered on stream
 *
 * Each plane is put straight into the neighbour's landing buffer together with a
 * signal; the receiver waits for the signal, copies the plane into its halo and
 * acknowledges. A sender waits for the acknowledgement of its previous exchange
 * before putting again: without it, a rank that runs ahead would overwrite a
 * landing buffer its neighbour has not read yet (two exchanges can follow each
 * other with no reduction in between, as x and r do before the first iteration).
 */
static void halo_nvshmem(CommContext* ctx, const double* d_send_prev, const double* d_send_next,
                         double* d_recv_prev, double* d_recv_next, int halo_elems,
                         cudaStream_t stream) {
    const int has_prev = ctx->rank > 0;
    const int has_next = ctx->rank < ctx->world_size - 1;
    const size_t bytes = (size_t)halo_elems * sizeof(double);
    const uint64_t seq = ++ctx->seq;
    double* from_prev = ctx->land;
    double* from_next = ctx->land + ctx->max_halo_elems;

    if (has_next) {  // my last plane lands in rank+1's from_prev slot
        nvshmemx_signal_wait_until_on_stream(&ctx->sig[3], NVSHMEM_CMP_GE, seq - 1, stream);
        nvshmemx_putmem_signal_on_stream(from_prev, d_send_next, bytes, &ctx->sig[0], seq,
                                         NVSHMEM_SIGNAL_SET, ctx->rank + 1, stream);
    }
    if (has_prev) {  // my first plane lands in rank-1's from_next slot
        nvshmemx_signal_wait_until_on_stream(&ctx->sig[2], NVSHMEM_CMP_GE, seq - 1, stream);
        nvshmemx_putmem_signal_on_stream(from_next, d_send_prev, bytes, &ctx->sig[1], seq,
                                         NVSHMEM_SIGNAL_SET, ctx->rank - 1, stream);
    }
    if (has_prev) {
        nvshmemx_signal_wait_until_on_stream(&ctx->sig[0], NVSHMEM_CMP_GE, seq, stream);
        CUDA_CHECK(
            cudaMemcpyAsync(d_recv_prev, from_prev, bytes, cudaMemcpyDeviceToDevice, stream));
        nvshmemx_signal_op_on_stream(&ctx->sig[3], seq, NVSHMEM_SIGNAL_SET, ctx->rank - 1, stream);
    }
    if (has_next) {
        nvshmemx_signal_wait_until_on_stream(&ctx->sig[1], NVSHMEM_CMP_GE, seq, stream);
        CUDA_CHECK(
            cudaMemcpyAsync(d_recv_next, from_next, bytes, cudaMemcpyDeviceToDevice, stream));
        nvshmemx_signal_op_on_stream(&ctx->sig[2], seq, NVSHMEM_SIGNAL_SET, ctx->rank + 1, stream);
    }
}
#endif

CommContext* comm_create(CommBackendKind kind, MPI_Comm mpi_comm, size_t max_halo_elems) {
    CommContext* ctx = (CommContext*)calloc(1, sizeof(CommContext));
    ctx->kind = kind;
    ctx->comm = mpi_comm;
    ctx->max_halo_elems = max_halo_elems;
    MPI_Comm_rank(mpi_comm, &ctx->rank);
    MPI_Comm_size(mpi_comm, &ctx->world_size);
    snprintf(ctx->halo_label, sizeof(ctx->halo_label), "Halo_%s", comm_backend_name(kind));
    // Host scalars are reduced with MPI whatever the halo backend; device
    // scalars (comm_allreduce_sum_device) use the backend itself.
    snprintf(ctx->allreduce_label, sizeof(ctx->allreduce_label), "Allreduce_mpi_host");
    snprintf(ctx->allreduce_device_label, sizeof(ctx->allreduce_device_label), "Allreduce_%s",
             comm_backend_name(kind));

    if (kind == COMM_GPUAWARE) {
        int support = mpi_cuda_support();
        if (support == 0) {
            if (ctx->rank == 0)
                fprintf(stderr, "Error: --comm=gpuaware needs a CUDA-aware MPI; this one is not\n");
            free(ctx);
            return NULL;
        }
        if (support < 0 && ctx->rank == 0)
            printf("Warning: cannot query CUDA support of this MPI, assuming CUDA-aware\n");
    }

    if (kind == COMM_NCCL) {
#ifdef HAS_NCCL
        // Reported because this setup, once charged to the solve, is what a timed NCCL run
        // must never contain: it is paid here, once, before any timed region.
        double t0 = MPI_Wtime();
        if (nccl_init(ctx) != 0) {
            free(ctx);
            return NULL;
        }
        double setup_ms = (MPI_Wtime() - t0) * 1e3, max_ms;
        MPI_Reduce(&setup_ms, &max_ms, 1, MPI_DOUBLE, MPI_MAX, 0, mpi_comm);
        if (ctx->rank == 0)
            printf("NCCL setup (communicator + first exchange with each neighbour): %.1f ms\n",
                   max_ms);
#else
        if (ctx->rank == 0)
            fprintf(stderr,
                    "Error: --comm=nccl requested but this binary was built without NCCL\n");
        free(ctx);
        return NULL;
#endif
    }

    if (kind == COMM_NVSHMEM) {
#ifdef HAS_NVSHMEM
        if (nvshmem_setup(ctx) != 0) {
            free(ctx);
            return NULL;
        }
#else
        if (ctx->rank == 0)
            fprintf(stderr,
                    "Error: --comm=nvshmem requested but this binary was built without NVSHMEM\n");
        free(ctx);
        return NULL;
#endif
    }

    if (kind == COMM_STAGED) {
        size_t bytes = max_halo_elems * sizeof(double);
        CUDA_CHECK(cudaMallocHost(&ctx->h_send_prev, bytes));
        CUDA_CHECK(cudaMallocHost(&ctx->h_send_next, bytes));
        CUDA_CHECK(cudaMallocHost(&ctx->h_recv_prev, bytes));
        CUDA_CHECK(cudaMallocHost(&ctx->h_recv_next, bytes));
        CUDA_CHECK(cudaMallocHost(&ctx->h_scalar, sizeof(double)));
    }
    return ctx;
}

void comm_destroy(CommContext* ctx) {
    if (ctx == NULL)
        return;
    if (ctx->kind == COMM_STAGED) {
        cudaFreeHost(ctx->h_send_prev);
        cudaFreeHost(ctx->h_send_next);
        cudaFreeHost(ctx->h_recv_prev);
        cudaFreeHost(ctx->h_recv_next);
        cudaFreeHost(ctx->h_scalar);
    }
#ifdef HAS_NCCL
    if (ctx->kind == COMM_NCCL)
        ncclCommDestroy(ctx->nccl);
#endif
#ifdef HAS_NVSHMEM
    if (ctx->kind == COMM_NVSHMEM) {
        nvshmem_free(ctx->land);
        nvshmem_free(ctx->sig);
        nvshmem_free(ctx->red);
        nvshmem_finalize();
    }
#endif
    free(ctx);
}

static void halo_staged(CommContext* ctx, const double* d_send_prev, const double* d_send_next,
                        double* d_recv_prev, double* d_recv_next, int halo_elems,
                        cudaStream_t stream) {
    const int has_prev = ctx->rank > 0;
    const int has_next = ctx->rank < ctx->world_size - 1;
    const size_t bytes = (size_t)halo_elems * sizeof(double);
    MPI_Request requests[4];
    int req_count = 0;

    if (has_prev)
        CUDA_CHECK(
            cudaMemcpyAsync(ctx->h_send_prev, d_send_prev, bytes, cudaMemcpyDeviceToHost, stream));
    if (has_next)
        CUDA_CHECK(
            cudaMemcpyAsync(ctx->h_send_next, d_send_next, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (has_prev) {
        MPI_Isend(ctx->h_send_prev, halo_elems, MPI_DOUBLE, ctx->rank - 1, 0, ctx->comm,
                  &requests[req_count++]);
        MPI_Irecv(ctx->h_recv_prev, halo_elems, MPI_DOUBLE, ctx->rank - 1, 0, ctx->comm,
                  &requests[req_count++]);
    }
    if (has_next) {
        MPI_Isend(ctx->h_send_next, halo_elems, MPI_DOUBLE, ctx->rank + 1, 0, ctx->comm,
                  &requests[req_count++]);
        MPI_Irecv(ctx->h_recv_next, halo_elems, MPI_DOUBLE, ctx->rank + 1, 0, ctx->comm,
                  &requests[req_count++]);
    }
    if (req_count > 0)
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    if (has_prev)
        CUDA_CHECK(
            cudaMemcpyAsync(d_recv_prev, ctx->h_recv_prev, bytes, cudaMemcpyHostToDevice, stream));
    if (has_next)
        CUDA_CHECK(
            cudaMemcpyAsync(d_recv_next, ctx->h_recv_next, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void halo_gpuaware(CommContext* ctx, const double* d_send_prev, const double* d_send_next,
                          double* d_recv_prev, double* d_recv_next, int halo_elems,
                          cudaStream_t stream) {
    const int has_prev = ctx->rank > 0;
    const int has_next = ctx->rank < ctx->world_size - 1;
    MPI_Request requests[4];
    int req_count = 0;

    // MPI is not stream-aware: the kernel that produced the send planes must
    // have finished before the library reads them.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (has_prev) {
        MPI_Isend(d_send_prev, halo_elems, MPI_DOUBLE, ctx->rank - 1, 0, ctx->comm,
                  &requests[req_count++]);
        MPI_Irecv(d_recv_prev, halo_elems, MPI_DOUBLE, ctx->rank - 1, 0, ctx->comm,
                  &requests[req_count++]);
    }
    if (has_next) {
        MPI_Isend(d_send_next, halo_elems, MPI_DOUBLE, ctx->rank + 1, 0, ctx->comm,
                  &requests[req_count++]);
        MPI_Irecv(d_recv_next, halo_elems, MPI_DOUBLE, ctx->rank + 1, 0, ctx->comm,
                  &requests[req_count++]);
    }
    // A completed receive has landed in device memory: no copy, no second sync.
    if (req_count > 0)
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}

#ifdef HAS_NCCL
    #define NCCL_CHECK(call)                                                                  \
        do {                                                                                  \
            ncclResult_t r_ = (call);                                                         \
            if (r_ != ncclSuccess) {                                                          \
                fprintf(stderr, "NCCL error %s at %s:%d\n", ncclGetErrorString(r_), __FILE__, \
                        __LINE__);                                                            \
                MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
            }                                                                                 \
        } while (0)

static void halo_nccl(CommContext* ctx, const double* d_send_prev, const double* d_send_next,
                      double* d_recv_prev, double* d_recv_next, int halo_elems,
                      cudaStream_t stream) {
    const int has_prev = ctx->rank > 0;
    const int has_next = ctx->rank < ctx->world_size - 1;

    // Enqueued behind the kernel that wrote the send planes: no host
    // synchronization, the stream orders it. The group lets the sends and
    // receives to both neighbours progress together instead of deadlocking.
    NCCL_CHECK(ncclGroupStart());
    if (has_prev) {
        NCCL_CHECK(ncclSend(d_send_prev, halo_elems, ncclDouble, ctx->rank - 1, ctx->nccl, stream));
        NCCL_CHECK(ncclRecv(d_recv_prev, halo_elems, ncclDouble, ctx->rank - 1, ctx->nccl, stream));
    }
    if (has_next) {
        NCCL_CHECK(ncclSend(d_send_next, halo_elems, ncclDouble, ctx->rank + 1, ctx->nccl, stream));
        NCCL_CHECK(ncclRecv(d_recv_next, halo_elems, ncclDouble, ctx->rank + 1, ctx->nccl, stream));
    }
    NCCL_CHECK(ncclGroupEnd());
}
#endif

void comm_halo_exchange(CommContext* ctx, const double* d_send_prev, const double* d_send_next,
                        double* d_recv_prev, double* d_recv_next, int halo_elems,
                        cudaStream_t stream) {
    if ((size_t)halo_elems > ctx->max_halo_elems) {
        fprintf(stderr, "comm_halo_exchange: %d elements exceeds context capacity %zu\n",
                halo_elems, ctx->max_halo_elems);
        MPI_Abort(ctx->comm, 1);
    }
    nvtxRangePushA(ctx->halo_label);
    switch (ctx->kind) {
        case COMM_STAGED:
            halo_staged(ctx, d_send_prev, d_send_next, d_recv_prev, d_recv_next, halo_elems,
                        stream);
            break;
        case COMM_GPUAWARE:
            halo_gpuaware(ctx, d_send_prev, d_send_next, d_recv_prev, d_recv_next, halo_elems,
                          stream);
            break;
        case COMM_NCCL:
#ifdef HAS_NCCL
            halo_nccl(ctx, d_send_prev, d_send_next, d_recv_prev, d_recv_next, halo_elems, stream);
#endif
            break;
        case COMM_NVSHMEM:
#ifdef HAS_NVSHMEM
            halo_nvshmem(ctx, d_send_prev, d_send_next, d_recv_prev, d_recv_next, halo_elems,
                         stream);
#endif
            break;
    }
    nvtxRangePop();
}

void comm_halo_begin(CommContext* ctx, const double* d_send_prev, const double* d_send_next,
                     double* d_recv_prev, double* d_recv_next, int halo_elems,
                     cudaStream_t stream) {
    if ((size_t)halo_elems > ctx->max_halo_elems) {
        fprintf(stderr, "comm_halo_begin: %d elements exceeds context capacity %zu\n", halo_elems,
                ctx->max_halo_elems);
        MPI_Abort(ctx->comm, 1);
    }
    ctx->pending.send_prev = d_send_prev;
    ctx->pending.send_next = d_send_next;
    ctx->pending.recv_prev = d_recv_prev;
    ctx->pending.recv_next = d_recv_next;
    ctx->pending.elems = halo_elems;
    ctx->pending.stream = stream;
    ctx->pending.req_count = 0;
    nvtxRangePushA(ctx->halo_label);
    if (ctx->kind == COMM_STAGED) {
        const size_t bytes = (size_t)halo_elems * sizeof(double);
        if (ctx->rank > 0)
            CUDA_CHECK(cudaMemcpyAsync(ctx->h_send_prev, d_send_prev, bytes, cudaMemcpyDeviceToHost,
                                       stream));
        if (ctx->rank < ctx->world_size - 1)
            CUDA_CHECK(cudaMemcpyAsync(ctx->h_send_next, d_send_next, bytes, cudaMemcpyDeviceToHost,
                                       stream));
    }
}

void comm_halo_post(CommContext* ctx) {
    const int has_prev = ctx->rank > 0;
    const int has_next = ctx->rank < ctx->world_size - 1;
    const int n = ctx->pending.elems;
    MPI_Request* rq = ctx->pending.requests;
    int* rc = &ctx->pending.req_count;
    switch (ctx->kind) {
        case COMM_STAGED:
            CUDA_CHECK(cudaStreamSynchronize(ctx->pending.stream));  // D2H landed
            if (has_prev) {
                MPI_Isend(ctx->h_send_prev, n, MPI_DOUBLE, ctx->rank - 1, 0, ctx->comm,
                          &rq[(*rc)++]);
                MPI_Irecv(ctx->h_recv_prev, n, MPI_DOUBLE, ctx->rank - 1, 0, ctx->comm,
                          &rq[(*rc)++]);
            }
            if (has_next) {
                MPI_Isend(ctx->h_send_next, n, MPI_DOUBLE, ctx->rank + 1, 0, ctx->comm,
                          &rq[(*rc)++]);
                MPI_Irecv(ctx->h_recv_next, n, MPI_DOUBLE, ctx->rank + 1, 0, ctx->comm,
                          &rq[(*rc)++]);
            }
            break;
        case COMM_GPUAWARE:
            CUDA_CHECK(cudaStreamSynchronize(ctx->pending.stream));  // send planes written
            if (has_prev) {
                MPI_Isend(ctx->pending.send_prev, n, MPI_DOUBLE, ctx->rank - 1, 0, ctx->comm,
                          &rq[(*rc)++]);
                MPI_Irecv(ctx->pending.recv_prev, n, MPI_DOUBLE, ctx->rank - 1, 0, ctx->comm,
                          &rq[(*rc)++]);
            }
            if (has_next) {
                MPI_Isend(ctx->pending.send_next, n, MPI_DOUBLE, ctx->rank + 1, 0, ctx->comm,
                          &rq[(*rc)++]);
                MPI_Irecv(ctx->pending.recv_next, n, MPI_DOUBLE, ctx->rank + 1, 0, ctx->comm,
                          &rq[(*rc)++]);
            }
            break;
        case COMM_NCCL:
#ifdef HAS_NCCL
            halo_nccl(ctx, ctx->pending.send_prev, ctx->pending.send_next, ctx->pending.recv_prev,
                      ctx->pending.recv_next, n, ctx->pending.stream);
#endif
            break;
        case COMM_NVSHMEM:
#ifdef HAS_NVSHMEM
            halo_nvshmem(ctx, ctx->pending.send_prev, ctx->pending.send_next,
                         ctx->pending.recv_prev, ctx->pending.recv_next, n, ctx->pending.stream);
#endif
            break;
    }
}

void comm_halo_end(CommContext* ctx) {
    if (ctx->pending.req_count > 0)
        MPI_Waitall(ctx->pending.req_count, ctx->pending.requests, MPI_STATUSES_IGNORE);
    if (ctx->kind == COMM_STAGED) {
        const size_t bytes = (size_t)ctx->pending.elems * sizeof(double);
        if (ctx->rank > 0)
            CUDA_CHECK(cudaMemcpyAsync(ctx->pending.recv_prev, ctx->h_recv_prev, bytes,
                                       cudaMemcpyHostToDevice, ctx->pending.stream));
        if (ctx->rank < ctx->world_size - 1)
            CUDA_CHECK(cudaMemcpyAsync(ctx->pending.recv_next, ctx->h_recv_next, bytes,
                                       cudaMemcpyHostToDevice, ctx->pending.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx->pending.stream));
    }
    ctx->pending.req_count = 0;
    nvtxRangePop();
}

double comm_allreduce_sum(CommContext* ctx, double local) {
    double global;
    nvtxRangePushA(ctx->allreduce_label);
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, ctx->comm);
    nvtxRangePop();
    return global;
}

void comm_allreduce_sum_device(CommContext* ctx, double* d_value, cudaStream_t stream) {
    nvtxRangePushA(ctx->allreduce_device_label);
    switch (ctx->kind) {
        case COMM_STAGED:
            CUDA_CHECK(cudaMemcpyAsync(ctx->h_scalar, d_value, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Allreduce(MPI_IN_PLACE, ctx->h_scalar, 1, MPI_DOUBLE, MPI_SUM, ctx->comm);
            // Ordered before any later kernel on stream; the pinned buffer is
            // only rewritten after the next synchronization.
            CUDA_CHECK(cudaMemcpyAsync(d_value, ctx->h_scalar, sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
            break;
        case COMM_GPUAWARE:
            CUDA_CHECK(cudaStreamSynchronize(stream));
            MPI_Allreduce(MPI_IN_PLACE, d_value, 1, MPI_DOUBLE, MPI_SUM, ctx->comm);
            break;
        case COMM_NCCL:
#ifdef HAS_NCCL
            NCCL_CHECK(ncclAllReduce(d_value, d_value, 1, ncclDouble, ncclSum, ctx->nccl, stream));
#endif
            break;
        case COMM_NVSHMEM:
#ifdef HAS_NVSHMEM
            // The reduction reads and writes symmetric memory only
            CUDA_CHECK(cudaMemcpyAsync(&ctx->red[0], d_value, sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));
            nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, &ctx->red[1], &ctx->red[0], 1,
                                                 stream);
            CUDA_CHECK(cudaMemcpyAsync(d_value, &ctx->red[1], sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));
#endif
            break;
    }
    nvtxRangePop();
}
