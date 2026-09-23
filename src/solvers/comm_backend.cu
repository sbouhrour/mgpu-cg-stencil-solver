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

    char halo_label[32];
    char allreduce_label[32];
};

int comm_backend_parse(const char* name, CommBackendKind* kind) {
    if (strcmp(name, "staged") == 0) {
        *kind = COMM_STAGED;
    } else if (strcmp(name, "gpuaware") == 0) {
        *kind = COMM_GPUAWARE;
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

CommContext* comm_create(CommBackendKind kind, MPI_Comm mpi_comm, size_t max_halo_elems) {
    CommContext* ctx = (CommContext*)calloc(1, sizeof(CommContext));
    ctx->kind = kind;
    ctx->comm = mpi_comm;
    ctx->max_halo_elems = max_halo_elems;
    MPI_Comm_rank(mpi_comm, &ctx->rank);
    MPI_Comm_size(mpi_comm, &ctx->world_size);
    snprintf(ctx->halo_label, sizeof(ctx->halo_label), "Halo_%s", comm_backend_name(kind));
    snprintf(ctx->allreduce_label, sizeof(ctx->allreduce_label), "Allreduce_%s",
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

    if (kind == COMM_STAGED) {
        size_t bytes = max_halo_elems * sizeof(double);
        CUDA_CHECK(cudaMallocHost(&ctx->h_send_prev, bytes));
        CUDA_CHECK(cudaMallocHost(&ctx->h_send_next, bytes));
        CUDA_CHECK(cudaMallocHost(&ctx->h_recv_prev, bytes));
        CUDA_CHECK(cudaMallocHost(&ctx->h_recv_next, bytes));
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
    }
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
    }
    nvtxRangePop();
}

double comm_allreduce_sum(CommContext* ctx, double local) {
    double global;
    nvtxRangePushA(ctx->allreduce_label);
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, ctx->comm);
    nvtxRangePop();
    return global;
}
