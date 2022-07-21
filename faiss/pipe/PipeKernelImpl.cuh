/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/pipe/PipeKernel.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>


#define KERNEL_COMPUTE_IMPL(THREADS, WARP_Q, THREAD_Q)            \
                                                                    \
    void KernelComputeImpl_##WARP_Q##_(                            \
            int d,                                                  \
            int k,                                                  \
            int nquery,                                             \
            int maxcluster_per_query,                               \
            PipeTensor<int, 1, true> queryids,                          \
            PipeTensor<float, 2, true> queries,                         \
            PipeTensor<int, 2, true> query_cluster_matrix,              \
            PipeTensor<size_t, 3, true> best_indices,                   \
            PipeTensor<float, 3, true> best_distances,                  \
            void** deviceListDataPointers_,                         \
            IndicesOptions indicesOptions,                          \
            int* deviceListLengths_,                                \
            faiss::MetricType metric,                               \
            cudaStream_t stream){                                   \
        FAISS_ASSERT(k <= WARP_Q);                                  \
                                                                    \
        IVFINT_METRICS_PIPE(THREADS, WARP_Q, THREAD_Q);             \
                                                                    \
        CUDA_TEST_ERROR();                                          \
    }

#define KERNEL_COMPUTE_DECL(WARP_Q)                                 \
                                                                    \
    void KernelComputeImpl_##WARP_Q##_(                             \
            int d,                                                  \
            int k,                                                  \
            int nquery,                                       \
            int maxcluster_per_query,                              \
            PipeTensor<int, 1, true> queryids,                           \
            PipeTensor<float, 2, true> queries,                         \
            PipeTensor<int, 2, true> query_cluster_matrix,              \
            PipeTensor<size_t, 3, true> best_indices,                  \
            PipeTensor<float, 3, true> best_distances,                 \
            void** deviceListDataPointers_,                         \
            IndicesOptions indicesOptions,                          \
            int* deviceListLengths_,                                \
            faiss::MetricType metric,                               \
            cudaStream_t stream)

#define KERNEL_COMPUTE_CALL(WARP_Q)    \
    KernelComputeImpl_##WARP_Q##_( \
            d,                          \
            k,                          \
            nquery,                     \
            maxcluster_per_query,       \
            queryids,                   \
            queries,                    \
            query_cluster_matrix,       \
            best_indices,               \
            best_distances,             \
            deviceListDataPointers_,    \
            indicesOptions,             \
            deviceListLengths_,         \
            metric,                     \
            stream)



namespace faiss {
namespace gpu {

KERNEL_COMPUTE_DECL(1);
KERNEL_COMPUTE_DECL(32);
KERNEL_COMPUTE_DECL(64);
KERNEL_COMPUTE_DECL(128);
KERNEL_COMPUTE_DECL(256);
KERNEL_COMPUTE_DECL(512);
KERNEL_COMPUTE_DECL(1024);

#if GPU_MAX_SELECTION_K >= 2048
KERNEL_COMPUTE_DECL(2048);
#endif

} // namespace gpu
} // namespace faiss
