/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/pipe/PipeKernel.cuh>
#include <faiss/pipe/PipeKernelImpl.cuh>
#include <faiss/gpu/PipeGpuResources.h>
#include <faiss/Index.h>

namespace faiss {
namespace gpu {

constexpr uint32_t kMaxUInt32 = std::numeric_limits<int32_t>::max();

// Second-pass kernel to further k-select the results from the first pass across
// IVF lists and produce the final results
template <int ThreadsPerBlock, int NumWarpQ, int NumThreadQ>
__global__ void KernelReduce(
        int maxcluster,
        PipeTensor<float, 3, true> best_distances,
        PipeTensor<int, 3, true> best_indices,
        PipeTensor<int, 2, true> query_bcluster_matrix,
        int k,
        void** listIndices,
        IndicesOptions opt,
        bool dir,
        PipeTensor<float, 2, true> out_distances,
        PipeTensor<int, 2, true> out_indices,
        int split) {
    int queryId = blockIdx.x;

    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ float smemK[kNumWarps * NumWarpQ];
    __shared__ uint32_t smemV[kNumWarps * NumWarpQ];

    // To avoid creating excessive specializations, we combine direction
    // kernels, selecting for the smallest element. If `dir` is true, we negate
    // all values being selected (so that we are selecting the largest element).
    BlockSelect<
            float,
            uint32_t,
            false,
            Comparator<float>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(kFloatMax, kMaxUInt32, smemK, smemV, k);

    // nprobe x k
    int num = maxcluster * k * split;

    float* distanceBase = best_distances(queryId).data();
    int limit = utils::roundDown(num, kWarpSize);

    // This will keep our negation factor
    float adj = dir ? -1 : 1;

    int i = threadIdx.x;
    for (; i < limit; i += blockDim.x) {
        // We represent the index as (probe id)(k)
        // Right now, both are limited to a maximum of 2048, but we will
        // dedicate each to the high and low words of a uint32_t
        static_assert(GPU_MAX_SELECTION_K <= 65536, "");

        uint32_t curProbe = i / (k * split);
        uint32_t curK = i % (k * split);
        uint32_t index = (curProbe << 16) | (curK & (uint32_t)0xffff);

        int listId = query_bcluster_matrix(queryId)[curProbe];
        if (listId != -1) {
            // Adjust the value we are selecting based on the sorting order
            heap.addThreadQ(distanceBase[i] * adj, index);
        }

        heap.checkThreadQ();
    }

    // Handle warp divergence separately
    if (i < num) {
        uint32_t curProbe = i / (k * split);
        uint32_t curK = i % (k * split);
        uint32_t index = (curProbe << 16) | (curK & (uint32_t)0xffff);

        int listId = query_bcluster_matrix(queryId)[curProbe];
        if (listId != -1) {
            heap.addThreadQ(distanceBase[i] * adj, index);
        }
    }

    // Merge all final results
    heap.reduce();

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        // Re-adjust the value we are selecting based on the sorting order
        out_distances(queryId)[i] = smemK[i] * adj;
        auto packedIndex = smemV[i];

        // We need to remap to the user-provided indices
        Index::idx_t index = -1;

        // We may not have at least k values to return; in this function, max
        // uint32 is our sentinel value
        if (packedIndex != kMaxUInt32) {
            uint32_t curProbe = packedIndex >> 16;
            uint32_t curK = packedIndex & 0xffff;

            int listId = query_bcluster_matrix(queryId)[curProbe];
            int listOffset = best_indices(queryId)[curProbe][curK];

            if (opt == INDICES_32_BIT) {
                index = (Index::idx_t)((int*)listIndices[listId])[listOffset];
            } else if (opt == INDICES_64_BIT) {
                index = ((Index::idx_t*)listIndices[listId])[listOffset];
            } else {
                index = ((Index::idx_t)listId << 32 | (Index::idx_t)listOffset);
            }
        }

        out_indices(queryId)[i] = index;
    }
}


// Final-pass kernel
template <int ThreadsPerBlock, int NumWarpQ, int NumThreadQ>
__global__ void KernelMerge(
        PipeTensor<int, 1, true> cnt_per_query,
        PipeTensor<float*, 2, true> best_distances,
        PipeTensor<int*, 2, true> best_indices,
        int k,
        IndicesOptions opt,
        bool dir,
        PipeTensor<float, 2, true> out_distances,
        PipeTensor<int, 2, true> out_indices) {
    int queryId = blockIdx.x;

    constexpr int kNumWarps = ThreadsPerBlock / kWarpSize;

    __shared__ float smemK[kNumWarps * NumWarpQ];
    __shared__ int smemV[kNumWarps * NumWarpQ];

    // To avoid creating excessive specializations, we combine direction
    // kernels, selecting for the smallest element. If `dir` is true, we negate
    // all values being selected (so that we are selecting the largest element).
    BlockSelect<
            float,
            int,
            false,
            Comparator<float>,
            NumWarpQ,
            NumThreadQ,
            ThreadsPerBlock>
            heap(kFloatMax, kMaxUInt32, smemK, smemV, k);

    // nprobe x k
    int num = cnt_per_query(queryId) * k;

    float** distanceBase = best_distances(queryId).data();
    int** indicesBase = best_indices(queryId).data();
    int limit = utils::roundDown(num, kWarpSize);

    // This will keep our negation factor
    float adj = dir ? -1 : 1;

    int i = threadIdx.x;
    for (; i < limit; i += blockDim.x) {
        static_assert(GPU_MAX_SELECTION_K <= 65536, "");

        int curResult = i / k;
        int curK = i % k;

        int index = indicesBase[curResult][curK];

        heap.addThreadQ(distanceBase[curResult][curK] * adj, index);

        heap.checkThreadQ();
    }

    // Handle warp divergence separately
    if (i < num) {
        int curResult = i / k;
        int curK = i % k;
        int index = indicesBase[curResult][curK];
        heap.addThreadQ(distanceBase[curResult][curK] * adj, index);
    }

    // Merge all final results
    heap.reduce();

    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        // Re-adjust the value we are selecting based on the sorting order
        out_distances(queryId)[i] = smemK[i] * adj;
        out_indices(queryId)[i] = smemV[i];
    }
}







void runKernelReduce(
        int maxcluster,
        PipeTensor<float, 3, true> best_distances,
        PipeTensor<int, 3, true> best_indices,
        PipeTensor<int, 2, true> query_bcluster_matrix,
        int k,
        void** listIndices,
        IndicesOptions indicesOptions,
        bool dir,
        PipeTensor<float, 2, true> out_distances,
        PipeTensor<int, 2, true> out_indices,
        cudaStream_t stream,
        int split) {
#define IVF_REDUCE(THREADS, NUM_WARP_Q, NUM_THREAD_Q)               \
    KernelReduce<THREADS, NUM_WARP_Q, NUM_THREAD_Q>                 \
            <<<best_distances.getSize(0), THREADS, 0, stream>>>(    \
                    maxcluster,                                     \
                    best_distances,                                 \
                    best_indices,                                   \
                    query_bcluster_matrix,                          \
                    k,                                              \
                    listIndices,                                    \
                    indicesOptions,                                 \
                    dir,                                            \
                    out_distances,                                  \
                    out_indices,                                    \
                    split)

    if (k == 1) {
        IVF_REDUCE(128, 1, 1);
    } else if (k <= 32) {
        IVF_REDUCE(128, 32, 2);
    } else if (k <= 64) {
        IVF_REDUCE(128, 64, 3);
    } else if (k <= 128) {
        IVF_REDUCE(128, 128, 3);
    } else if (k <= 256) {
        IVF_REDUCE(128, 256, 4);
    } else if (k <= 512) {
        IVF_REDUCE(128, 512, 8);
    } else if (k <= 1024) {
        IVF_REDUCE(128, 1024, 8);
    }
#if GPU_MAX_SELECTION_K >= 2048
    else if (k <= 2048) {
        IVF_REDUCE(64, 2048, 8);
    }
#endif
}

void runKernelCompute(
    int d,
    int k,
    int nquery,
    int maxcluster_per_query,
    PipeTensor<int, 1, true> queryids,
    PipeTensor<float, 2, true> queries,
    PipeTensor<int, 2, true> query_cluster_matrix,
    PipeTensor<int, 3, true> best_indices,
    PipeTensor<float, 3, true> best_distances,
    void** deviceListDataPointers_,
    IndicesOptions indicesOptions,
    int* deviceListLengths_,
    faiss::MetricType metric,
    cudaStream_t stream,
    int split)
    {
    // caught for exceptions at a higher level
    FAISS_ASSERT(k <= GPU_MAX_SELECTION_K);

    if (k == 1) {
        KERNEL_COMPUTE_CALL(1);
    } else if (k <= 32) {
        KERNEL_COMPUTE_CALL(32);
    } else if (k <= 64) {
        KERNEL_COMPUTE_CALL(64);
    } else if (k <= 128) {
        KERNEL_COMPUTE_CALL(128);
    } else if (k <= 256) {
        KERNEL_COMPUTE_CALL(256);
    } else if (k <= 512) {
        KERNEL_COMPUTE_CALL(512);
    } else if (k <= 1024) {
        KERNEL_COMPUTE_CALL(1024);
    }
#if GPU_MAX_SELECTION_K >= 2048
    else if (k <= 2048) {
        KERNEL_COMPUTE_CALL(2048);
    }
#endif
}


void runKernelComputeReduce(
    int d,
    int k,
    int nquery,
    int maxcluster_per_query,
    PipeTensor<int, 1, true> queryids,
    PipeTensor<float, 2, true> queries,
    PipeTensor<int, 2, true> query_cluster_matrix,
    void** deviceListDataPointers_,
    IndicesOptions indicesOptions,
    int* deviceListLengths_,
    void** listIndices,
    faiss::MetricType metric,
    bool dir,
    PipeTensor<float, 2, true> out_distances,
    PipeTensor<int, 2, true> out_indices,
    PipeCluster* pc,
    PipeGpuResources* pipe_res,
    int device,
    int split)
{

    auto exe_stream = pipe_res->getExecuteStream(device);
    DeviceScope scope(device);

    bool goodsplit = false;
    for(int i = 1; i <= 256; i *= 2) {
        if(split == i) {
            goodsplit = true;
            break;
        }
    }
    FAISS_ASSERT(goodsplit == true);

    PipeTensor<float, 3, true> best_distances({nquery, maxcluster_per_query, (int)k * split}, pc);
    best_distances.setResources(pc, pipe_res);
    best_distances.reserve();
    PipeTensor<int, 3, true> best_indices({nquery, maxcluster_per_query, (int)k * split}, pc);
    best_indices.setResources(pc, pipe_res);
    best_indices.reserve();

    runKernelCompute(d,
                     k,
                     nquery,
                     maxcluster_per_query,
                     queryids,
                     queries,
                     query_cluster_matrix,
                     best_indices,
                     best_distances,
                     deviceListDataPointers_,
                     indicesOptions,
                     deviceListLengths_,
                     metric,
                     exe_stream,
                     split);


                    

                    //reduce() test
                
    runKernelReduce(maxcluster_per_query,
                    best_distances,
                    best_indices,
                    query_cluster_matrix,
                    k,
                    listIndices,
                    indicesOptions,
                    dir,
                    out_distances,
                    out_indices,
                    exe_stream,
                    split);



}




void runKernelMerge(
    PipeTensor<int, 1, true> cnt_per_query,
    PipeTensor<float*, 2, true> result_distances,
    PipeTensor<int*, 2, true> result_indices,
    int k,
    IndicesOptions indicesOptions,
    bool dir,
    PipeTensor<float, 2, true> out_distances,
    PipeTensor<int, 2, true> out_indices,
    cudaStream_t stream)
    {

#define IVF_MERGE(THREADS, NUM_WARP_Q, NUM_THREAD_Q)                \
    KernelMerge<THREADS, NUM_WARP_Q, NUM_THREAD_Q>                  \
            <<<result_distances.getSize(0), THREADS, 0, stream>>>(  \
                    cnt_per_query,                                  \
                    result_distances,                               \
                    result_indices,                                 \
                    k,                                              \
                    indicesOptions,                                 \
                    dir,                                            \
                    out_distances,                                  \
                    out_indices)


    if (k == 1) {
        IVF_MERGE(128, 1, 1);
    } else if (k <= 32) {
        IVF_MERGE(128, 32, 2);
    } else if (k <= 64) {
        IVF_MERGE(128, 64, 3);
    } else if (k <= 128) {
        IVF_MERGE(128, 128, 3);
    } else if (k <= 256) {
        IVF_MERGE(128, 256, 4);
    } else if (k <= 512) {
        IVF_MERGE(128, 512, 8);
    } else if (k <= 1024) {
        IVF_MERGE(128, 1024, 8);
    }
#if GPU_MAX_SELECTION_K >= 2048
    else if (k <= 2048) {
        IVF_MERGE(64, 2048, 8);
    }
#endif
    }





} // namespace gpu
} // namespace faiss


