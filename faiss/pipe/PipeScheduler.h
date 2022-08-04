/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <omp.h>
#include <unordered_map>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/gpu/utils/PipeTensor.cuh>
#include <faiss/gpu/PipeGpuResources.h>
#include <faiss/pipe/PipeKernel.cuh>
#include <faiss/pipe/PipeCluster.h>
#include <faiss/pipe/PipeStructure.h>
#include <faiss/pipe/PipeProfiler.cuh>

namespace faiss{
namespace gpu{

// The class reorders the computation and divides the pipeline groups
class PipeScheduler{
protected:
    struct pipelinegroup{
        // Group slice
        std::vector<int> content;

        // Time (ms)
        float time = 1e9;

        float delay = 1e9;

        float pre = 1e9;
    };

public:
    // construct function
    PipeScheduler(PipeCluster* pc, PipeGpuResources* pgr, int bcluster_cnt_,
            int* bcluster_list_, int* query_per_bcluster_, int maxquery_per_bcluster_,
            int* bcluster_query_matrix_, PipeProfiler* profiler_,
            int queryMax_, int clusMax_, bool free_ = true);

    ~PipeScheduler();

    // Reorder the clusters to minimize the pipeline overhead
    void reorder();

    // Optimal query-aware Grouping
    void group();

    pipelinegroup group(int staclu, float total, float delay, int depth);

    // Mesure the tran and com time for the number of clusters (ms)
    float measure_tran(int num);

    float measure_com(int sta, int end);

    void compute();

public:

    int canv;

    // the corresponding pipecluster
    PipeCluster* pc_;

    // the corresponding PipeGpurespurce
    PipeGpuResources* pgr_;

    // if free the sample params
    bool free;

    // the total number of query to compute
    int queryMax;

    // the total number of clusters the index keeps
    int clusMax;

    // the number of balanced clusters concerning
    int bcluster_cnt;

    // the id of balanced clusters concerning
    int* bcluster_list;

    // the number of queries referenced by each balanced cluster
    int* query_per_bcluster;

    // the max value of query_per_bcluster
    int maxquery_per_bcluster;

    // matrix of cluster x queries, shape of [bcluster_cnt][maxquery_per_bcluster],
    // padding with -1
    int* bcluster_query_matrix;

    // reorder cluster list
    std::vector<int> reorder_list;

    // the size of the clusters already on gpu
    int part_size;

    // reverse map of bcluster_list
    std::unordered_map<int, int> reversemap;

    // the group number
    int num_group;

    int max_size;

    // result of group algorithm
    std::vector<int> groups;

    PipeProfiler* profiler;

};


void transpose(int* clusQueryMat, int** queryClusMat, int* clus, int* query, int queryMax, int clusMax, std::vector<int>& rows, int* clusIds, int** queryIds);

} // namespace gpu
} // namespace faiss