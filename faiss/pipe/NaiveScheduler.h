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
#include <faiss/pipe/IndexIVFPipe.h>
#include <faiss/pipe/PipeScheduler.h>

namespace faiss{
namespace gpu{

// The class reorders the computation and divides the pipeline groups
class NaiveScheduler{
protected:
    struct pipelinegroup{
        // Group slice
        std::vector<int> content;

        // Time (ms)
        float time = 1e9;

        float delay = 1e9;

        float pre = 1e9;
    };

    struct malloc_res{
        void *pointer;

        bool valid;

        void* getPage(int id, size_t size){
            float* p = (float *)pointer;
            p += id * size;
            return (void*)p;
        }
    };

public:
    // construct function
    NaiveScheduler(IndexIVFPipe* index, PipeCluster* pc, PipeGpuResources* pgr, int bcluster_cnt_,
            int* bcluster_list_, int* query_per_bcluster_, int maxquery_per_bcluster_,
            int* bcluster_query_matrix_, PipeProfiler* profiler_,
            int queryMax_, int clusMax_, bool free_ = true);

    NaiveScheduler(IndexIVFPipe* index, PipeCluster* pc, PipeGpuResources* pgr,
            int n, float *xq, int k, float *dis, int *label, bool free_ = true);

    ~NaiveScheduler();

    // Reorder the clusters to minimize the pipeline overhead
    void reorder();

    void nonReorder();

    // Optimal query-aware Grouping
    void group();

    pipelinegroup group(int staclu, float total, float delay, int depth);

    // enter multi-threads
    void process(int num, float *xq, int k, float *dis, int *label);

    // Mesure the tran and com time for the number of clusters (ms)
    float measure_tran(int num);

    float measure_com(int sta, int end);

    std::pair<int, int> genematrix(int **queryClusMat, int **queryIds, 
        const std::vector<int> & group, int* dataCnt );

public:

    int canv;

    int grain = -1;

    // the corresponding pipecluster
    PipeCluster* pc_;

    // the corresponding PipeGpurespurce
    PipeGpuResources* pgr_;

    // The corresponding index
    IndexIVFPipe* index_;

    // if free the sample params
    bool free;

    // If the first serch
    bool cold_start;

    // The size of the current batch queries
    int batch_size = -1;

    // the total number of query to compute
    int queryMax;

    // the total number of clusters the index keeps
    int clusMax;

    // Extra varibles to delete in deconstruct function
    float* coarse_dis;
    int* ori_idx;
    int* bcluster_per_query;
    int* query_bcluster_matrix;

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

    // mutex for copy engine preemption
    pthread_mutex_t preemption_mutex;

    // cluster address on gpu
    std::unordered_map<int, void*> address;

    malloc_res gpu_malloc(size_t size);

    bool preemption = true;

    // the group number
    int num_group;

    int max_size;

    // result of group algorithm
    std::vector<int> groups;

    PipeProfiler* profiler;

    PipeTensor<float, 2, true> *queries_gpu = nullptr;

    std::vector<PipeTensor<float, 2, true>* > dis_buffer;

    std::vector<PipeTensor<int, 2, true>* > ids_buffer;

    // Group queries idx
    std::vector<int *> queries_ids;

    std::vector<int> queries_num;

    std::vector<int> cnt_per_query;

    int max_split = -1;

    // int max_quries_num = -1;

    int com_index = 0;

    double com_time = 0.;

    double com_transmission = 0.;

    double group_time = 0.;

    double reorder_time = 0.;

    bool verbose = false;

};

} // namespace gpu
}  // namespace faiss