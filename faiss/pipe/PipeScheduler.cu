/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <sys/time.h>

#include <faiss/pipe/PipeScheduler.h>

namespace faiss {
namespace gpu{

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-3;
}

PipeScheduler::PipeScheduler(PipeCluster* pc, PipeGpuResources* pgr, int bcluster_cnt_,
        int* bcluster_list_, int* query_per_bcluster_, int maxquery_per_bcluster_,
        int* bcluster_query_matrix_, bool free_) 
        : pc_(pc), pgr_(pgr), bcluster_cnt(bcluster_cnt_),
        bcluster_list(bcluster_list_), query_per_bcluster(query_per_bcluster_), \
        maxquery_per_bcluster(maxquery_per_bcluster_), bcluster_query_matrix(bcluster_query_matrix_), free(free_), \
        num_group(1){
            reorder_list.resize(bcluster_cnt);

            reorder();

        }

PipeScheduler::~PipeScheduler(){
    // Free the input resource
    if(free){
        delete[] bcluster_list;
        delete[] query_per_bcluster;
        delete[] bcluster_query_matrix;
    }

    // Free the PipeTensor in order
}

void PipeScheduler::reorder(){
    double t0 , t1;
    // The new order list consists of two components:
    // 1. clusters already in GPU  2. sorted cluster according 
    // to reference number in descending order
    std::vector<std::pair<int, int> > temp_list(bcluster_cnt);
    int index = 0;
    int index2 = 0;
    // Part 1
    for (int i = 0; i < bcluster_cnt; i++){
        int cluid = bcluster_list[i];
        // Initialize the map
        reversemap[cluid] = i;

        bool che = false;
        if (pc_ != nullptr)
            che = pc_->readonDevice(cluid);
        
        if (che){
            reorder_list[index++] = cluid;
        }
        else {
            temp_list[index2].second = cluid;
            temp_list[index2++].first = query_per_bcluster[i];
        }
    }

    FAISS_ASSERT(index + index2 == bcluster_cnt);

    // Part 2
    t0 = elapsed();
    // multi_sort<int, int> (temp_list.data(), index2);
    std::sort(temp_list.begin(), temp_list.begin() + index2, Com<int,int>);
    t1 = elapsed();
    printf("Part 2 : %.3f ms\n", t1 - t0);

    // for (int i = 0; i < 20; i++){
    //     printf("%d %d\n", temp_list[i].second, temp_list[i].first);
    // }

    // Merge the two parts
    for (int i = index; i < bcluster_cnt; i++)
        reorder_list[i] = temp_list[i - index].second;

}


} // namespace gpu
} // namespace faiss