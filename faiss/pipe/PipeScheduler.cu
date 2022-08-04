/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <sys/time.h>
#include <iostream>

#include <faiss/pipe/PipeScheduler.h>
#include <faiss/gpu/utils/PipeTensor.cuh>

namespace faiss {
namespace gpu{

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

PipeScheduler::PipeScheduler(PipeCluster* pc, PipeGpuResources* pgr, int bcluster_cnt_,
        int* bcluster_list_, int* query_per_bcluster_, int maxquery_per_bcluster_,
        int* bcluster_query_matrix_, PipeProfiler* profiler_,
        int queryMax_, int clusMax_, bool free_) 
        : pc_(pc), pgr_(pgr), bcluster_cnt(bcluster_cnt_), profiler(profiler_),
        bcluster_list(bcluster_list_), query_per_bcluster(query_per_bcluster_), \
        maxquery_per_bcluster(maxquery_per_bcluster_), \
        bcluster_query_matrix(bcluster_query_matrix_), free(free_),\
        queryMax(queryMax_), clusMax(clusMax_), num_group(1){
            reorder_list.resize(bcluster_cnt);

            reorder();

            group();

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
    std::vector<std::pair<int, int> > lru_list(bcluster_cnt);
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
            lru_list[index].second = cluid;
            lru_list[index++].first = pc_->readGlobalCount(cluid);
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
    // multi_sort<int, int> (lru_list.data(), index);
    std::sort(lru_list.begin(), lru_list.begin() + index, Com<int,int>);
    t1 = elapsed();
    printf("Part 2 : %.3f ms\n", (t1 - t0) * 1000);

    // for (int i = 0; i < 20; i++){
    //     printf("%d %d\n", temp_list[i].second, temp_list[i].first);
    // }

    // Merge the two parts
    for (int i = index; i < bcluster_cnt; i++)
        reorder_list[i] = temp_list[i - index].second;

    for (int i = 0; i < index; i++)
        reorder_list[index - i - 1] = lru_list[i].second;

    part_size = index;

}

void PipeScheduler::group(){
    canv = 0;

    pipelinegroup opt;

    // The 4 here is hyperparamter
    // Each group size can not overstep this value
    if (pgr_)
        max_size = pgr_->pageNum_ - part_size + part_size / 4;
    else
        max_size = reorder_list.size() - part_size + part_size / 4;

    if (part_size != 0) {
        if (part_size / 4 == part_size){
            groups.push_back(part_size);
        }
        else {
            groups.push_back(part_size / 4);
            groups.push_back(part_size);
        }
    }

    float delay = measure_com(part_size/4, part_size);
    int f1 = 1;
    int n = reorder_list.size();

    // prune 1
    for (int i = part_size + 1; i <= n; i++){
        float trantime = measure_tran(i - part_size);
        if (trantime < delay)
            f1 = i;
        else
            break;
        if (i - part_size >= max_size)
            break;
    }

    for (int i = f1; i <= n; i++){
        if (i - part_size > max_size)
            break;
        // prune 2
        float totaltime;
        float delaytime;
        if (!opt.content.empty()){
            float tran1 = measure_tran(i - part_size);
            float tran2 = measure_tran(n - i);
            float trantime = tran1 + tran2;
            float interval = delay - tran1;
            interval = interval > 0 ? interval : 0;
            delaytime = measure_com(part_size, i) + interval;
            totaltime = std::max(tran1, delay) + delaytime - interval;
            float comtime = totaltime + measure_com(i, n);
            float time = std::max(comtime, trantime);
            if (time >= opt.time)
                continue;
        }
        else{
            float tran1 = measure_tran(i - part_size);
            float interval = delay - tran1;
            interval = interval > 0 ? interval : 0;
            delaytime = measure_com(part_size, i) + interval;
            totaltime = std::max(tran1, delay) + delaytime - interval;
        }
        // recursively find rest groups
        pipelinegroup first_gr;
        first_gr.content.push_back(i);
        first_gr.time = totaltime;
        first_gr.delay = delaytime;
        pipelinegroup rest = group(i, totaltime, delaytime, 1);
        auto size = first_gr.content.size();
        first_gr.content.resize(size + rest.content.size());
        memcpy (first_gr.content.data() + size, rest.content.data(), sizeof(int) * rest.content.size());
        first_gr.time = rest.time;
        if (opt.time > first_gr.time)
            opt = first_gr;
    }

    std::cout << "OPT: " << opt.time << "ms \n";

    auto size = groups.size();
    groups.resize(size + opt.content.size());
    memcpy(groups.data() + size,  opt.content.data(), sizeof(int) * opt.content.size());

}

PipeScheduler::pipelinegroup PipeScheduler::group(int staclu, float total, float delay, int depth){
    pipelinegroup opt;
    int n = reorder_list.size();
    if (staclu == n){
        opt.time = total;
        opt.delay = delay;
        
        canv += 1;

        if(canv % 1000000 == 0){
            printf("%d\n", canv/1000000);
        }
        return opt;
    }
    int f1 = staclu + 1;

    // prune 1
    for (int i = staclu + 1; i <= n; i++){
        float trantime = measure_tran(i - staclu);
        if (trantime < delay)
            f1 = i;
        else
            break;
        if (i - staclu >= max_size)
            break;
    }

    for (int i = f1; i <= n; i++){
        if (i - staclu > max_size)
            break;
        // prune 2
        float totaltime;
        float delaytime;
        if (!opt.content.empty()){
            float tran1 = measure_tran(i - staclu);
            float tran2 = measure_tran(n - i);
            float trantime = tran1 + tran2;
            float interval = delay - tran1;
            interval = interval > 0 ? interval : 0;
            delaytime = measure_com(staclu, i) + interval;
            if (delay > tran1)
                totaltime = total + delaytime - interval;
            else
                totaltime = total - delay + tran1 + delaytime;
            float comtime = totaltime + measure_com(i, n);
            float time = std::max(comtime, trantime);
            if (time >= opt.time)
                continue;
        }
        else{
            float tran1 = measure_tran(i - staclu);
            float interval = delay - tran1;
            interval = interval > 0 ? interval : 0;
            delaytime = measure_com(staclu, i) + interval;
            if (delay > tran1)
                totaltime = total + delaytime - interval;
            else
                totaltime = total - delay + tran1 + delaytime;
        }
        // recursively find rest groups
        pipelinegroup first_gr;
        first_gr.content.push_back(i);
        first_gr.time = totaltime;
        first_gr.delay = delaytime;
        // std::cout << "IN " << i << " Time : " << totaltime << "ms, " << delaytime << "ms \n";
        pipelinegroup rest = group(i, totaltime, delaytime, depth + 1);
        // std::cout << "The " << i << "th OUT: ";
        // for (int j = 0; j < rest.content.size(); j++){
        //     std::cout << rest.content[j] << " ";
        // }
        // std::cout << " Time: " << rest.time << "ms \n";
        auto size = first_gr.content.size();
        first_gr.content.resize(size + rest.content.size());
        memcpy (first_gr.content.data() + size, rest.content.data(), sizeof(int) * rest.content.size());
        first_gr.time = rest.time;
        if (opt.time > first_gr.time)
            opt = first_gr;
    }

    canv += 1;

    if(canv % 1000000 == 0){
        printf("%d\n", canv/1000000);
    }
    return opt;

}

float PipeScheduler::measure_tran(int num){
    if (num == 0)
        return 0.;
    
    if (profiler != nullptr) {
        return profiler->queryTran(num);
    }
    return 0.01 * num + 0.02;
}

float PipeScheduler::measure_com(int sta, int end){
    if (sta == end)
        return 0.;
    // return 0.6 * (end - sta) * (float(reorder_list.size())/end) + 0.1;
    if (profiler != nullptr) {
        int dataCnt = 0;
        for( int i = sta; i < end; i++) {
            dataCnt += query_per_bcluster[reversemap[reorder_list[i]]];
        }
        return profiler->queryCom(dataCnt);
    }
    else {
        int dataCnt = 0;
        for( int i = sta; i < end; i++) {
            dataCnt += query_per_bcluster[reversemap[reorder_list[i]]];
        }
        return 0.0002 * (double)dataCnt + .02;
    }

}


void PipeScheduler::compute(){

    auto groupNum = groups.size();
 
    for (int i = 0; i < groupNum; i++) {
        int start = (i == 0) ? 0 : groups[i - 1];
        int cnt = groups[i] - start;

        std::vector<int> rows;
        rows.resize(cnt);
        int* queryClusSubmat;

        for (int j = 0; j < cnt; j++){
            int order = reversemap[reorder_list[start + j]];
            rows[j] = order;
        }
        int clus = cnt;
        int query = query_per_bcluster[reversemap[reorder_list[start]]];
        int* queryIds;

        transpose(bcluster_query_matrix, &queryClusSubmat, &clus, &query, queryMax, clusMax, rows, bcluster_list, &queryIds);

        delete[] queryClusSubmat;
        delete[] queryIds;

    }
/*
    PipeTensor<int, 2, true> bcluster_query_tensor({n, cluster_split}, pc);


    bcluster_query_tensor.copyFrom(result_distances, h2d_stream);
    bcluster_query_tensor.setResources(pc, pipe_res);
    bcluster_query_tensor.memh2d(h2d_stream);
*/


    
}


void transpose(int* clusQueryMat, int** queryClusMat, int* clus, int* query, int queryMax, int clusMax, std::vector<int>& rows, int* clusIds, int** queryIds) {

    int oriClus = *clus;
    int oriQuery = *query;
    int afterClus = 0;
    int afterQuery = 0;

    int* clusPerQuery = new int[queryMax];
    *queryIds = new int[queryMax];

    std::fill(clusPerQuery, clusPerQuery + queryMax, 0);
    std::vector<std::vector<int>> queryClus;
    queryClus.resize(queryMax);

    omp_set_num_threads(8);
    int nt = omp_get_max_threads();

    int** clusPerQuerySlave = new int*[nt];
    int** SlaveOffset = new int*[nt];
    int** queryClusMatSlave = new int*[nt];

    #pragma omp parallel for
    for (int i = 0; i < nt; i++){
        clusPerQuerySlave[i] = new int[queryMax];
        queryClusMatSlave[i] = new int[queryMax * clusMax];
        SlaveOffset[i] = new int[queryMax];
        std::fill(clusPerQuerySlave[i], clusPerQuerySlave[i] + queryMax, 0);
    }



    #pragma omp parallel for
    for (int i = 0; i < oriClus; i++) {
        int clusRow = rows[i];
        int rank = omp_get_thread_num();
        int clus = clusIds[clusRow];
        for (int j = 0; j < oriQuery; j++) {
            int query = clusQueryMat[oriQuery * clusRow + j];
            if (query == -1) {
                continue;
            }
            queryClusMatSlave[rank][query * clusMax + clusPerQuerySlave[rank][query]] = clus;
            clusPerQuerySlave[rank][query] += 1;
        }

    }

    for(int i = 0; i < queryMax; i++) {
        for (int j = 0; j < nt; j++) {
            SlaveOffset[j][i] = clusPerQuery[i];
            clusPerQuery[i] += clusPerQuerySlave[j][i];
        }
        afterClus = std::max(afterClus, clusPerQuery[i]);
        if (clusPerQuery[i] != 0) {
            (*queryIds)[afterQuery] = i;
            afterQuery ++;
        }
    }

    *queryClusMat = new int[afterQuery * afterClus];

    #pragma omp parallel for
    for (int i = 0; i < afterQuery ; i++) {
        int queryId = (*queryIds)[i];
        for (int j = 0; j < nt ; j++) {
            memcpy (*queryClusMat + afterClus * i + SlaveOffset[j][queryId], queryClusMatSlave[j] + queryId * clusMax, clusPerQuerySlave[j][queryId] * sizeof(int));
        }
        std::fill(*queryClusMat + afterClus * i + clusPerQuery[queryId], *queryClusMat + afterClus * (i + 1), -1);
    }


    *clus = afterClus;
    *query = afterQuery;
    return;

}


} // namespace gpu
} // namespace faiss