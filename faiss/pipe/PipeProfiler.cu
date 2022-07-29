/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>
#include <random>
#include <sys/time.h>
#include <faiss/pipe/PipeProfiler.cuh>

// Record the current time (ms)
double timepoint() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-3;
}

namespace faiss{
namespace gpu{

void PipeProfiler::TranProfiler::train(){
    // param space
    int end = p->pgr_->pageNum_;
    end = std::min(end, p->pc_->bnlist);
    //printf("end:%d\n", end);
    int i = 1;

    std::vector<int> pages;
    std::vector<double> perf; 

    while (i <= end){
        pages.push_back(i);

        auto t0 = timepoint();

        // allocate memory
        faiss::gpu::MemBlock mb = p->pgr_->allocMemory(i);

        FAISS_ASSERT(mb.valid == true);

        // radomly set allocated pages to clusters
        // and free these pages
        for (int j = 0; j < mb.pages.size(); j++){
            int clus = j;
            p->pgr_->pageinfo[mb.pages[j]] = clus;
            p->pc_->setonDevice(clus, mb.pages[j], true);

            // Memory transfer
            p->pgr_->memcpyh2d(mb.pages[j]);

            p->pc_->addGlobalCount(clus, mb.pages[j], 1);
        }

        auto t1 = timepoint();
        perf.push_back(t1 - t0);
        
        // Free these pages
        for (int j = 0; j < mb.pages.size(); j++){
            int clus = j;
            p->pgr_->pageinfo[mb.pages[j]] = -1;
            p->pc_->setonDevice(clus, mb.pages[j], false);
            p->pgr_->freetree_->insert(mb.pages[j], mb.pages[j]);
        }
        
        i = i << 1;
    }

    // Fit with least square method
    double t1 = 0, t2 = 0, t3 = 0, t4 = 0;  
    for (i = 0; i < pages.size(); i++){  
        t1 += pages[i] * pages[i];  
        t2 += pages[i];  
        t3 += pages[i] * perf[i];  
        t4 += perf[i];  
    }  
    a = (t3 * pages.size() - t2 * t4) / (t1 * pages.size() - t2 * t2);  
    // b = (t4 - a * t2) / pages.size();  
    b = (t1 * t4 - t2 * t3) / (t1 * pages.size() - t2 * t2);

    istrained = true;
}


void PipeProfiler::ComProfiler::train(){
    // param space

    int nt = 16;
    int d = p->pc_->d;
    int k = 10;


    std::mt19937 rng;
    float *trainvecs = new float[nt * d];
    {
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nt * d; i++) {
            trainvecs[i] = distrib(rng);
        }
    }
    auto pc = p->pc_;
    auto pgr = p->pgr_;
    auto h2d_stream = pgr->getCopyH2DStream(0);
    auto exe_stream = pgr->getExecuteStream(0);

    std::vector<void*> ListDataP_vec;
    std::vector<void*> ListIndexP_vec;//fake index
    ListDataP_vec.resize(pc->bnlist);
    ListIndexP_vec.resize(pc->bnlist);

    for(int i = 0; i < pc->bnlist; i++) {
        void* dataP;
        void* IndexP;
        int vectorNum = pc->BCluSize[i];
        vectorNum = (vectorNum + 31) / 32 * 32;
        int bytes = vectorNum * d * sizeof(float);
        cudaMalloc(&dataP, bytes);
        cudaMemcpy(dataP, pc->Mem[i], bytes, cudaMemcpyHostToDevice);
        ListDataP_vec[i] = dataP;

        cudaMalloc(&IndexP, sizeof(int) * pc->BCluSize[i]);
        cudaMemcpy(IndexP, pc->Balan_ids[i], sizeof(int) * pc->BCluSize[i], cudaMemcpyHostToDevice);
        ListIndexP_vec[i] = IndexP;
    }

    auto ListLength_vec = pc->BCluSize;
    faiss::gpu::PipeTensor<void*, 1, true> ListDataP_({pc->bnlist}, pc);
    ListDataP_.copyFrom(ListDataP_vec, h2d_stream);
    ListDataP_.setResources(pc, pgr);
    ListDataP_.memh2d(h2d_stream);

    void** ListDataP = ListDataP_.devicedata();

    faiss::gpu::PipeTensor<int, 1, true> ListLength_({pc->bnlist}, pc);
    ListLength_.copyFrom(ListLength_vec, h2d_stream);
    ListLength_.setResources(pc, pgr);
    ListLength_.memh2d(h2d_stream);

    int* ListLength = ListLength_.devicedata();

    faiss::gpu::PipeTensor<void*, 1, true> ListIndexP_({pc->bnlist}, pc);
    ListIndexP_.copyFrom(ListIndexP_vec, h2d_stream);
    ListIndexP_.setResources(pc, pgr);
    ListIndexP_.memh2d(h2d_stream);

    void** ListIndexP = ListIndexP_.devicedata();

    int* queryids = (int*)malloc(sizeof(int) * nt);
    for (int i = 0; i < nt; i++){
        queryids[i] = i;
    }

    faiss::gpu::PipeTensor<int, 1, true> queryids_gpu({nt}, pc);
    queryids_gpu.copyFrom(queryids, h2d_stream);
    queryids_gpu.setResources(pc, pgr);
    queryids_gpu.memh2d(h2d_stream);

    
    faiss::gpu::PipeTensor<float, 2, true> queries_gpu({nt, d}, pc);
    queries_gpu.copyFrom(trainvecs, h2d_stream);
    queries_gpu.setResources(pc, pgr);
    queries_gpu.memh2d(h2d_stream);

    bool dir;
    if (p->index_->metric_type == faiss::MetricType::METRIC_L2) {
        faiss::gpu::L2Distance metr;
        dir = metr.kDirection;                                            
    } else if (p->index_->metric_type == faiss::MetricType::METRIC_INNER_PRODUCT) {
        faiss::gpu::IPDistance metr;          
        dir = metr.kDirection;
    }

    // query*cluster
    

    int nq = nt;//fixed: 16
    int clus = 1;
    int split = 1;
    constexpr int maxClus = 4096 * 8;

    int* query_bcluster_matrix = new int[4096 * 8 * 16];

    for (int i = 0; i < maxClus; i++) {
        query_bcluster_matrix[i] = i % pc->bnlist;
    }
    for (int i = 0 ;i < 16; i++) {
        memcpy(query_bcluster_matrix + maxClus * i, query_bcluster_matrix, sizeof(int) * maxClus);
    }

    faiss::gpu::PipeTensor<int, 2, true> query_cluster_matrix_gpu({nq, maxClus}, pc);
    query_cluster_matrix_gpu.copyFrom(query_bcluster_matrix, h2d_stream);
    query_cluster_matrix_gpu.setResources(pc, pgr);
    query_cluster_matrix_gpu.memh2d(h2d_stream);

    cudaStreamSynchronize(h2d_stream);




    while (split <= 256) {
        clus = 1;
        while (clus <= maxClus / split)
        {
            double t0 = timepoint();

            faiss::gpu::PipeTensor<float, 2, true> out_distances({nq, (int)k}, pc);
            out_distances.setResources(pc, pgr);
            out_distances.reserve();

            faiss::gpu::PipeTensor<int, 2, true> out_indices({nq, (int)k}, pc);
            out_indices.setResources(pc, pgr);
            out_indices.reserve();

            
            auto query_cluster_matrix_gpu_ = query_cluster_matrix_gpu.narrow(1, 0, clus);

            faiss::gpu::runKernelComputeReduce(
                            d,
                            k,
                            nq,
                            clus,
                            queryids_gpu,
                            queries_gpu,
                            query_cluster_matrix_gpu_,
                            ListDataP,
                            p->index_->ivfPipeConfig_.indicesOptions,
                            ListLength,
                            ListIndexP,
                            p->index_->metric_type,
                            dir,
                            out_distances,
                            out_indices,
                            pc,
                            pgr,
                            0,
                            split);

            cudaStreamSynchronize(exe_stream);

            int dataCnt = nq * clus;
            long key = (((long)dataCnt) << 32) + split;
            double t1 = timepoint();
            double tCnt = t1 - t0;
            computeTimeDict[key] = tCnt;
            clus *= 2 ;
            //printf("dataCnt:%d, split:%d. Result:%lf\n", dataCnt, split, tCnt);
        }

        
        split *= 2;
    }
    
    
    

    delete queryids;
    
    istrained = true;
}



} // namespace gpu
} // namespace faiss