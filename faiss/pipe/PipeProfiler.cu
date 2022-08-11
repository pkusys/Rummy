/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <vector>
#include <random>
#include <sys/time.h>
#include <stdio.h>
#include <faiss/pipe/PipeProfiler.cuh>

// Record the current time (ms)
double timepoint() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

namespace faiss{
namespace gpu{




PipeProfiler::PipeProfiler(IndexIVFPipe *index)
    {
        index_ = index;
        pc_ = index->pipe_cluster;
        pgr_ = index->pipe_provider;
        trans = new TranProfiler(this);
        coms = new ComProfiler(this);
    }

void PipeProfiler::train(){
    // Train the sub-profilers
    coms->train();
    trans->train();

    istrained = true;
}

void PipeProfiler::save(const char* path_){
    char path[100];
    if(strcmp(path_, "") == 0 ){
        strcpy(path, "profileSave.txt");
    }
    else{
        strcpy(path, path_);
    }

    FILE * fp;

    fp = fopen (path, "w+");

    fprintf(fp, "{trans}\n");
    for (auto it = trans->tranTimeDict.begin(); it != trans->tranTimeDict.end(); it++){
        fprintf(fp, "%d %lf\n", it->first, it->second);
    }
    fprintf(fp, "%zu %lf\n", (unsigned long)0, 0.);

    fprintf(fp, "{coms}\n");
    for (auto it = coms->computeTimeDict.begin(); it != coms->computeTimeDict.end(); it++){
        int dataCnt = (it->first) >> 32;
        int split = (it->first) & 0xffffffff;
        fprintf(fp, "%d %d %lf\n", dataCnt, split, it->second);
    }
    fprintf(fp, "%zu %lf\n", (unsigned long)0, 0.);
    
    fprintf(fp,"{the-end}\n");

    fclose(fp);

}

void PipeProfiler::load(const char* path_){

    char path[100];
    char buffer[100];
    if(strcmp(path_, "") == 0 ){
        strcpy(path, "profileSave.txt");
    }
    else{
        strcpy(path, path_);
    }

    coms->computeTimeDict.clear();
    trans->tranTimeDict.clear();
    FILE * fp;

    fp = fopen (path, "r");
    fscanf(fp, "%s", buffer);
    // printf("loading profiler starts.\n");
    // printf("%s\n",buffer);

    while(true){
        int key;
        double value;
        fscanf(fp, "%d %lf", &key, &value);
        // printf("%d %lf\n", key, value);
        if(key == 0){
            break;
        }
        trans->tranTimeDict[key] = value;
    }        
    fscanf(fp, "%s", buffer);
    // printf("%s\n", buffer);

    while(true){
        unsigned long key;
        int dataCnt;
        int split;

        double value;
        fscanf(fp, "%d %d %lf", &dataCnt, &split, &value);
        key = ((unsigned long)dataCnt << 32) + (unsigned long)split;
        // printf("%zu %lf\n", key, value);
        if(key == 0){
            break;
        }
        coms->computeTimeDict[key] = value;
    }
    
    fscanf(fp, "%s", buffer);
    // printf("%s\n", buffer);
    fclose(fp);

    trans->istrained = true;
    coms->istrained = true;
    this->istrained = true;

    return;
}

double PipeProfiler::queryTran(int pageCnt) {
    FAISS_ASSERT(trans->istrained);

    auto target = trans->tranTimeDict.find(pageCnt);

    if (target != trans->tranTimeDict.end()){
        return target->second;
    }

    auto up = trans->tranTimeDict.lower_bound(pageCnt);
    auto down = up;
    if(up == trans->tranTimeDict.end()){
        up--;
        down = up;
        down--;
    }
    else if (down == trans->tranTimeDict.begin()){
        up ++;
    }
    else{
        down --;
    }


    double downTime = down->second;
    double upTime = up->second;
    double realTime = downTime + (upTime - downTime) * (pageCnt - (double)(down->first)) / ((double)(up->first) - (double)(down->first));


    return realTime;
}

double PipeProfiler::queryCom(int dataCnt, int split) {
    FAISS_ASSERT(coms->istrained);
    bool goodsplit = false;
    for(int i = 1; i <= 256; i *= 2) {
        if(split == i) {
            goodsplit = true;
            break;
        }
    }
    FAISS_ASSERT(goodsplit == true);
    unsigned long key = (((unsigned long)dataCnt) << 32) + split;

    auto target = coms->computeTimeDict.find(key);
    if (target != coms->computeTimeDict.end()) {
        return target->second;
    }
    
    auto up = coms->computeTimeDict.lower_bound(key);
    auto down = up;
    bool found = false;
    if(up == coms->computeTimeDict.begin()){
        up++;
        while(up != coms->computeTimeDict.end()){
            int split_ = up->first & 0xffffffff;
            if(split_ == split){
                found = true;
                break;
            }
            up++;
        }
        if(!found){
            printf("error1:%d\n",dataCnt);
        }
        FAISS_ASSERT(found);
    }
    else{
        if(up == coms->computeTimeDict.end()){
            up--;
        }
        down = up;   

        while (down != coms->computeTimeDict.begin()) {
            down--;
            int split_ = down->first & 0xffffffff;
            if (split_ == split){
                found = true;
                break;
            }
        }
        if(!found){
            printf("error2:%d\n",dataCnt);
        }
        FAISS_ASSERT(found);
    }
    


    double upTime = up->second;
    double downTime = down->second;
    unsigned long upDataCnt = up->first >> 32;
    unsigned long downDataCnt = down->first >> 32;
    double realTime = 
        downTime + (upTime - downTime) * (dataCnt - (double)downDataCnt) / ((double)upDataCnt - (double)downDataCnt);
    return realTime;
 }




void PipeProfiler::TranProfiler::train(){
    // param space
    int end = p->pgr_->pageNum_;
    end = std::min(end, p->pc_->bnlist);
    //printf("end:%d\n", end);
    int i = 1;

    std::vector<int> pages;
    std::vector<double> perf; 

    bool doubleone = false;

    while (i <= end){
        if(doubleone || i!=1)
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

        double totalTime = timepoint() - t0;

        tranTimeDict[i] = totalTime;
        
        // Free these pages
        for (int j = 0; j < mb.pages.size(); j++){
            int clus = j;
            p->pgr_->pageinfo[mb.pages[j]] = -1;
            p->pc_->setonDevice(clus, mb.pages[j], false);
            p->pgr_->freetree_->insert(mb.pages[j], mb.pages[j]);
        }
        
        if(doubleone || i!=1){
            i = i << 1;
        }
        else{
            doubleone = true;
        }
    }


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
    p->maxClus = 4096 * 8;

    int* query_bcluster_matrix = new int[p->maxClus * 16];

    for (int i = 0; i < p->maxClus; i++) {
        query_bcluster_matrix[i] = i % pc->bnlist;
    }
    for (int i = 0 ;i < 16; i++) {
        memcpy(query_bcluster_matrix + p->maxClus * i, query_bcluster_matrix, sizeof(int) * p->maxClus);
    }

    faiss::gpu::PipeTensor<int, 2, true> query_cluster_matrix_gpu({nq, p->maxClus}, pc);
    query_cluster_matrix_gpu.copyFrom(query_bcluster_matrix, h2d_stream);
    query_cluster_matrix_gpu.setResources(pc, pgr);
    query_cluster_matrix_gpu.memh2d(h2d_stream);

    cudaStreamSynchronize(h2d_stream);




    while (split <= 256) {
        clus = 1;
        while (clus <= p->maxClus / split)
        {
            

            faiss::gpu::PipeTensor<float, 2, true> out_distances({nq, (int)k}, pc);
            out_distances.setResources(pc, pgr);
            out_distances.reserve();

            faiss::gpu::PipeTensor<int, 2, true> out_indices({nq, (int)k}, pc);
            out_indices.setResources(pc, pgr);
            out_indices.reserve();

            double t0 = timepoint();

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

            double t1 = timepoint();
            int dataCnt = nq * clus;
            unsigned long key = (((unsigned long)dataCnt) << 32) + split; 
            double tCnt = t1 - t0;
            computeTimeDict[key] = tCnt;
            clus *= 2 ;
            //printf("dataCnt:%d, split:%d. Result:%lf\n", dataCnt, split, tCnt);
        }

        
        split *= 2;
    }
    
    
    
    delete[] trainvecs;
    delete[] query_bcluster_matrix;
    free(queryids);    
    istrained = true;
}



} // namespace gpu
} // namespace faiss