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
        maxDataCnt = 4096 * 8 * 2;
        nqMax = 1024;
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
        fprintf(fp, "%d %lf\n", it->first, it->second);   
    }
    fprintf(fp,"%d %lf\n", 0, 0.);
    
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
        int key;
        double value;
        fscanf(fp, "%d %lf", &key, &value);
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

double PipeProfiler::queryCom(int dataCnt) {
    FAISS_ASSERT(coms->istrained);

    auto target = coms->computeTimeDict.find(dataCnt);
    if(target != coms->computeTimeDict.end()){
        return target->second;
    }
    auto up = coms->computeTimeDict.lower_bound(dataCnt);
    auto down = up;
    if(up == coms->computeTimeDict.begin()){
        up++;
        FAISS_ASSERT(up != coms->computeTimeDict.end());
    }
    else{
        if(up == coms->computeTimeDict.end()){
            up--;
            down = up;
        }
        down--;
    }
    double upTime = up->second;
    double downTime = down->second;
    unsigned long upDataCnt = up->first;
    unsigned long downDataCnt = down->first;
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

int PipeProfiler::decideSplit(int queryCnt, int dataCnt){
        int split = 1;
        // Find an appropriate split num
        if (pc_->Min_Block > 0){
            while(dataCnt * split < pc_->Min_Block){
                split = split << 1;
            }
        }
        return split;
    }


void PipeProfiler::ComProfiler::train(){
    // param space

    int ntmax = p -> nqMax;
    int d = p->pc_->d;
    int k = 10;


    std::mt19937 rng;
    float *trainvecs = new float[ntmax * d];
    {
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < ntmax * d; i++) {
            trainvecs[i] = distrib(rng);
        }
    }
    auto pc = p->pc_;
    auto pgr = p->pgr_;
    auto index = p->index_;
    auto device = index->ivfPipeConfig_.device;
    auto h2d_stream = pgr->getCopyH2DStream(device);
    auto exe_stream = pgr->getExecuteStream(device);
    DeviceScope scope(device);

    std::vector<void*> ListDataP_vec;
    std::vector<void*> ListIndexP_vec;//fake index
    std::vector<int> ListLength_vec;

    int listNum = std::min(pc->bnlist, (int)pgr->pageNum_);

    faiss::gpu::MemBlock mb = p->pgr_->allocMemory(listNum);

    FAISS_ASSERT(mb.valid == true);

    ListDataP_vec.resize(listNum);
    ListIndexP_vec.resize(listNum);
    ListLength_vec.resize(listNum);

    // radomly set allocated pages to clusters
    // and free these pages
    for (int j = 0; j < mb.pages.size(); j++){
        int clus = j;
        p->pgr_->pageinfo[mb.pages[j]] = clus;
        p->pc_->setonDevice(clus, mb.pages[j], true);

        // Memory transfer
        p->pgr_->memcpyh2d(mb.pages[j]);
        ListDataP_vec[j] = pgr->getPageAddress(mb.pages[j]);
        ListIndexP_vec[j] = (void *)((float*)(ListDataP_vec[j]) + 
                pc->d * pc->BCluSize[clus]);
        ListLength_vec[j] = pc->BCluSize[clus];
    }


    faiss::gpu::PipeTensor<void*, 1, true> ListDataP_({listNum}, pc);
    ListDataP_.copyFrom(ListDataP_vec, h2d_stream);
    ListDataP_.setResources(pc, pgr);
    ListDataP_.memh2d(h2d_stream);

    void** ListDataP = ListDataP_.devicedata();

    faiss::gpu::PipeTensor<int, 1, true> ListLength_({listNum}, pc);
    ListLength_.copyFrom(ListLength_vec, h2d_stream);
    ListLength_.setResources(pc, pgr);
    ListLength_.memh2d(h2d_stream);

    int* ListLength = ListLength_.devicedata();

    faiss::gpu::PipeTensor<void*, 1, true> ListIndexP_({listNum}, pc);
    ListIndexP_.copyFrom(ListIndexP_vec, h2d_stream);
    ListIndexP_.setResources(pc, pgr);
    ListIndexP_.memh2d(h2d_stream);

    void** ListIndexP = ListIndexP_.devicedata();

    int* queryids = (int*)malloc(sizeof(int) * ntmax);
    for (int i = 0; i < ntmax; i++){
        queryids[i] = i;
    }

    int nq = 16;

    faiss::gpu::PipeTensor<int, 1, true> queryids_gpu({nq}, pc);
    queryids_gpu.copyFrom(queryids, h2d_stream);
    queryids_gpu.setResources(pc, pgr);
    queryids_gpu.memh2d(h2d_stream);

        
    faiss::gpu::PipeTensor<float, 2, true> queries_gpu({nq, d}, pc);
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
    
    int maxClus = p->maxDataCnt / nq;

    int clus = 1;
    while (clus <= maxClus)
    {

        int* query_bcluster_matrix = new int[2 * clus * nq];
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(clus, 0.8 * (double)clus);

        for (int i = 1; i < nq; i += 2) {
            int number = rand() % (2 * clus);//(int)distribution(generator);
            if(number >= 2 * clus){
                number = 2 * clus - 1;
            }
            else if(number == 0){
                number = 1;
            }
            for(int j = 0 ; j < 2 * clus ; j++){
                if(j < number){
                    query_bcluster_matrix[i * 2 * clus + j] = j % std::min(clus, listNum);
                }
                else{
                    query_bcluster_matrix[i * 2 * clus + j] = -1;
                }
            }
            for(int j = 0 ; j < 2 * clus ; j++){
                if(j < 2 * clus - number){
                    query_bcluster_matrix[(i - 1)  * 2 * clus + j] = j % std::min(clus, listNum);
                }
                else{
                    query_bcluster_matrix[(i - 1) * 2 * clus + j] = -1;
                }
            }
                
        }
        if(nq % 2 == 1){
            for(int j = 0 ; j < 2 * clus ; j++){
                if(j < clus){
                    query_bcluster_matrix[(nq - 1)  * 2 * clus + j] = j % std::min(clus, listNum);
                }
                else{
                    query_bcluster_matrix[(nq - 1) * 2 * clus + j] = -1;
                }
            }
        }
        faiss::gpu::PipeTensor<int, 2, true> query_cluster_matrix_gpu({nq, 2 * clus}, pc);
        query_cluster_matrix_gpu.copyFrom(query_bcluster_matrix, h2d_stream);
        query_cluster_matrix_gpu.setResources(pc, pgr);
        query_cluster_matrix_gpu.memh2d(h2d_stream);
        cudaStreamSynchronize(h2d_stream);

        int dataCnt = nq * clus;
        int split = p->decideSplit(nq, dataCnt);

        faiss::gpu::PipeTensor<float, 2, true> out_distances({nq, (int)k}, pc);
        out_distances.setResources(pc, pgr);
        out_distances.reserve();

        faiss::gpu::PipeTensor<int, 2, true> out_indices({nq, (int)k}, pc);
        out_indices.setResources(pc, pgr);
        out_indices.reserve();

        double t0 = timepoint();

        faiss::gpu::runKernelComputeReduce(
                        d,
                        k,
                        nq,
                        2 * clus,
                        queryids_gpu,
                        queries_gpu,
                        query_cluster_matrix_gpu,
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
                        device,
                        split);

        cudaStreamSynchronize(exe_stream);

        double t1 = timepoint();
        double tCnt = t1 - t0;
        computeTimeDict[dataCnt] = tCnt;
        clus *= 2 ;
        printf("query:%d, dataCnt:%d, split:%d. Result:%lf\n", nq, dataCnt, split, tCnt);
        delete[] query_bcluster_matrix;
        }

    
    for (int j = 0; j < mb.pages.size(); j++){
        int clus = j;
        p->pgr_->pageinfo[mb.pages[j]] = -1;
        p->pc_->setonDevice(clus, mb.pages[j], false);
        p->pgr_->freetree_->insert(mb.pages[j], mb.pages[j]);
    }
    
    delete[] trainvecs;
    
    free(queryids);    
    istrained = true;
}



} // namespace gpu
} // namespace faiss