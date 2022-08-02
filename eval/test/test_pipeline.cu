/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <algorithm>

#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <faiss/pipe/PipeScheduler.h>

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double t0;

void test_transpose(){

    // check correctness

    int clus = 100;
    int query = 10;
    int clusMax = 100;
    int queryMax = 100;

    int* clusQueryMat = new int[clus * query];
    std::vector<int> rows;
    rows.resize(clus);
    for(int i = 0; i < clus; i++){
        for (int j = 0; j < query; j++){
            clusQueryMat[i * query + j] = (i + j) % queryMax;
        }
        rows[i] = i;
    }

    int* queryClusMat;
    int* queryIds;
    int* clusIds = new int[10000];
    for (int i = 0; i < 10000; i++){
        clusIds[i] = i;
    }


    faiss::gpu::transpose(clusQueryMat, &queryClusMat, &clus, &query, queryMax, clusMax, rows, clusIds, &queryIds);

    printf("correctness check: query:%d, clus:%d\n", query, clus);
    for(int i = 0; i < query; i++) {
        for (int j = 0 ; j < clus; j++){
            printf("%d ", queryClusMat[i * clus + j]);
        }
        printf("\n");
    }

    

    
    // check performance

    clus = 10000;
    query = 500;
    clusMax = 10000;
    queryMax = 1000;

    delete[] clusQueryMat;
    delete[] clusIds;

    clusQueryMat = new int[clus * query];
    rows.resize(clus);
    for(int i = 0; i < clus; i++){
        for (int j = 0; j < query; j++){
            clusQueryMat[i * query + j] = (i + j) % queryMax;
        }
        rows[i] = i;
    }

    

    double t1 = elapsed();

    faiss::gpu::transpose(clusQueryMat, &queryClusMat, &clus, &query, queryMax, clusMax, rows, clusIds, &queryIds);

    double t2 = elapsed();

    printf("transpose finish in time: %.3f ms\n", (t2 - t1) * 1000);

    delete[] clusQueryMat;

    return;

}


void test_fake_scheduler(){

    // test reorder
    int cnt = 32;

    std::vector<int> bcluster_list(cnt);
    for(int i = 0; i < cnt; i++)
        bcluster_list[i] = i;

    std::vector<int> query_per_bcluster(cnt);
    std::uniform_real_distribution<> distrib{0.0, 30.0};
    std::random_device rd;
    std::default_random_engine rng {rd()};
    int maxval = 0;
    for(int i = 0; i < cnt; i++){
        query_per_bcluster[i] = int(distrib(rng));
        maxval = std::max(maxval, query_per_bcluster[i]);
    }

    double t1 = elapsed();
    faiss::gpu::PipeScheduler psch(nullptr, nullptr, cnt, bcluster_list.data(), 
        query_per_bcluster.data(), maxval, nullptr, nullptr, 0, 0, false);
    auto t2 = elapsed();

    
    printf(" ----------Demo pipeline groups ---------\n");
    for (int i = 0; i < psch.groups.size(); i++){
        printf("%d\n", psch.groups[i]);
    }

    printf("Construct pipescheduler time: %.3f ms\n", (t2 - t1) * 1000);


}


void test_real_scheduler(){


    double t1 = elapsed();


    // dimension of the vectors to index
    int d = 128;

    // size of the database we plan to index
    size_t nb = 400 * 10000;

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    size_t nt = 100 * 1000;

    // a reasonable number of centroids to index nb vectors
    int ncentroids = 1024;

    int dev_no = 0;
    
    printf ("[%.3f s] Begin d=%d nb=%ld nt=%ld dev_no=%d\n",
            elapsed() - t0, d, nb, nt, dev_no);


    // the coarse quantizer should not be dealloced before the index
    // 4 = nb of bytes per code (d must be a multiple of this)
    // 8 = nb of bits per sub-code (almost always 8)


    std::mt19937 rng;

    // training
    t1 = elapsed();
    printf("[%.3f s] Generating %ld vectors in %dD for training.\n",
           t1 - t0,
           nt,
           d);
    fflush(stdout);

    float *trainvecs = new float[nt * d];
    {
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nt * d; i++) {
            trainvecs[i] = distrib(rng);
        }
    }

    

    // remember a few elements from the database as queries
    size_t nq;
    std::vector<float> queries;
        
    int i0 = 1234;
    int i1 = 1234 + 128;

    nq = i1 - i0;

    t1 = elapsed();
    printf("[%.3f s] Building a dataset of %ld vectors and %ld queries to index.\n",
           t1 - t0,
           nb,
           nq);
    fflush(stdout);

    float *database = new float[nb * d];
    {
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }
    }

    queries.resize(nq * d);
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < d; j++) {
            queries[(i - i0) * d + j] = database[i * d + j];
        }
    }

    omp_set_num_threads(8);
#pragma omp parallel for
    for (int i = 0; i < 8; i++){
        if (i == 0)
            printf("omp is %d ", omp_in_parallel());
    }

    t1 = elapsed();
    printf("[%.3f s] training   ",
               t1 - t0);

    faiss::IndexIVFPipeConfig config;
    faiss::gpu::PipeGpuResources* pipe_res = new faiss::gpu::PipeGpuResources();
    faiss::IndexIVFPipe* index = new faiss::IndexIVFPipe(d, ncentroids, config, pipe_res, faiss::METRIC_L2);
    FAISS_ASSERT (config.interleavedLayout == true);

    index->train(nt, trainvecs);
    delete[] trainvecs;

    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
    t1 = elapsed();
    printf("[%.3f s] adding   ",
               t1 - t0);

    index->add(nb, database);
    //for (int i = 0; i < 16; i++)
    //    printf("%zu\n", index->get_list_size(i));
    printf("Prepare delete\n");
    sleep(1);
    delete[] database;
    printf("delete finishing\n");
    sleep(1);
    printf("Add finishing\n");
    sleep(1);
    index->balance();
    printf("Balance finishing\n");
    sleep(1);
    
    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
   
    auto pc = index->pipe_cluster;


    pipe_res->initializeForDevice(0, pc);


    float* coarse_dis;
    int* ori_idx;
    int* bcluster_per_query;
    int actual_nprobe;
    int* query_bcluster_matrix;
    int maxbcluster_per_query;
    int bcluster_cnt;
    int* bcluster_list;
    int* query_per_bcluster;
    int maxquery_per_bcluster;
    int* bcluster_query_matrix;



    printf("[%.3f s] Sample list starts.\n", elapsed() - t0);

    // sample_list() test

    index->sample_list(nq, queries.data(), &coarse_dis, &ori_idx,\
                    &bcluster_per_query, &actual_nprobe, &query_bcluster_matrix, &maxbcluster_per_query,\
                    &bcluster_cnt, &bcluster_list, &query_per_bcluster, &maxquery_per_bcluster,\
                    &bcluster_query_matrix);


    printf("[%.3f s] Profiler starts.\n", elapsed() - t0);

    index->profile();

    printf("[%.3f s] Scheduler starts.\n", elapsed() - t0);

    t1 = elapsed();
    faiss::gpu::PipeScheduler psch(pc, pipe_res, bcluster_cnt, bcluster_list, 
        query_per_bcluster, maxquery_per_bcluster, bcluster_query_matrix,
        index->profiler, nq, pc->bnlist, false);
    auto t2 = elapsed();

    
    printf(" ----------Demo pipeline groups ---------\n");
    for (int i = 0; i < psch.groups.size(); i++){
        printf("%d\n", psch.groups[i]);
    }

    printf("Construct pipescheduler time: %.3f ms\n", (t2 - t1) * 1000);


    psch.compute();
    
}



int main(){
    // Set the max threads num as 8
    omp_set_num_threads(8);
    t0 = elapsed();

 #pragma omp parallel for
     for (int i = 0; i < 8; i++){
         if (i==1)
             printf("omp is %d\n", omp_in_parallel());
     }

    test_transpose();

    test_fake_scheduler();

    test_real_scheduler();

    
}