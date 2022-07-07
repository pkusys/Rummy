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

#include <omp.h>
#include <sys/time.h>
#include <unistd.h>

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/pipe/IndexIVFPipe.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/impl/FaissAssert.h>
#include <faiss/pipe/PipeKernel.cuh>
#include <cstdio>
#include <omp.h>
#include <cinttypes>
#include <stdint.h>
#include <algorithm>
#include <mutex>
#include <string.h>
#include <limits>
#include <memory>
#include <unistd.h>
#include <cuda_runtime.h>

/*
    // check success after kernel function calls
    cudaStreamSynchronize(stream);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        FAISS_THROW_FMT("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

*/


double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

double t0;

void search_demo(
            faiss::gpu::StandardGpuResources* res,
            faiss::IndexIVFPipe* index,
            size_t n,
            const float* x,
            size_t k,
            float* distances,
            size_t* labels,
            bool display){
                
            float* coarse_dis;
            int* ori_idx;
            size_t* bcluster_per_query;
            size_t actual_nprobe;
            int* query_bcluster_matrix;
            size_t maxbcluster_per_query;
            int bcluster_cnt;
            int* bcluster_list;
            int* query_per_bcluster;
            int maxquery_per_bcluster;
            int* bcluster_query_matrix;

            double t1 = elapsed();

            // sample_list() test
            //NOTE: omp is not enabled as this is a .cu file
            index->sample_list(n, x, &coarse_dis, &ori_idx,\
                    &bcluster_per_query, &actual_nprobe, &query_bcluster_matrix, &maxbcluster_per_query,\
                    &bcluster_cnt, &bcluster_list, &query_per_bcluster, &maxquery_per_bcluster,\
                    &bcluster_query_matrix);
            
            printf("{FINISHED in %.3f ms}\n", (elapsed() - t1) * 1000);

            if(display) {
                // Display the query-bcluster matrix
                printf("[%.3f s] Query results (vector ids, then distances)(only display 10 queries):\n",
                    elapsed() - t0);

                printf("maxbcluster_per_query:%zu, actual_nprobe:%zu\n", maxbcluster_per_query, actual_nprobe);
            
                for (int i = 0; i < 10; i++) {
                    printf("query %2d: width %zu \n", i, bcluster_per_query[i]);
                    for(int j = 0; j < actual_nprobe; j++) {
                        printf("%d cluster,dis:%f   ", ori_idx[j + i * actual_nprobe], coarse_dis[j + i * actual_nprobe]);
                    }
                    printf("\n");
                    
                }

                printf("balanced clusters:\n");
                for (int i = 0; i < 10; i++) {
                    for (int j = 0; j < maxbcluster_per_query; j++) {
                        printf("%d  ", query_bcluster_matrix[j + i * maxbcluster_per_query]);
                    }
                printf("\n");
                }
                

                // Display the bcluster-query matrix
                printf ("[%.3f s] bcluster-query results(only display 10 queries):\n",
                    elapsed() - t0);
                
                printf ("bcluster_cnt:%d, maxquery_per_bcluster:%d\n", bcluster_cnt, maxquery_per_bcluster);
                for (int i = 0; i < bcluster_cnt; i++) {
                    printf("bcluster_id:%d, %d queries:   ", bcluster_list[i], query_per_bcluster[i]);
                    for (int j = 0; j < query_per_bcluster[i]; j++) {
                        printf("%d  ", bcluster_query_matrix[i * maxquery_per_bcluster + j]);
                    }
                    for(int j = query_per_bcluster[i]; j < maxquery_per_bcluster; j++){
                        FAISS_ASSERT(bcluster_query_matrix[i * maxquery_per_bcluster + j]==-1);
                    }
                    printf("\n");
                }
            }
            else{
                printf("[%.3f s] Query results (vector ids, then distances)(only display shape info):\n",
                    elapsed() - t0);

                printf("nquery:%zu,maxbcluster_per_query:%zu, actual_nprobe:%zu\n", n, maxbcluster_per_query, actual_nprobe);

            }


            //move the cluster data to GPU. The time of this part is not measured.
            faiss::gpu::DeviceScope scope(0);
            std::vector<void*> ListDataP_vec;
            std::vector<void*> ListIndexP_vec;//fake index
            faiss::PipeCluster* pc = index->pipe_cluster;
            ListDataP_vec.resize(pc->bnlist);
            ListIndexP_vec.resize(pc->bnlist);
            int cnt = 0;

            for(int i = 0; i < pc->bnlist; i++) {
                std::vector<size_t> ListIndices;
                ListIndices.resize(pc->BCluSize[i]);
                void* dataP;
                void* IndexP;
                cudaMalloc(&dataP, pc->MemBytes[i]);
                cudaMemcpy(dataP, pc->Mem[i], pc->MemBytes[i], cudaMemcpyHostToDevice);
                ListDataP_vec[i] = dataP;
                for (int j = 0; j < pc->BCluSize[i]; j++) {
                    ListIndices[j] = cnt;
                    cnt++;
                }
                cudaMalloc(&IndexP, sizeof(size_t) * pc->BCluSize[i]);
                cudaMemcpy(IndexP, ListIndices.data(), sizeof(size_t) * pc->BCluSize[i], cudaMemcpyHostToDevice);
                ListIndexP_vec[i] = IndexP;
            }
  
            auto ListLength_vec = pc->BCluSize.data();

            auto res_ = res->getResources().get();
            auto stream = res->getDefaultStream(0);
            
            faiss::gpu::DeviceTensor<void*, 1, true> ListDataP_ = \
                    faiss::gpu::toDeviceTemporary<void*, 1>(res_, 0, ListDataP_vec.data(), stream, {pc->bnlist});

            void** ListDataP = ListDataP_.data();

            faiss::gpu::DeviceTensor<int, 1, true> ListLength_ = \
                    faiss::gpu::toDeviceTemporary<int, 1>(res_, 0, ListLength_vec, stream, {pc->bnlist});

            int* ListLength = ListLength_.data();

            faiss::gpu::DeviceTensor<void*, 1, true> ListIndexP_ = \
                    faiss::gpu::toDeviceTemporary<void*, 1>(res_, 0, ListIndexP_vec.data(), stream, {pc->bnlist});

            void** ListIndexP = ListIndexP_.data();
            
            int* queryids = (int*)malloc(sizeof(int) * n);
            for (int i = 0; i < n; i++){
                queryids[i] = i;
            }

            bool dir;
            if (index->metric_type == faiss::MetricType::METRIC_L2) {
                faiss::gpu::L2Distance metr;
                dir = metr.kDirection;                                            
            } else if (index->metric_type == faiss::MetricType::METRIC_INNER_PRODUCT) {
                faiss::gpu::IPDistance metr;          
                dir = metr.kDirection;
            }



            // First test without spliting.
            {
                
                // start to measure time from here on.
                
                
                faiss::gpu::DeviceTensor<float, 3, true> best_distances(
                            res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {(int)n, (int)maxbcluster_per_query, (int)k});
                faiss::gpu::DeviceTensor<size_t, 3, true> best_indices(
                            res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {(int)n, (int)maxbcluster_per_query, (int)k});
                
                faiss::gpu::DeviceTensor<int, 1, true> queryids_gpu = \
                        faiss::gpu::toDeviceTemporary<int, 1>(res_, 0, queryids, stream, {(int)n});

                faiss::gpu::DeviceTensor<float, 2, true> queries_gpu = \
                        faiss::gpu::toDeviceTemporary<float, 2>(res_, 0, (float*)x, stream, {(int)n, index->d});
                faiss::gpu::DeviceTensor<int, 2, true> query_cluster_matrix_gpu = \
                        faiss::gpu::toDeviceTemporary<int, 2>(res_, 0, query_bcluster_matrix, stream, {(int)n, (int)maxbcluster_per_query});

                faiss::gpu::DeviceTensor<float, 2, true> out_distances(
                            res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {(int)n, (int)k});
                faiss::gpu::DeviceTensor<size_t, 2, true> out_indices(
                            res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {(int)n, (int)k});

                // compute() test
                
                t1 = elapsed();
                printf("[%.3f s]start to measure time of compute\n", t1 - t0);

                faiss::gpu::runKernelCompute(
                    index->d,
                    k,
                    n,
                    maxbcluster_per_query,
                    queryids_gpu,
                    queries_gpu,
                    query_cluster_matrix_gpu,
                    best_indices,
                    best_distances,
                    ListDataP,
                    index->ivfPipeConfig_.indicesOptions,
                    ListLength,
                    index->metric_type,
                    stream);
                cudaStreamSynchronize(stream);
                printf("{FINISHED in %.3f ms}\n", (elapsed() - t1) * 1000);

                t1 = elapsed();
                printf("[%.3f s]start to measure time of reduce\n", t1 - t0);


                //reduce() test
                
                

                faiss::gpu::runKernelReduce(
                    maxbcluster_per_query,
                    best_distances,
                    best_indices,
                    query_cluster_matrix_gpu,
                    k,
                    ListIndexP,
                    index->ivfPipeConfig_.indicesOptions,
                    dir,
                    out_distances,
                    out_indices,
                    stream);
                cudaStreamSynchronize(stream);
                printf("{FINISHED in %.3f ms}\n", (elapsed() - t1) * 1000);

                auto outcome_distance = faiss::gpu::toHost<float, 2>(out_distances.data(), stream, {(int)n, (int)k});
                auto outcome_indices = faiss::gpu::toHost<size_t, 2>(out_indices.data(), stream, {(int)n, (int)k});


                if(display) {
                    for(int q0 = 0; q0 < 10; q0++) {
                        for(int q1 = 0;q1 < k; q1++) {
                            printf("%f,%zu  ",*(outcome_distance[q0][q1].data()),  *(outcome_indices[q0][q1].data()));
                        }
                        printf("\n");
                    }
                }
            }




            // Then test split the matrix by query and cluster to test correctness.
            int query_split = 4;
            int cluster_split = 4;

            // start to measure time from here on.
            t1 = elapsed();
            double accumulated = 0.0;
            printf("[%.3f s]start to test split\n", t1 - t0);

            std::vector<float*> result_distances;
            std::vector<size_t*> result_indices;

            result_distances.resize(n * cluster_split);
            result_indices.resize(n * cluster_split);

            faiss::gpu::DeviceTensor<float, 2, true> queries_gpu = \
                            faiss::gpu::toDeviceTemporary<float, 2>(res_, 0, (float*)x, stream, {(int)n, index->d});

            for (int i = 0; i < query_split; i++) {
                printf("the %dth query split\n", i);
                int query_start = i * n / query_split;
                int query_end = (i + 1) * n / query_split;
                int query_cnt = query_end - query_start;

                faiss::gpu::DeviceTensor<int, 1, true> queryids_gpu = \
                            faiss::gpu::toDeviceTemporary<int, 1>(res_, 0, queryids + query_start, stream, {query_cnt});
                

                for (int j = 0; j < cluster_split; j++) {
                    printf("the %d,%d  query-cluster split\n", i, j);
                    int cluster_start = j * maxbcluster_per_query / cluster_split;
                    int cluster_end = (j + 1) * maxbcluster_per_query / cluster_split;
                    int cluster_cnt = cluster_end - cluster_start;



                    faiss::gpu::DeviceTensor<float, 3, true> best_distances(
                        res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {query_cnt, cluster_cnt, (int)k});
                    faiss::gpu::DeviceTensor<size_t, 3, true> best_indices(
                                res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {query_cnt, cluster_cnt, (int)k});
                    

                    int* cluster_query_sub = new int[query_cnt * cluster_cnt];
                    for (int i0 = query_start; i0 < query_end; i0++) {
                        memcpy (cluster_query_sub + (i0 - query_start) * cluster_cnt,
                            query_bcluster_matrix + i0 * maxbcluster_per_query + cluster_start,
                            sizeof(int) * cluster_cnt);
                    }


                    faiss::gpu::DeviceTensor<int, 2, true> query_cluster_matrix_gpu = \
                            faiss::gpu::toDeviceTemporary<int, 2>(res_, 0, cluster_query_sub, stream, {(int)query_cnt, (int)cluster_cnt});
                    cudaStreamSynchronize(stream);
                    delete[] cluster_query_sub;

                    faiss::gpu::DeviceTensor<float, 2, true>* out_distances = new faiss::gpu::DeviceTensor<float, 2, true>(
                                res_, faiss::gpu::makeDevAlloc(faiss::gpu::AllocType::Other, stream), {query_cnt, (int)k});
                    faiss::gpu::DeviceTensor<size_t, 2, true>* out_indices = new faiss::gpu::DeviceTensor<size_t, 2, true>(
                                res_, faiss::gpu::makeDevAlloc(faiss::gpu::AllocType::Other, stream), {query_cnt, (int)k});


                    
                    

                    // compute() test
                    
                    
                    t1 = elapsed();

                    faiss::gpu::runKernelCompute(
                        index->d,
                        k,
                        query_cnt,
                        cluster_cnt,
                        queryids_gpu,
                        queries_gpu,
                        query_cluster_matrix_gpu,
                        best_indices,
                        best_distances,
                        ListDataP,
                        index->ivfPipeConfig_.indicesOptions,
                        ListLength,
                        index->metric_type,
                        stream);

                    for(int q0 = query_start; q0 < query_end; q0++) {
                        result_distances[q0 * cluster_split + j] = (*out_distances)[q0 - query_start].data();
                        result_indices[q0 * cluster_split + j] = (*out_indices)[q0 - query_start].data();
                    }
                    

                    //reduce() test
                
                    faiss::gpu::runKernelReduce(
                        cluster_cnt,
                        best_distances,
                        best_indices,
                        query_cluster_matrix_gpu,
                        k,
                        ListIndexP,
                        index->ivfPipeConfig_.indicesOptions,
                        dir,
                        *out_distances,
                        *out_indices,
                        stream);

                    accumulated += elapsed() - t1;

                }
            }

            std::vector<int> cnt_per_query;
            cnt_per_query.resize(n);
            std::fill(cnt_per_query.data(), cnt_per_query.data() + n, 4);
            faiss::gpu::DeviceTensor<int, 1, true> cnt_per_query_gpu = \
                            faiss::gpu::toDeviceTemporary<int, 1>(res_, 0, cnt_per_query.data(), stream, {(int)n});


            faiss::gpu::DeviceTensor<float, 2, true> out_distances(
                        res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {(int)n, (int)k});
            faiss::gpu::DeviceTensor<size_t, 2, true> out_indices(
                                res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {(int)n, (int)k});


            t1 = elapsed();


            faiss::gpu::DeviceTensor<size_t*, 2, true> result_indices_gpu = \
                            faiss::gpu::toDeviceTemporary<size_t*, 2>(res_, 0, result_indices.data(), stream, {(int)n, cluster_split});
            faiss::gpu::DeviceTensor<float*, 2, true> result_distances_gpu = \
                            faiss::gpu::toDeviceTemporary<float*, 2>(res_, 0, result_distances.data(), stream, {(int)n, cluster_split});
        

            runKernelMerge(
                cnt_per_query_gpu,
                result_distances_gpu,
                result_indices_gpu,
                k,
                index->ivfPipeConfig_.indicesOptions,
                dir,
                out_distances,
                out_indices,
                stream);
            cudaStreamSynchronize(stream);

            accumulated += elapsed() - t1;



            auto outcome_distance = faiss::gpu::toHost<float, 2>(out_distances.data(), stream, {(int)n, (int)k});
            auto outcome_indices = faiss::gpu::toHost<size_t, 2>(out_indices.data(), stream, {(int)n, (int)k});
            cudaStreamSynchronize(stream);

            if(display) {
                for(int q0 = 0; q0 < 10; q0++) {
                    for(int q1 = 0;q1 < k; q1++) {
                        printf("%f,%zu  ",*(outcome_distance[q0][q1].data()),  *(outcome_indices[q0][q1].data()));
                    }
                    printf("\n");
                }
            }

            free(coarse_dis);
            free(ori_idx);
            free(bcluster_per_query);
            free(query_bcluster_matrix);
            free(bcluster_list);
            free(query_per_bcluster);
            free(bcluster_query_matrix);
            printf("{FINISHED IN ACCUMULATE TIME %f ms}\n", accumulated * 1000);
    }



int main() {
    //
    t0 = elapsed();
    double t1 = 0.0;

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
    int i1 = 1234 + 512;

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

    faiss::gpu::StandardGpuResources* resources = new faiss::gpu::StandardGpuResources();
    faiss::IndexIVFPipeConfig config;
    faiss::IndexIVFPipe* index = new faiss::IndexIVFPipe(d, ncentroids, config, nullptr, faiss::METRIC_L2);
    FAISS_ASSERT (config.interleavedLayout == true);

    index->train(nt, trainvecs);
    delete[] trainvecs;

    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
    t1 = elapsed();
    printf("[%.3f s] adding   ",
               t1 - t0);

    index->add(nb, database);
    for (int i = 0; i < 16; i++)
        printf("%zu\n", index->get_list_size(i));
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
   
    // display results:searching the database
    {
        float distances[20][5];
        size_t labels[20][5];

        index->set_nprobe(5);
        search_demo(resources, index, 20, queries.data(), 5, distances[0], labels[0], true);
    }
    
    // display performance: searching the database
    {
        float distances[512][5];
        size_t labels[512][5];

        index->set_nprobe(512);
        search_demo(resources, index, 512, queries.data(), 5, distances[0], labels[0], false);
    }


    return 0;
}




