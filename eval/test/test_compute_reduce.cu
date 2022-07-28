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
#include <faiss/gpu/PipeGpuResources.h>
#include <faiss/gpu/impl/DistanceUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/PipeTensor.cuh>
#include <faiss/gpu/impl/IVFInterleaved.cuh>
#include <faiss/impl/FaissAssert.h>
#include <faiss/pipe/PipeKernel.cuh>
#include <thrust/device_vector.h>
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
            faiss::gpu::PipeGpuResources* pipe_res,
            faiss::gpu::StandardGpuResources* res,
            faiss::IndexIVFPipe* index,
            int n,
            const float* x,
            int k,
            float* distances,
            int* labels,
            bool display){
                
            printf("\n entering search_demo\n");
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

                printf("maxbcluster_per_query:%d, actual_nprobe:%d\n", maxbcluster_per_query, actual_nprobe);
            
                for (int i = 0; i < 10; i++) {
                    printf("query %2d: width %d \n", i, bcluster_per_query[i]);
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
                printf("[%.3f s] Query results (only display shape info):\n",
                    elapsed() - t0);

                printf("nquery:%d,maxbcluster_per_query:%d, actual_nprobe:%d\n", n, maxbcluster_per_query, actual_nprobe);

            }


            //move the cluster data to GPU. The time of this part is not measured.
            faiss::gpu::DeviceScope scope(0);
            std::vector<void*> ListDataP_vec;
            std::vector<void*> ListIndexP_vec;//fake index
            faiss::PipeCluster* pc = index->pipe_cluster;
            ListDataP_vec.resize(pc->bnlist);
            ListIndexP_vec.resize(pc->bnlist);

            for(int i = 0; i < pc->bnlist; i++) {
                void* dataP;
                void* IndexP;
                int vectorNum = pc->BCluSize[i];
                vectorNum = (vectorNum + 31) / 32 * 32;
                int bytes = vectorNum * index->d * sizeof(float);
                cudaMalloc(&dataP, bytes);
                cudaMemcpy(dataP, pc->Mem[i], bytes, cudaMemcpyHostToDevice);
                ListDataP_vec[i] = dataP;

                cudaMalloc(&IndexP, sizeof(int) * pc->BCluSize[i]);
                cudaMemcpy(IndexP, pc->Balan_ids[i], sizeof(int) * pc->BCluSize[i], cudaMemcpyHostToDevice);
                ListIndexP_vec[i] = IndexP;
            }
            
            auto ListLength_vec = pc->BCluSize;
            auto res_ = res->getResources().get();
            auto stream = res->getDefaultStream(0);
            
            auto d2h_stream = pipe_res->getCopyD2HStream(0);
            auto exe_stream = pipe_res->getExecuteStream(0);
            auto h2d_stream = pipe_res->getCopyH2DStream(0);
            faiss::gpu::PipeTensor<void*, 1, true> ListDataP_({pc->bnlist}, pc);
            ListDataP_.copyFrom(ListDataP_vec, h2d_stream);
            ListDataP_.setResources(pc, pipe_res);
            ListDataP_.memh2d(h2d_stream);

            void** ListDataP = ListDataP_.devicedata();

            cudaStreamSynchronize(h2d_stream);


            faiss::gpu::PipeTensor<int, 1, true> ListLength_({pc->bnlist}, pc);
            ListLength_.copyFrom(ListLength_vec, h2d_stream);
            ListLength_.setResources(pc, pipe_res);
            ListLength_.memh2d(h2d_stream);

            int* ListLength = ListLength_.devicedata();



            faiss::gpu::PipeTensor<void*, 1, true> ListIndexP_({pc->bnlist}, pc);
            ListIndexP_.copyFrom(ListIndexP_vec, h2d_stream);
            ListIndexP_.setResources(pc, pipe_res);
            ListIndexP_.memh2d(h2d_stream);

            void** ListIndexP = ListIndexP_.devicedata();



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
            printf("1okay\n");
            // The below code will cause pinned memory overflow for the second test.

            // First test without spliting.
            {
                
                // start to measure time from here on.

                faiss::gpu::PipeTensor<int, 1, true> queryids_gpu({(int)n}, pc);
                queryids_gpu.copyFrom(queryids, h2d_stream);
                queryids_gpu.setResources(pc, pipe_res);
                queryids_gpu.memh2d(h2d_stream);


                faiss::gpu::PipeTensor<float, 2, true> queries_gpu({(int)n, index->d}, pc);
                queries_gpu.copyFrom((float*)x, h2d_stream);
                queries_gpu.setResources(pc, pipe_res);
                queries_gpu.memh2d(h2d_stream);


                faiss::gpu::PipeTensor<int, 2, true> query_cluster_matrix_gpu({(int)n, (int)maxbcluster_per_query}, pc);
                query_cluster_matrix_gpu.copyFrom(query_bcluster_matrix, h2d_stream);
                query_cluster_matrix_gpu.setResources(pc, pipe_res);
                query_cluster_matrix_gpu.memh2d(h2d_stream);


                faiss::gpu::PipeTensor<float, 2, true> out_distances({(int)n, (int)k}, pc);
                out_distances.setResources(pc, pipe_res);
                out_distances.reserve();

                faiss::gpu::PipeTensor<int, 2, true> out_indices({(int)n, (int)k}, pc);
                out_indices.setResources(pc, pipe_res);
                out_indices.reserve();


                // computeReduce() test
                cudaStreamSynchronize(h2d_stream);
                t1 = elapsed();
                printf("[%.3f s]start to measure time of compute and reduce\n", t1 - t0);


                faiss::gpu::runKernelComputeReduce(
                    index->d,
                    k,
                    n,
                    maxbcluster_per_query,
                    queryids_gpu,
                    queries_gpu,
                    query_cluster_matrix_gpu,
                    ListDataP,
                    index->ivfPipeConfig_.indicesOptions,
                    ListLength,
                    ListIndexP,
                    index->metric_type,
                    dir,
                    out_distances,
                    out_indices,
                    pc,
                    pipe_res,
                    0,
                    1);

                cudaStreamSynchronize(exe_stream);
                {
                    auto cudaStatus = cudaGetLastError();
                    if (cudaStatus != cudaSuccess)
                    {
                        FAISS_THROW_FMT("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                    }
                }


                {
                    printf("{FINISHED in %.3f ms}\n", (elapsed() - t1) * 1000);


                    out_distances.memd2h(d2h_stream);
                    out_indices.memd2h(d2h_stream);
                    cudaStreamSynchronize(d2h_stream);


                    // display 10 query's results.
                    if (display) {
                        for(int q0 = 0; q0 < 10; q0++) {
                            for(int q1 = 0;q1 < k; q1++) {
                                float distance = out_distances[q0][q1];
                                int indice = out_indices[q0][q1];
                                printf("%f,%d  ", distance , indice);
                            }
                            printf("\n");
                        }
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
            std::vector<int*> result_indices;

            result_distances.resize(n * cluster_split);
            result_indices.resize(n * cluster_split);



            faiss::gpu::PipeTensor<float, 2, true> queries_gpu({(int)n, index->d}, pc);
            queries_gpu.copyFrom((float*)x, h2d_stream);
            queries_gpu.setResources(pc, pipe_res);
            queries_gpu.memh2d(h2d_stream);

            faiss::gpu::PipeTensor<int, 2, true> query_cluster_matrix_whole({(int)n, (int)maxbcluster_per_query}, pc);
            query_cluster_matrix_whole.copyFrom(query_bcluster_matrix, h2d_stream);
            query_cluster_matrix_whole.setResources(pc, pipe_res);
            query_cluster_matrix_whole.memh2d(h2d_stream);

            std::vector<faiss::gpu::PipeTensor<float, 2, true>*> distance_temp_vec;
            std::vector<faiss::gpu::PipeTensor<int, 2, true>*> indice_temp_vec;
            distance_temp_vec.resize(query_split * cluster_split);
            indice_temp_vec.resize(query_split * cluster_split);

            faiss::gpu::PipeTensor<float, 2, true> out_distances({(int)n, (int)k}, pc);
            out_distances.setResources(pc, pipe_res);
            out_distances.reserve();                

            faiss::gpu::PipeTensor<int, 2, true> out_indices({(int)n, (int)k}, pc);
            out_indices.setResources(pc, pipe_res);
            out_indices.reserve();  



            t1 = elapsed();

            faiss::gpu::PipeTensor<int, 1, true> queryids_gpu_whole({(int)n}, pc);
            queryids_gpu_whole.copyFrom(queryids, h2d_stream);
            queryids_gpu_whole.setResources(pc, pipe_res);
            queryids_gpu_whole.memh2d(h2d_stream);
            cudaStreamSynchronize(h2d_stream);

            for (int i = 0; i < query_split; i++) {
                int query_start = i * n / query_split;
                int query_end = (i + 1) * n / query_split;
                int query_cnt = query_end - query_start;

                auto queryids_gpu = queryids_gpu_whole.narrow(0, query_start, query_cnt);

                for (int j = 0; j < cluster_split; j++) {
                    //printf("the %d,%d  query-cluster split\n", i, j);

                    int cluster_start = j * maxbcluster_per_query / cluster_split;
                    int cluster_end = (j + 1) * maxbcluster_per_query / cluster_split;
                    int cluster_cnt = cluster_end - cluster_start;

                    
                    auto query_cluster_matrix_gpu_ = query_cluster_matrix_whole.narrow(0, query_start, query_cnt);
                    auto query_cluster_matrix_gpu = query_cluster_matrix_gpu_.narrow(1, cluster_start, cluster_cnt);

                    faiss::gpu::PipeTensor<float, 2, true>* out_distance = new faiss::gpu::PipeTensor<float, 2, true>({query_cnt, (int)k}, pc);
                    out_distance->setResources(pc, pipe_res);
                    out_distance->reserve();

                    faiss::gpu::PipeTensor<int, 2, true>* out_indice = new faiss::gpu::PipeTensor<int, 2, true>({query_cnt, (int)k}, pc);
                    out_indice->setResources(pc, pipe_res);
                    out_indice->reserve();

                    indice_temp_vec[i * cluster_split + j] = out_indice;
                    distance_temp_vec[i * cluster_split + j] = out_distance;



                    for(int q0 = query_start; q0 < query_end; q0++) {
                        result_distances[q0 * cluster_split + j] = (*out_distance)(q0 - query_start).data();
                        result_indices[q0 * cluster_split + j] = (*out_indice)(q0 - query_start).data();
                    }

                    
                    // computeReduce() test

                    faiss::gpu::runKernelComputeReduce(
                        index->d,
                        k,
                        query_cnt,
                        cluster_cnt,
                        queryids_gpu,
                        queries_gpu,
                        query_cluster_matrix_gpu,
                        ListDataP,
                        index->ivfPipeConfig_.indicesOptions,
                        ListLength,
                        ListIndexP,
                        index->metric_type,
                        dir,
                        *out_distance,
                        *out_indice,
                        pc,
                        pipe_res,
                        0,
                        2);

                }
            }

            cudaStreamSynchronize(exe_stream);
            auto cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess)
            {
                FAISS_THROW_FMT("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            }

            {

                std::vector<int> cnt_per_query;
                cnt_per_query.resize(n);
                std::fill(cnt_per_query.data(), cnt_per_query.data() + n, 4);

                faiss::gpu::PipeTensor<int, 1, true> cnt_per_query_gpu({(int)n}, pc);
                cnt_per_query_gpu.copyFrom(cnt_per_query, h2d_stream);
                cnt_per_query_gpu.setResources(pc, pipe_res);
                cnt_per_query_gpu.memh2d(h2d_stream);



            
                faiss::gpu::PipeTensor<int*, 2, true> result_indices_gpu({(int)n, cluster_split}, pc);
                result_indices_gpu.copyFrom(result_indices, h2d_stream);
                result_indices_gpu.setResources(pc, pipe_res);
                result_indices_gpu.memh2d(h2d_stream);


                faiss::gpu::PipeTensor<float*, 2, true> result_distances_gpu({(int)n, cluster_split}, pc);
                result_distances_gpu.copyFrom(result_distances, h2d_stream);
                result_distances_gpu.setResources(pc, pipe_res);
                result_distances_gpu.memh2d(h2d_stream);

                cudaStreamSynchronize(h2d_stream);


                faiss::gpu::runKernelMerge(
                    cnt_per_query_gpu,
                    result_distances_gpu,
                    result_indices_gpu,
                    k,
                    index->ivfPipeConfig_.indicesOptions,
                    dir,
                    out_distances,
                    out_indices,
                    exe_stream);
                cudaStreamSynchronize(exe_stream);

                accumulated += elapsed() - t1;

                auto cudaStatus = cudaGetLastError();
                if (cudaStatus != cudaSuccess)
                {
                    FAISS_THROW_FMT("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                }

                out_distances.memd2h(d2h_stream);
                out_indices.memd2h(d2h_stream);
                cudaStreamSynchronize(d2h_stream);

                if (display) {
                    // display 10 query's results.
                    for(int q0 = 0; q0 < 10; q0++) {
                        for(int q1 = 0;q1 < k; q1++) {
                            float distance = out_distances[q0][q1];
                            int indice = out_indices[q0][q1];
                            printf("%f,%d  ", distance, indice);
                        }
                        printf("\n");
                    }
                }
                

            }


            for(int i = query_split * cluster_split - 1; i >= 0; i--) {
                delete indice_temp_vec[i];
                delete distance_temp_vec[i];
            }



            printf("{FINISHED IN ACCUMULATE TIME %f ms}\n", accumulated * 1000);

            printf("\ntest origional\n");

            faiss::gpu::DeviceTensor<float, 2, true> queries_gpu_device = \
                            faiss::gpu::toDeviceTemporary<float, 2>(res_, 0, (float*)x, stream, {(int)n, index->d});

            faiss::gpu::DeviceTensor<float, 2, true> outDistances(
                        res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {(int)n, (int)k});
            faiss::gpu::DeviceTensor<faiss::Index::idx_t, 2, true> outIndices(
                                res_, faiss::gpu::makeTempAlloc(faiss::gpu::AllocType::Other, stream), {(int)n, (int)k});
            faiss::gpu::DeviceTensor<float, 3, true> residualBase(res_, faiss::gpu::makeTempAlloc( faiss::gpu::AllocType::Other, stream), {(int)n, (int)maxbcluster_per_query, index->d});
                            faiss::gpu::DeviceTensor<int, 2, true> query_cluster_matrix_gpu = \
                        faiss::gpu::toDeviceTemporary<int, 2>(res_, 0, query_bcluster_matrix, stream, {(int)n, (int)maxbcluster_per_query});
       

            /// test origional kernel function


            thrust::device_vector<void*> deviceListDataPointers_;

            /// Device representation of all inverted list index pointers
            /// id -> data
            thrust::device_vector<void*> deviceListIndexPointers_;

            /// Device representation of all inverted list lengths
            /// id -> length in number of vectors
            thrust::device_vector<int> deviceListLengths_;



            deviceListDataPointers_.clear();
            deviceListIndexPointers_.clear();
            deviceListLengths_.clear();


            deviceListDataPointers_.resize(pc->bnlist, nullptr);
            deviceListIndexPointers_.resize(pc->bnlist, nullptr);
            deviceListLengths_.resize(pc->bnlist, 0);

            for (int i = 0;i < pc->bnlist; i++) {
                deviceListDataPointers_[i] = ListDataP_vec[i];
                deviceListIndexPointers_[i] = ListIndexP_vec[i];
                deviceListLengths_[i] = pc->BCluSize[i];
                ListDataP_vec.resize(pc->bnlist);
                ListIndexP_vec.resize(pc->bnlist);
            }


            t1 = elapsed();

            faiss::gpu::runIVFInterleavedScan(
                queries_gpu_device,
                query_cluster_matrix_gpu,
                deviceListDataPointers_,
                deviceListIndexPointers_,
                index->ivfPipeConfig_.indicesOptions,
                deviceListLengths_,
                k,
                index->metric_type,
                false,
                residualBase,
                nullptr,
                outDistances,
                outIndices,
                res_);

            cudaStreamSynchronize(stream);
            printf("{FINISHED in %.3f ms}\n", (elapsed() - t1) * 1000);

            printf("%d\n", pc->bnlist);

            free(coarse_dis);
            free(ori_idx);
            free(bcluster_per_query);
            free(query_bcluster_matrix);
            free(bcluster_list);
            free(query_per_bcluster);
            free(bcluster_query_matrix);

            auto hostDistance = faiss::gpu::toHost<float, 2>(outDistances.data(), res_->getDefaultStream(index->ivfPipeConfig_.device),
                {(int)n, (int)k});
            
            auto hostIndice = faiss::gpu::toHost<faiss::Index::idx_t, 2>(outIndices.data(), res_->getDefaultStream(index->ivfPipeConfig_.device),
                {(int)n, (int)k});
            
            bool correctness = true;
            
            // display 10 query's results.
            if(display) {
                for(int q0 = 0; q0 < 10; q0++) {
                    for(int q1 = 0;q1 < k; q1++) {
                        float distance = hostDistance[q0][q1];
                        int indice = hostIndice[q0][q1];
                        printf("%f,%d  ", distance, indice);                        
                    }
                    printf("\n");
                }
            }
            

            
            for(int q0 = 0; q0 < n; q0++) {
                int subco = 0;
                for(int q1 = 0;q1 < k; q1++) {
                    float distance = hostDistance[q0][q1];
                    int indice = hostIndice[q0][q1];
                    if (out_distances[q0][q1] != distance || out_indices[q0][q1] != indice) {
                        correctness = false;
                        subco += 1;
                    }
                }
                if(subco != 0) {
                    printf("wrong at query %d,wrongcnt %d;  ", q0, subco);
                }
            }
            
            if (!correctness) {
                printf("wrong!!!\n");
            }
            else {
                printf("right\n");
            }
            
            
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
   
    auto pc = index->pipe_cluster;

    faiss::gpu::PipeGpuResources* pipe_res = new faiss::gpu::PipeGpuResources();
    pipe_res->initializeForDevice(0, pc);

    // display results:searching the database
    {
        float distances[20][80];
        int labels[20][80];

        index->set_nprobe(5);
        search_demo(pipe_res, resources, index, 20, queries.data(), 40, distances[0], labels[0], false);
    }
    // display performance: searching the database
    {
        float distances[512][10];
        int labels[512][10];

        index->set_nprobe(512);
        search_demo(pipe_res, resources, index, 128, queries.data(), 10, distances[0], labels[0], false);
    }


    return 0;
}



