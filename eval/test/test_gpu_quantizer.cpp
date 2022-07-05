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
#include <faiss/pipe/IndexIVFPipe.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    //
    double t0 = elapsed();
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
    printf("[%.3f s] Generating %ld vectors in %dD for training.    ",
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
    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
    t1 = elapsed();

    // remember a few elements from the database as queries
    size_t nq;
    std::vector<float> queries;
        
    int i0 = 1234;
    int i1 = 1234 + 512;

    nq = i1 - i0;

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

    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
    t1 = elapsed();
    printf("[%.3f s] training   ",
               t1 - t0);

    faiss::IndexIVFPipeConfig config;
    faiss::IndexIVFPipe* index = new faiss::IndexIVFPipe(
            d, ncentroids, config, faiss::METRIC_L2);
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
    printf("Balance finishing %d\n", index->pipe_cluster->bnlist);
    sleep(1);
    
    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
   


    // display results:searching the database

    {
        int k = 5;
        nq = 10;
        index->set_nprobe(5);
        t1 = elapsed();
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the GPU index   ",
               t1 - t0,
               k,
               nq);
        fflush(stdout);

        float* coarse_dis;
        int* ori_idx;
        int64_t* ori_offset;
        size_t* bcluster_per_query;
        size_t actual_nprobe;
        int* query_bcluster_matrix;
        size_t maxbcluster_per_query;
        int bcluster_cnt;
        int* bcluster_list;
        int* query_per_bcluster;
        int maxquery_per_bcluster;
        int* bcluster_query_matrix;

        index->sample_list(nq, queries.data(), &coarse_dis, &ori_idx, &ori_offset,\
             &bcluster_per_query, &actual_nprobe, &query_bcluster_matrix, &maxbcluster_per_query,\
             &bcluster_cnt, &bcluster_list, &query_per_bcluster, &maxquery_per_bcluster,\
             &bcluster_query_matrix);

    
        printf("{FINISHED in %.3f ms}\n", (elapsed() - t1) * 1000);

        // Display the query-bcluster matrix
        printf("[%.3f s] Query results (vector ids, then distances)(only display 10 queries):\n",
               elapsed() - t0);

        printf("maxbcluster_per_query:%zu, actual_nprobe:%zu\n", maxbcluster_per_query, actual_nprobe);
    
        for (int i = 0; i < 10; i++) {
            printf("query %2d: width %zu \n", i, bcluster_per_query[i]);
            for(int j = 0; j < actual_nprobe; j++) {
                printf("%d cluster::offset %ld,dis:%f   ", ori_idx[j + i * actual_nprobe], ori_offset[j + i * actual_nprobe], coarse_dis[j + i * actual_nprobe]);
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

        free(coarse_dis);
        free(ori_idx);
        free(ori_offset);
        free(bcluster_per_query);
        free(query_bcluster_matrix);
        free(bcluster_list);
        free(query_per_bcluster);
        free(bcluster_query_matrix);

    }
    int totalit = 1;//10;
    //double ave = 0.0;
    // display performance: searching the database
    for(int it = 0; it < totalit; it++)
    {
        int k = 5;
        index->set_nprobe(64);
        nq = 512;
        t1 = elapsed();
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the GPU index   ",
               t1 - t0,
               k,
               nq);
        fflush(stdout);

        float* coarse_dis;
        int* ori_idx;
        int64_t* ori_offset;
        size_t* bcluster_per_query;
        size_t actual_nprobe;
        int* query_bcluster_matrix;
        size_t maxbcluster_per_query;
        int bcluster_cnt;
        int* bcluster_list;
        int* query_per_bcluster;
        int maxquery_per_bcluster;
        int* bcluster_query_matrix;

        index->sample_list(nq, queries.data(), &coarse_dis, &ori_idx, &ori_offset,\
             &bcluster_per_query, &actual_nprobe, &query_bcluster_matrix, &maxbcluster_per_query,\
             &bcluster_cnt, &bcluster_list, &query_per_bcluster, &maxquery_per_bcluster,\
             &bcluster_query_matrix);

        free(coarse_dis);
        free(ori_idx);
        free(ori_offset);
        free(bcluster_per_query);
        free(query_bcluster_matrix);
        free(bcluster_list);
        free(query_per_bcluster);
        free(bcluster_query_matrix);

        printf("{FINISHED in %.3f ms}\n", (elapsed() - t1) * 1000);
        //ave += (elapsed() - t1) * 1000;

    }
    //printf("%f\n", ave / float(totalit));

    return 0;
}


