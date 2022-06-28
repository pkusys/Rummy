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

#include <faiss/IndexFlat.h>
#include <faiss/pipe/CpuIndexIVFPipe.h>
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

    faiss::CpuIndexIVFPipe index(d, ncentroids);

    std::mt19937 rng;

    { // training
        t1 = elapsed();
        printf("[%.3f s] Generating %ld vectors in %dD for training.    ",
               t1 - t0,
               nt,
               d);
        fflush(stdout);

        std::vector<float> trainvecs(nt * d);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nt * d; i++) {
            trainvecs[i] = distrib(rng);
        }
        printf("{FINISHED in %.3f s}\n", elapsed() - t1);
        t1 = elapsed();
        printf("[%.3f s] Training the index.   ", t1 - t0);
        fflush(stdout);
        index.verbose = true;

        index.train(nt, trainvecs.data());

        printf("{FINISHED in %.3f s}\n", elapsed() - t1);
        t1 = elapsed();

    }

    
    /*
    { // I/O demo
        const char* outfilename = "/tmp/index_trained.faissindex";
        printf("[%.3f s] storing the pre-trained index to %s\n",
               elapsed() - t0,
               outfilename);

        write_index(&index, outfilename);
    }
    */


    size_t nq;
    std::vector<float> queries;

    { // populating the database
        printf("[%.3f s] Building a dataset of %ld vectors to index.   ",
               t1 - t0,
               nb);
        fflush(stdout);

        std::vector<float> database(nb * d);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }

        printf("{FINISHED in %.3f s}\n", elapsed() - t1);
        t1 = elapsed();
        printf("[%.3f s] Adding the vectors to the index.   ", t1 - t0);
        fflush(stdout);

        index.add(nb, database.data());


        printf("{FINISHED in %.3f s}\n", elapsed() - t1);
        t1 = elapsed();
        printf("[%.3f s] imbalance factor: %g.   ",
               t1 - t0,
               index.invlists->imbalance_factor());
        fflush(stdout);

        // remember a few elements from the database as queries
        int i0 = 1234;
        int i1 = 1234 + 512;

        nq = i1 - i0;
        queries.resize(nq * d);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < d; j++) {
                queries[(i - i0) * d + j] = database[i * d + j];
            }
        }
    }

    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
    t1 = elapsed();

    printf("[%.3f s] Balancing the clusters   ", t1 - t0);
    fflush(stdout);

    index.balance();

    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
    t1 = elapsed();


    index.set_nprobe(5);
    omp_set_num_threads(8);
    { // searching the database
        int k = 5;
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index   ",
               t1 - t0,
               k,
               nq);
        fflush(stdout);

        float* coarse_dis;
        int64_t* ori_idx;
        int64_t* ori_offset;
        size_t* bcluster_cnt;
        size_t actual_nprobe;
        int* pipe_cluster_idx;
        size_t batch_width;

        index.sample_list(nq, queries.data(), &coarse_dis, &ori_idx, &ori_offset, &bcluster_cnt, &actual_nprobe, &pipe_cluster_idx, &batch_width);


        printf("{FINISHED in %.3f s}\n", elapsed() - t1);
        t1 = elapsed();
        printf("[%.3f s] Query results (vector ids, then distances)(only display 10 queries):\n",
               elapsed() - t0);

        printf("batch_width:%zu, actual_nprobe:%zu\n", batch_width, actual_nprobe);
    
        for (int i = 0; i < 10; i++) {
            printf("query %2d: width %zu \n", i, bcluster_cnt[i]);
            for(int j = 0; j < actual_nprobe; j++) {
                printf("%ld cluster::offset %ld,dis:%f   ", ori_idx[j + i * actual_nprobe], ori_offset[j + i * actual_nprobe], coarse_dis[j + i * actual_nprobe]);
            }
            printf("\n");
            
        }

        printf("balanced clusters:\n");
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < batch_width; j++) {
                printf("%d  ", pipe_cluster_idx[j + i * batch_width]);
            }
        printf("\n");
        }
    
    }
    
    return 0;
}
