/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <assert.h>

#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/index_io.h>

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// int main() {
//     omp_set_num_threads(16);
//     double t0 = elapsed();

//     // dimension of the vectors to index
//     int d = 128;

//     // size of the database we plan to index
//     size_t nb = 10*1000 * 1000 ;

//     // make a set of nt training vectors in the unit cube
//     // (could be the database)
//     size_t nt = 100 * 1000;

//     int dev_no = 0;
//     /*
//     printf ("[%.3f s] Begin d=%d nb=%ld nt=%nt dev_no=%d\n",
//             elapsed() - t0, d, nb, nt, dev_no);
//     */
//     // a reasonable number of centroids to index nb vectors
//     int ncentroids = int(4 * sqrt(nb));

//     faiss::gpu::StandardGpuResources resources;

//     // the coarse quantizer should not be dealloced before the index
//     // 4 = nb of bytes per code (d must be a multiple of this)
//     // 8 = nb of bits per sub-code (almost always 8)
//     faiss::gpu::GpuIndexFlatConfig config;
//     config.device = dev_no;

//     faiss::gpu::GpuIndexFlat index(
//             &resources, d, faiss::METRIC_L2, config);

//     std::mt19937 rng;

//     size_t nq;
//     std::vector<float> queries;
//     // sleep(10);
//     { // populating the database
//         printf("[%.3f s] Import a dataset of %ld vectors to index\n",
//                elapsed() - t0,
//                nb);
//         size_t nn,nnn;
//         float *database = fvecs_read("/workspace/data/sift/sift10M/sift10M.fvecs", &nn, &nnn);
//         printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);
//         double t0 = elapsed();
//         index.add(nb, database);
//         double t1 = elapsed();
//         printf("\n\nTransmission Time:%.3f\n\n", t1-t0);

//         printf("[%.3f s] done\n", elapsed() - t0);

//         // remember a few elements from the database as queries
//         int i0 = 1000;
//         int i1 = 1064;

//         nq = i1 - i0;
//         queries.resize(nq * d);
//         for (int i = i0; i < i1; i++) {
//             for (int j = 0; j < d; j++) {
//                 queries[(i - i0) * d + j] = database[i * d + j];
//             }
//         }
//     }
//     sleep(5);
//     { // searching the database
//         int k = 10;
//         printf("[%.3f s] Searching the %d nearest neighbors "
//                "of %ld vectors in the index\n",
//                elapsed() - t0,
//                k,
//                nq);

//         std::vector<faiss::Index::idx_t> nns(k * nq);
//         std::vector<float> dis(k * nq);
//         double t0 = elapsed();
//         for (int ij = 0; ij < 1; ij++){
//             index.search(nq, queries.data(), k, dis.data(), nns.data());
//         }
//         double t1 = elapsed();
//         printf("\n\nSearch Time: %.3f\n\n\n", t1-t0);

//         // printf("[%.3f s] Query results (vector ids, then distances):\n",
//         //        elapsed() - t0);

//         // for (int i = 0; i < 10; i++) {
//         //     printf("query %2d: ", i);
//         //     for (int j = 0; j < k; j++) {
//         //         printf("%7ld ", nns[j + i * k]);
//         //     }
//         //     printf("\n     dis: ");
//         //     for (int j = 0; j < k; j++) {
//         //         printf("%7g ", dis[j + i * k]);
//         //     }
//         //     printf("\n");
//         // }

//         // printf("note that the nearest neighbor is not at "
//         //        "distance 0 due to quantization errors\n");
//     }

//     return 0;
// }


int main() {
    omp_set_num_threads(16);
    double t0 = elapsed();

    // dimension of the vectors to index
    int d = 128;

    // size of the database we plan to index
    size_t nb = 10*1000 * 1000 ;

    // make a set of nt training vectors in the unit cube
    // (could be the database)
    size_t nt = 100 * 1000;

    int dev_no = 0;
    /*
    printf ("[%.3f s] Begin d=%d nb=%ld nt=%nt dev_no=%d\n",
            elapsed() - t0, d, nb, nt, dev_no);
    */
    // a reasonable number of centroids to index nb vectors
    int ncentroids = 1024;

    faiss::gpu::StandardGpuResources resources;

    // the coarse quantizer should not be dealloced before the index
    // 4 = nb of bytes per code (d must be a multiple of this)
    // 8 = nb of bits per sub-code (almost always 8)
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = dev_no;

    faiss::gpu::GpuIndexIVFFlat index(
            &resources, d, ncentroids, faiss::METRIC_L2, config);

    std::mt19937 rng;

    size_t nq;
    std::vector<float> queries;
    sleep(1);

    { // training
        printf("[%.3f s] Generating %ld vectors in %dD for training\n",
               elapsed() - t0,
               nt,
               d);

        std::vector<float> trainvecs(nt * d);
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nt * d; i++) {
            trainvecs[i] = distrib(rng);
        }

        printf("[%.3f s] Training the index\n", elapsed() - t0);
        index.verbose = true;

        index.train(nt, trainvecs.data());
    }

    // sleep(10);
    { // populating the database
        printf("[%.3f s] Import a dataset of %ld vectors to index\n",
               elapsed() - t0,
               nb);
        size_t nn,nnn;
        float *database = fvecs_read("/workspace/data/sift/sift10M/sift10M.fvecs", &nn, &nnn);
        printf("[%.3f s] Adding the vectors to the index\n", elapsed() - t0);
        double t0 = elapsed();
        index.add(nb, database);
        double t1 = elapsed();
        printf("\n\nTransmission Time:%.3f\n\n", t1-t0);

        printf("[%.3f s] done\n", elapsed() - t0);

        // remember a few elements from the database as queries
        int i0 = 1000;
        int i1 = 1064;

        nq = i1 - i0;
        queries.resize(nq * d);
        for (int i = i0; i < i1; i++) {
            for (int j = 0; j < d; j++) {
                queries[(i - i0) * d + j] = database[i * d + j];
            }
        }
    }
    sleep(1);
    { // searching the database
        int k = 10;
        printf("[%.3f s] Searching the %d nearest neighbors "
               "of %ld vectors in the index\n",
               elapsed() - t0,
               k,
               nq);

        std::vector<faiss::Index::idx_t> nns(k * nq);
        std::vector<float> dis(k * nq);
        double t0 = elapsed();
        index.nprobe = 1;
        printf("%d\n", index.nprobe);
        for (int i =0;i<1;i++)
            index.search(nq, queries.data(), k, dis.data(), nns.data());
        double t1 = elapsed();
        printf("\n\nSearch Time: %.3f\n\n\n", t1-t0);

        // printf("[%.3f s] Query results (vector ids, then distances):\n",
        //        elapsed() - t0);

        // for (int i = 0; i < 10; i++) {
        //     printf("query %2d: ", i);
        //     for (int j = 0; j < k; j++) {
        //         printf("%7ld ", nns[j + i * k]);
        //     }
        //     printf("\n     dis: ");
        //     for (int j = 0; j < k; j++) {
        //         printf("%7g ", dis[j + i * k]);
        //     }
        //     printf("\n");
        // }

        // printf("note that the nearest neighbor is not at "
        //        "distance 0 due to quantization errors\n");
    }

    return 0;
}