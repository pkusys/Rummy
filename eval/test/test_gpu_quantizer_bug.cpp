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
#include <faiss/pipe/IndexIVFPipe.cuh>
#include <faiss/pipe/CpuIndexIVFPipe.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/index_io.h>
#include <faiss/Clustering.h>
#include <faiss/gpu/StandardGpuResources.h>

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
    /*
    printf ("[%.3f s] Begin d=%d nb=%ld nt=%nt dev_no=%d\n",
            elapsed() - t0, d, nb, nt, dev_no);
    */

    // the coarse quantizer should not be dealloced before the index
    // 4 = nb of bytes per code (d must be a multiple of this)
    // 8 = nb of bits per sub-code (almost always 8)
    faiss::IndexIVFPipeConfig config;
    config.device = dev_no;

    
    
    std::mt19937 rng;

    // training
    t1 = elapsed();
    printf("[%.3f s] Generating %ld vectors in %dD for training.    ",
           t1 - t0,
           nt,
           d);
    fflush(stdout);

    std::vector<float> trainvecs(nt * d);
    {
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nt * d; i++) {
            trainvecs[i] = distrib(rng);
        }
    }
    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
    t1 = elapsed();

    printf("[%.3f s] Building a dataset of %ld vectors to index.   ",
           t1 - t0,
           nb);
    fflush(stdout);

    std::vector<float> database(nb * d);
    {
        std::uniform_real_distribution<> distrib;
        for (size_t i = 0; i < nb * d; i++) {
            database[i] = distrib(rng);
        }
    }

    // remember a few elements from the database as queries
    size_t nq;
    std::vector<float> queries;
        
    int i0 = 1234;
    int i1 = 1234 + 512;

    nq = i1 - i0;
    queries.resize(nq * d);
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < d; j++) {
            queries[(i - i0) * d + j] = database[i * d + j];
        }
    }

    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
    t1 = elapsed();


    omp_set_num_threads(8);

    faiss::ClusteringParameters cp;
    
    
    faiss::gpu::StandardGpuResources resources;

    for(int i=0;i<10;i++){
        faiss::Clustering clus(d, ncentroids, cp);
        faiss::gpu::GpuIndexFlatL2* quantizer = new faiss::gpu::GpuIndexFlatL2(&resources, d);

        quantizer->reset();
        clus.verbose = true;
        clus.train(nt, trainvecs.data(), *quantizer);
        quantizer->is_trained = true;
        delete quantizer;
    }
    




    return 0;
}


