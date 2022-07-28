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
#include <faiss/pipe/PipeProfiler.cuh>
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

static double tstart;


int main() {
    //
    tstart = elapsed();
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
            elapsed() - tstart, d, nb, nt, dev_no);


    // the coarse quantizer should not be dealloced before the index
    // 4 = nb of bytes per code (d must be a multiple of this)
    // 8 = nb of bits per sub-code (almost always 8)


    std::mt19937 rng;

    // training
    t1 = elapsed();
    printf("[%.3f s] Generating %ld vectors in %dD for training.\n",
           t1 - tstart,
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
           t1 - tstart,
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
               t1 - tstart);

    faiss::gpu::StandardGpuResources* resources = new faiss::gpu::StandardGpuResources();
    faiss::IndexIVFPipeConfig config;
    faiss::IndexIVFPipe* index = new faiss::IndexIVFPipe(d, ncentroids, config, nullptr, faiss::METRIC_L2);
    FAISS_ASSERT (config.interleavedLayout == true);

    index->train(nt, trainvecs);
    delete[] trainvecs;

    printf("{FINISHED in %.3f s}\n", elapsed() - t1);
    t1 = elapsed();
    printf("[%.3f s] adding   ",
               t1 - tstart);

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


    t1 = elapsed();
    printf("[%.3f s] profiling   ", t1 - tstart);


    faiss::gpu::PipeProfiler* profiler = new faiss::gpu::PipeProfiler(pipe_res, pc, index);

    profiler->train();

    printf("{FINISHED in %.3f s}\n", elapsed() - t1);

    return 0;
}



