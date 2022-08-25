/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
#include <iostream>
#include <vector>
#include <list>
#include <algorithm>

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

std::vector<float *> fvecs_reads(const char* fname, size_t* d_out, size_t* n_out, int slice = 10){
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
    std::vector<float *> res;
    size_t nr = 0;
    size_t slice_size = n / slice * (d + 1);
    size_t total_size = size_t(d + 1) * size_t(n);

    for (int i = 0; i < slice; i++){
        float* x = new float[slice_size];
        nr += fread(x, sizeof(float), slice_size, f);
        for (size_t j = 0; j < n / slice; j++)
            memmove(x + j * d, x + 1 + j * (d + 1), d * sizeof(*x));
        res.push_back(x);
    }

    assert(nr == total_size || !"could not read whole file");
    fclose(f);
    return res;
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

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double inter_sec(faiss::Index::idx_t *taget, int *gt, int k){
    double res = 0.;
    for (int i = 0; i < k; i++){
        int val = taget[i];
        for (int j = 0; j < k; j++){
            if (val == gt[j]){
                res += 1.;
                break;
            }
        }
    }
    return res / k;
}

int main(int argc,char **argv){
    std::cout << argc << " arguments" <<std::endl;
    if(argc - 1 != 3){
        printf("You should at least input 3 params: the dataset name, batch size and topk\n");
        return 0;
    }
    std::string p1 = argv[1];
    std::string p2 = argv[2];
    std::string p3 = argv[3];
    int input_k = std::stoi(p3);
    int bs = std::stoi(p2);

    int ncentroids = -1;

    std::string db, train_db, query, gtI, gtD;
    int dim;
    if (input_k>100 || input_k <=0){
        printf("Input topk must be lower than or equal to 100 and greater than 0\n");
        return 0;
    }
    if (p1 == "sift"){
        db = "/workspace/data-gpu/sift/sift40M.fvecs";
        train_db = "/workspace/data/sift/sift10M/sift10M.fvecs";
        query = "/workspace/data-gpu/sift/query.fvecs";
        gtI = "/workspace/data-gpu/sift/sift40Mgti.ivecs";
        gtD = "/workspace/data-gpu/sift/sift40Mgtd.fvecs";
        dim = 128;
        ncentroids = 1879;
    }
    else if (p1 == "deep"){
        db = "/workspace/data-gpu/deep/deep50M.fvecs";
        train_db = "/workspace/data/deep/deep10M.fvecs";
        query = "/workspace/data-gpu/deep/query.fvecs";
        gtI = "/workspace/data-gpu/deep/deep50Mgti.ivecs";
        gtD = "/workspace/data-gpu/deep/deep50Mgtd.fvecs";
        dim = 96;
        ncentroids = 2863;
    }
    else if (p1 == "text"){
        db = "/workspace/data-gpu/text/text25M.fvecs";
        train_db = "/workspace/data/text/text10M.fvecs";
        query = "/workspace/data-gpu/text/query.fvecs";
        gtI = "/workspace/data-gpu/text/text25Mgti.ivecs";
        gtD = "/workspace/data-gpu/text/text25Mgtd.fvecs";
        dim = 200;
        ncentroids = 1326;
    }
    else{
        printf("Your input dataset is not included yet! \n");
        return 0;
    }

    auto t0 = elapsed();

    omp_set_num_threads(8);

    // size_t nBytes = 25 * 1024 * 1024 * 1024ll;
    // float *x;
    // cudaMalloc((void**)&x, nBytes);

    int dev_no = 0;
    faiss::gpu::StandardGpuResources resources;
    faiss::gpu::GpuIndexIVFFlatConfig config;
    config.device = dev_no;
    faiss::gpu::GpuIndexIVFFlat *index;

    if (p1 == "text"){
        index = new faiss::gpu::GpuIndexIVFFlat(
            &resources, dim, ncentroids, faiss::METRIC_INNER_PRODUCT, config);
    }
    else{
        index = new faiss::gpu::GpuIndexIVFFlat(
            &resources, dim, ncentroids, faiss::METRIC_L2, config);
    }

    size_t d;
    // Train the index
    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float* xt = fvecs_read(train_db.c_str(), &d, &nt);

        FAISS_ASSERT(d == dim);
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->train(nt, xt);
        delete[] xt;
    }

    // Add the data
    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);

        size_t nb, d2;
        int slice = 10;
        std::vector<float *> xbs = fvecs_reads(db.c_str(), &d2, &nb, slice);
        assert(d == d2 || !"dataset does not have same dimension as train set");

        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               nb,
               d);

        for (int i = 0; i < slice; i++){
            index->add(nb / slice, xbs[i]);
            delete[] xbs[i];
        }
    }

    size_t nq;
    float* xq;
    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read(query.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
    }

    size_t k;                // nb of results per query in the GT
    int* gt; // nq * k matrix of ground-truth nearest-neighbors

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read(gtI.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new int[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
    }

    float *gtd;
    {
        printf("[%.3f s] Loading ground truth for %ld queries\n",
               elapsed() - t0,
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        gtd = fvecs_read(gtD.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");
    }

    nq = 10000;
    // Start queries
    std::vector<float> dis(nq * input_k);
    std::vector<faiss::Index::idx_t> idx(nq * input_k);
    index->nprobe = ncentroids / 32;

    auto tt0 = elapsed();
    int i;
    for (i = 0; i < nq / bs; i++){
        index->search(bs, xq + d * (bs * i), input_k, dis.data() + input_k * (bs * i), idx.data() + input_k * (bs * i));
    }
    auto tt1 = elapsed();
    double total = tt1 - tt0;

    double acc = 0.;
    for (int j = 0; j < i * bs; j++){
        auto tmp = inter_sec(idx.data() + input_k * j, gt + k * j, input_k);
        acc += tmp;
    }
    acc /= (i * bs);
    acc *= 100;

    printf("Ave Latency : %.3f s\n", total / i);
    printf("Ave accuracy : %.1f%% \n", acc);

    // cudaFree(x);

    delete[] xq;
    delete[] gt;
    delete[] gtd;
    delete index;

    return 0;
}