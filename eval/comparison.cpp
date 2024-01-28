/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/time.h>

#include <omp.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/IndexIVF.h>

#define DC(classname) classname* ix = dynamic_cast<classname*>(index)

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
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

// ./comparison 256 10 (bs, topk)
int main(int argc,char **argv){
    std::cout << argc << " arguments" <<std::endl;
    if(argc - 1 != 2){
        printf("You should at least input 2 params: batch size and topk\n");
        return 0;
    }
    // fix the dataset sift
    std::string db = "/workspace/data/sift/sift10M/sift10M.fvecs";
    std::string train_db = db;
    std::string query = "/workspace/data/sift/sift10M/query.fvecs";
    std::string gtI = "/workspace/data/sift/sift10M/idx.ivecs";
    std::string gtD = "/workspace/data/sift/sift10M/dis.fvecs";

    int dim = 128;

    std::string p1 = argv[1];
    std::string p2 = argv[2];

    int input_k = std::stoi(p2);
    int bs = std::stoi(p1);

    auto t0 = elapsed();

    omp_set_num_threads(8);
    int ncentroids = 1024;

    std::string index_c = "IVF" + std::to_string(ncentroids) + ",Flat";
    faiss::Index* index;

    size_t d;
    // Train the index
    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float* xt = fvecs_read(train_db.c_str(), &d, &nt);

        FAISS_ASSERT(d == dim);

        index = faiss::index_factory(d, index_c.c_str());
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);

        index->train(nt, xt);
        delete[] xt;
    }

    // Add the data
    {
        printf("[%.3f s] Loading database\n", elapsed() - t0);
        size_t nb, d2;
        float* xb = fvecs_read(db.c_str(), &d2, &nb);

        FAISS_ASSERT(d2 == dim);

        printf("[%.3f s] Indexing database, size %ld*%ld\n",
               elapsed() - t0,
               nb,
               d);
        index->add(nb, xb);

        delete[] xb;
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

        // load ground-truth
        size_t nq2;
        gtd = fvecs_read(gtD.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");
    }

    if (bs <= 1024)
        nq = 1024;
    else
        nq = 2048;

    auto tt0 = elapsed();

    if (DC(faiss::IndexIVF)){
        ix->nprobe = 18;
    }

    omp_set_num_threads(8);

    faiss::Index::idx_t* I = new faiss::Index::idx_t[nq * input_k];
    float* D = new float[nq * input_k];
    int i;
    for (i = 0; i < nq / bs; i++){
        if (i % 5 == 0)
            printf("Finish 5 batches\n");
        index -> search(bs, xq + d * (bs * i), input_k, D + input_k * (bs * i), I + input_k * (bs * i));
    }

    auto tt1 = elapsed();

    double total = tt1 - tt0;

    double acc = 0.;
    for (int j = 0; j < i * bs; j++){
        acc += inter_sec(I + input_k * j, gt + k * j, input_k);
    }

    printf("Ave Search Time : %.3f s\n", total / i);
    printf("Ave accuracy : %.1f%% \n", acc * 100 / (i*bs));

    delete[] xq;
    delete[] gt;
    delete[] gtd;
    delete index;
    return 0;
}