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
#include <faiss/pipe/PipeStructure.h>
#include <faiss/utils/utils.h>
#include <faiss/pipe/CpuIndexIVFPipe.h>

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(){
    // Set the max threads num as 8
    omp_set_num_threads(8);

    using pi = std::pair<int,int>;
    // std::vector<int> db(1024*16);
    std::vector<pi> db(1024 * 4);

    // Randomly init
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    for (int i = 0; i < db.size(); i++) {
        int con = int(distrib(rng) * 1000);
        db[i].first = con;
        db[i].second = con;
    }
    // printf("%d\n", faiss::check_openmp());
#pragma omp parallel for
    for (int i = 0; i < 8; i++){
        if (i==1)
            printf("omp is %d\n", omp_in_parallel());
    }

    auto vec = db;

    double t0 = elapsed();
    // std::sort(db.begin(), db.end(), faiss::Com<int,int>);
    faiss::multi_sort<int, int> (db.data(), db.size());
    double t1 = elapsed();
    printf("Sort Time: %.3f ms\n", (t1 - t0) * 1000);

    std::sort(vec.begin(), vec.end(), faiss::Com<int,int>);

    for (int i = 0; i < db.size(); i++){
        FAISS_ASSERT(vec[i].first == db[i].first);
    }

}