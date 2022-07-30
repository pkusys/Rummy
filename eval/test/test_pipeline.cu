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
#include <algorithm>

#include <omp.h>
#include <sys/time.h>
#include <unistd.h>
#include <faiss/pipe/PipeScheduler.h>

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

// for (int i = 0; i < 8; i++) 
//         if (i == 0)
//             printf("Omp works well ? : %d\n", omp_in_parallel());

int main(){
    // Set the max threads num as 8
    omp_set_num_threads(8);

 #pragma omp parallel for
     for (int i = 0; i < 8; i++){
         if (i==1)
             printf("omp is %d\n", omp_in_parallel());
     }

    // test reorder
    int cnt = 1024 * 4;

    std::vector<int> bcluster_list(cnt);
    for(int i = 0; i < cnt; i++)
        bcluster_list[i] = i;

    std::vector<int> query_per_bcluster(cnt);
    std::uniform_real_distribution<> distrib{0.0, 30.0};
    std::random_device rd;
    std::default_random_engine rng {rd()};
    int maxval = 0;
    for(int i = 0; i < cnt; i++){
        query_per_bcluster[i] = int(distrib(rng));
        maxval = std::max(maxval, query_per_bcluster[i]);
    }

    auto t0 = elapsed();
    faiss::gpu::PipeScheduler psch(nullptr, nullptr, cnt, bcluster_list.data(), 
        query_per_bcluster.data(), maxval, nullptr, false);
    auto t1 = elapsed();

    printf("Construct pipescheduler time: %.3f ms\n", t1 - t0);

    
}