/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/pipe/PipeCluster.h>
#include <faiss/gpu/PipeGpuResources.h>
#include <faiss/pipe/PipeStructure.h>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// #define gpumem
#ifdef gpumem
int main(){
    omp_set_num_threads(8);
    double t0, t1;
    int nlist = 1024, d = 128;
    std::vector<int> sizes;
    std::vector<float *> pointer;
    std::vector<int *> ids;
    for (int i = 0; i < nlist; i++){
        float *p;
        int *in;
        int sz = (i%2 ==0 ? 256*32: 512*32);
        p = (float*) malloc(sz * d * sizeof(float));
        in = (int*) malloc(sz * sizeof(int));
        sizes.push_back(sz);
        pointer.push_back(p);
        ids.push_back(in);
    }

    t0 = elapsed();
    faiss::PipeCluster *pc = new faiss::PipeCluster(nlist, d, sizes, pointer, ids, true);
    printf("Nlist after balaced: %d %d, Time: %.3f s\n", pc->bnlist, 
        pc->bcs * pc->bnlist, elapsed() - t0);
    
    faiss::gpu::PipeGpuResources pg;
    pg.setMaxDeviceSize((pc->bcs * pc->bnlist)*d*sizeof(float));
    pg.setPageSize(pc->bcs*d*sizeof(float));
    pg.initializeForDevice(0, pc);
    sleep(5);
    t0 = elapsed();
    faiss::gpu::MemBlock mb = pg.allocMemory(100);
    for(int i= 0; i < mb.pages.size(); i++){
        // pc->addGlobalCount(i, i);
        pc->setonDevice(i, mb.pages[i], true);
    }
    for(int i = 0; i < mb.pages.size(); i++){
        pg.pageinfo[mb.pages[i]] = 1;
        // pc->setPinnedonDevice(i, mb.pages[i], true); // test if pinned api modify LRU_TREE
    }
    t1 = elapsed();
    printf("Alloc Time: %f ms\n", (t1 - t0)*1000);
    t0 = elapsed();
    mb = pg.allocMemory(100);
    t1 = elapsed();
    printf("Alloc Time: %f ms\n", (t1 - t0)*1000);
    auto vec = mb.pages;
    std::cout << "Valid: " << mb.valid << "\n";
    for (int i= 0; i < vec.size(); i++)
        std::cout << vec[i] << " ";
    std::cout << "\n";
    delete pc;
}
#else
int main(){

    omp_set_num_threads(8);
    double t0, t1;
    int nlist = 1024, d = 128;
    std::vector<int> sizes;
    std::vector<float *> pointer;
    std::vector<int *> ids;
    for (int i = 0; i < nlist; i++){
        float *p;
        int *in;
        int sz = (i%2 ==0 ? 256: 512);
        p = (float*) malloc(sz * d * sizeof(float));
        in = (int*) malloc(sz * sizeof(int));
        sizes.push_back(sz);
        pointer.push_back(p);
        ids.push_back(in);
    }

    t0 = elapsed();
    faiss::PipeCluster *pc = new faiss::PipeCluster(nlist, d, sizes, pointer, ids, true);
    printf("Nlist after balaced: %d %d, Time: %.3f s\n", pc->bnlist, 
        pc->bcs * pc->bnlist, elapsed() - t0);

    std::cout << pc->PinTempStatus() << "\n";

    float *pointer1 = (float *)pc->allocPinTemp(1024);

    std::cout << pc->PinTempStatus() << "\n";

    pc->freePinTemp(pointer1, 1024);
    // pc->freePinTemp(pointer1, 2048); // Cause failure

    std::cout << pc->PinTempStatus() << "\n";

    pointer1 = (float *)pc->allocPinTemp(32*1024);

    std::cout << pc->PinTempStatus() << "\n";

    delete pc;

}
#endif