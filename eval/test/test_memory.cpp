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

int main(){
    // Test Heap
    // std::vector<int> vec = {5,5,5,5,9,8,4,5,5,12,34};
    // std::vector<int> vecv = {1,2,3,4,5,6,7,8,9,10,11};
    // int ca = 4;
    // faiss::HeapType t = faiss::HeapType::MAXHEAP;
    // faiss::PipeHeap<int, int> heap(ca, t);
    // for(int i = 0; i < vec.size(); i++){
    //     int m = heap.read();
    //     if (heap.isFull() && vec[i] >= m)
    //         continue;
    //     std::pair<int,int> p(vec[i],vecv[i]);
    //     heap.push(p);
    //     auto ret = heap.dump();
    //     for(int i = 0; i < ret.size(); i++){
    //         std::cout << ret[i] << " ";
    //     }
    //     std::cout << "\n";
    // }
    
    double t0, t1;
    int nlist = 1024, d = 128;
    std::vector<int> sizes;
    std::vector<float *> pointer;
    for (int i = 0; i < nlist; i++){
        float *p;
        int sz = (i%2 ==0 ? 256: 512);
        p = (float*) malloc(sz * d * sizeof(float));
        sizes.push_back(sz);
        pointer.push_back(p);
    }

    faiss::PipeCluster *pc = new faiss::PipeCluster(nlist, d, sizes, pointer, true);
    printf("Nlist after balaced: %d %d\n", pc->bnlist, pc->bcs * pc->bnlist);
    
    faiss::gpu::PipeGpuResources pg;
    pg.setMaxDeviceSize((pc->bcs * pc->bnlist)*d*sizeof(float));
    pg.setPageSize(pc->bcs*d*sizeof(float));
    pg.initializeForDevice(0, pc);
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