/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/pipe/PipeCluster.h>
#include <faiss/gpu/PipeGpuResources.h>
#include <faiss/pipe/PipeStructure.h>
#include <faiss/gpu/utils/PipeTensor.cuh>
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
    omp_set_num_threads(8);
    double t0, t1;
    int nlist = 1024, d = 128;
    std::vector<int> sizes;
    std::vector<float *> pointer;
    std::vector<int *> ids;
    for (int i = 0; i < nlist; i++){
        float *p;
        int *in;
        int sz = (i%2 ==0 ? 256*8: 512*8);
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

    // Test PipeTensor
    // You are required to create and delete the tensor in order to adapt to the stack
    faiss::gpu::PipeTensor<int, 3>* pt = new faiss::gpu::PipeTensor<int, 3>({2,3,4}, pc);
    pt->setResources(pc, &pg);
    std::cout << pc->PinTempStatus() << "\n";
    pt->memh2d();
    std::cout << "Device ptr : " << pt->devicedata() << "\n";
    std::cout << pg.tempMemory_[0]->getSizeAvailable() << "\n";
    delete pt;
    std::cout << pc->PinTempStatus() << "\n";
    std::cout << pg.tempMemory_[0]->getSizeAvailable() << "\n";
}