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
    std::vector<int> vec = {5,5,5,5,9,8,4,5,5,12,34};
    std::vector<int> vecv = {1,2,3,4,5,6,7,8,9,10,11};
    int ca = 4;
    faiss::HeapType t = faiss::HeapType::MAXHEAP;
    faiss::PipeHeap<int, int> heap(ca, t);
    for(int i = 0; i < vec.size(); i++){
        int m = heap.read();
        if (heap.isFull() && vec[i] >= m)
            continue;
        std::pair<int,int> p(vec[i],vecv[i]);
        heap.push(p);
        auto ret = heap.dump();
        for(int i = 0; i < ret.size(); i++){
            std::cout << ret[i] << " ";
        }
        std::cout << "\n";
    }
    
}