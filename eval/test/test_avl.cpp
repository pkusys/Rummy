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
    double t0,t1;
    std::vector<int> vec;
    for (int i = 0 ; i < 1024*16;i++)
        vec.push_back(i);
    faiss::PipeAVLTree<int,int>* tree=new faiss::PipeAVLTree<int,int>();
    t0 = elapsed();
    for(int i = 0; i < vec.size();i++){
        tree->insert(vec[i], vec[i]);
    }
    t1 = elapsed();
    printf("Insert Time: %f ms\n", (t1 - t0)*1000);

    t0 = elapsed();
    for(int i = 0; i < 1024*16; i++){
        std::pair<int, int> m = tree->minimum();
        tree->remove(m.first, m.second);
    }
    t1 = elapsed();
    printf("Remove Time: %f ms\n", (t1 - t0)*1000);
    // auto t0 = elapsed();
    // int size = 30 * 1000 * 1000;
    // int64_t *a = new int64_t[size];
    // for(int i = 0; i < size; i++)
    //     a[i] = 34;
    // int *b = new int[size];
    // for(int i = 0; i < size; i++)
    //     b[i] = a[i];
    // auto t1 = elapsed();
    // printf("Time: %f ms\n", (t1 - t0)*1000);
}