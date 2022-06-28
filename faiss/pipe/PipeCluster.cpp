/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/pipe/PipeCluster.h>
#include <faiss/impl/FaissAssert.h>

#include <numeric>
#include <algorithm>
#include <cmath>
#include <string.h>
#include <iostream>

namespace faiss {

namespace {

// Default expected standard variance ratio
const float StdVarRation = 0.2;

float StdDev (std::vector<int> vec){
    // Caculate the mean value
    float sum = std::accumulate(std::begin(vec), std::end(vec), 0.0);
    float mean =  sum / vec.size();
    
    // Caculate the standard deviation
    float accum  = 0.0;
    std::for_each (std::begin(vec), std::end(vec), [&](const float d) {
            accum  += (d-mean)*(d-mean);
        });
    
    float stdev = std::sqrt(accum/(vec.size()-1));

    return stdev;
}

}

//
// PipeCluster
//

PipeCluster::PipeCluster(int nlist_, int d_, std::vector<int> & sizes, 
            std::vector<float *> & pointers){
    
    // Initialize some attributes
    nlist = nlist_;

    d = d_;

    CluSize = sizes;

    noPinnedMem = pointers;

    PinnedMem.resize(nlist);

    // Balance the clusters
    balance(StdVarRation);

    // Initialize the device memory info
    isonDevice.resize(BCluSize.size());
    isPinnedDevice.resize(BCluSize.size());
    PinnedMem.resize(BCluSize.size());
    GlobalCount.resize(BCluSize.size());
    DeviceMem.resize(BCluSize.size());

    std::fill(isonDevice.begin(), isonDevice.end(), false);
    std::fill(isPinnedDevice.begin(), isPinnedDevice.end(), false);
    std::fill(GlobalCount.begin(), GlobalCount.end(), 0);

    // Construct the OnDevice cluster
    auto tmptree = std::unique_ptr<PipeAVLTree<int,int> >
            (new PipeAVLTree<int, int>());
    
    LRUtree_ = std::move(tmptree);

    mallocPinnedMem();
}

PipeCluster::~PipeCluster(){
    // Free the allocated memory if necessary
    if (!PinnedMem.empty())
        freePinnedMem();
}

void PipeCluster::balance(const float svr){

    // The average number of the cluster sizes
    int ave = std::accumulate(CluSize.begin(), CluSize.end(), 0);
    ave /= nlist;

    // Round down to the multiple of 256
    int sta = (ave / 256) * 256;
    FAISS_ASSERT(sta >= 256);

    // The balanced cluster number
    std::vector<int> tmp;
    bool re = false;
    
    // Test if the balance number satifies the requirement
    while(sta >= 256){
        tmp.clear();
        for(int i = 0; i < CluSize.size(); i++){
            int cnum = CluSize[i];
            while(cnum >= sta){
                tmp.push_back(sta);
                cnum -= sta;
            }
            if (cnum > 0)
                tmp.push_back(cnum);
        }
        float std = StdDev(tmp);
        if (std / sta <= svr){
            std::cout << "The std dev after balancing: " 
                    << std / sta << "\n";
            re = true;
            break;
        }
        sta -= 256;
    }


    // Initialize the balanced attributes
    bcs = re ? sta : sta + 256;

    BCluSize = std::move(tmp);

    bnlist = BCluSize.size();

    int index = 0;


    for (int i = 0; i < CluSize.size(); i++){
        int cnum = CluSize[i];
        std::vector<int> val;
        while(cnum >= bcs){
            val.push_back(index++);
            cnum-=bcs;
        }
        if (cnum > 0)
            val.push_back(index++);
        
        O2Bmap[i] = std::move(val);
        O2Bcnt[i] = O2Bmap[i].size();
    }

    // Check the correctness of initialization
    FAISS_ASSERT_FMT(index == BCluSize.size(), "Index: %d, Bcluse size: %d",
            index, index == BCluSize.size());
}

void PipeCluster::mallocPinnedMem(){
    // Malloc the pinned memory and free the nopinned memory
    int index = 0;

    for (int i = 0; i < nlist; i++){
        auto balaClu = O2Bmap[i];
        int num = balaClu.size();
        float *nop = noPinnedMem[i];
        
        for(int j = 0; j < num; j++){
            size_t bytes = BCluSize[index] * d * sizeof(float);
            float *p;

            // Malloc pinned memory
            auto error = cudaMallocHost((void **) &p, bytes);
            FAISS_ASSERT_FMT(
                    error == cudaSuccess,
                    "Failed to malloc pinned memory: %d vectors (error %d %s)",
                    BCluSize[index],
                    (int)error,
                    cudaGetErrorString(error));

            // Substitute pinned memory for nopinned
            memcpy(p, nop, bytes);

            PinnedMem[index] = p;

            // ADD the original memory address
            nop = nop + bytes/sizeof(float);
            index++;
        }
        // Free the nopinned memory
        free(noPinnedMem[i]);
    }
    // Check the correctness of pinned malloc
    FAISS_ASSERT(index == BCluSize.size());
}

void PipeCluster::freePinnedMem(){
    // Free the cudaMallocHost memory
    for (int i = 0; i < PinnedMem.size(); i++){
        auto error = cudaFreeHost(PinnedMem[i]);
        FAISS_ASSERT_FMT(
                error == cudaSuccess,
                "Failed to free pinned memory (error %d %s)",
                (int)error,
                cudaGetErrorString(error));
    }
}

void PipeCluster::setPinnedonDevice(int id, bool b){
    // Set the status
    isPinnedDevice[id] = b;
}

bool PipeCluster::readPinnedonDevice(int id){
    // Read the status
    return isPinnedDevice[id];
}

void PipeCluster::setonDevice(int id, bool b, bool avl){
    // Set the status
    isonDevice[id] = b;
    if (avl){
        if (b){
            LRUtree_->insert(readGlobalCount(id), id);
        }
        else{
            LRUtree_->remove(readGlobalCount(id), id);
        }
    }
}

bool PipeCluster::readonDevice(int id){
    return isonDevice[id];
}

void PipeCluster::addGlobalCount(int id, int num){
    // ADD the count
    GlobalCount[id] += num;
}

int PipeCluster::readGlobalCount(int id){
    // Read Count
    return GlobalCount[id];
}

}