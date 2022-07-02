/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>
#include <faiss/pipe/PipeStructure.h>

#include <list>
#include <memory>
#include <tuple>

namespace faiss {

// This class controls the memory management on CPU side
class PipeCluster {
public:

    // Construct the cluster info
    PipeCluster(int nlist, int d, std::vector<int> & sizes, 
            std::vector<float *> & pointers);

    ~PipeCluster();

    // Balance each cluster's number of vectors
    void balance(const float svr);

    // Malloc Pinned memory in host for balanced clusters
    void mallocPinnedMem();

    // Malloc No-Pinned memory in host for balanced clusters(for comparison)
    void mallocNoPinnedMem();

    // Free Pinned memory in host for balanced clusters
    void freeMem();

    // Set the balanced clusters's pinned status on device
    void setPinnedonDevice(int id, int page_id, bool b, bool avl = true);

    // Check the balanced clusters's pinned status
    bool readPinnedonDevice(int id);

    // Set the balanced clusters's status on device 
    // (avl means if you want to change the avl tree)
    void setonDevice(int clu_id, int page_id, bool b, bool avl = true);

    // Return if the cluster is on device
    bool readonDevice(int id);

    // Add the Global count by reference number
    void addGlobalCount(int id, int num);

    // Check the Global count by reference number
    int readGlobalCount(int id);

public:
    /// AVL Tree to manage the allocated clusters 
    /// (Key-Value: LRU count -> page id) 
    /// (only contains no-pinned clusters on device)
    std::unique_ptr<PipeAVLTree<int,int> > LRUtree_;

public: // For convenient, may change the mode to public later
// private

    // Number of the clusters
    int nlist;

    // Number of the balanced clusters
    int bnlist;

    // Vector dimension
    int d;

    // Balaced cluster size
    int bcs;

    // Check if we use pinned memory for Mem
    bool pinned = false;

    // Each cluster's number of vectors
    std::vector<int> CluSize;

    // Each balanced cluster's number of vectors
    std::vector<int> BCluSize;

    // Map original clusters to balanced clusters
    std::unordered_map<int, std::vector<int> > O2Bmap;

    // Map original clusters to number of balanced clusters
    std::unordered_map<int, int > O2Bcnt;

    // Each cluster's storage on noPinned Memory before balance
    std::vector<float*> noPinnedMem;

    // Each balanced cluster's storage
    std::vector<float*> Mem;

    // Each cluster's storage on Device
    std::vector<int> DeviceMem;

    // Check if the cluster is resident on the device
    std::vector<bool> isonDevice;

    // Check if the cluster is pinned on the device
    std::vector<bool> isPinnedDevice;

    /// LRU count
    std::vector<int> GlobalCount;
};

}