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

    // Free Pinned memory in host for balanced clusters
    void freePinnedMem();

    // Set the balanced clusters's pinned status on device
    void setPinnedonDevice(int id, bool b);

    // Check the balanced clusters's pinned status
    bool readPinnedonDevice(int id);

    // Set the balanced clusters's status on device 
    // (avl means if you want to change the avl tree)
    void setonDevice(int id, bool b, bool avl = true);

    // Return if the cluster is on device
    bool readonDevice(int id);

    // Add the Global count by reference number
    void addGlobalCount(int id, int num);

    // Check the Global count by reference number
    int readGlobalCount(int id);

public:
    /// AVL Tree to manage the allocated clusters (LRU count -> offset)
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

    // Each cluster's number of vectors
    std::vector<int> CluSize;

    // Each balanced cluster's number of vectors
    std::vector<int> BCluSize;

    // Map original clusters to balanced clusters
    std::unordered_map<int, std::vector<int> > O2Bmap;

    // Each cluster's storage on noPinned Memory
    std::vector<float*> noPinnedMem;

    // Each cluster's storage on Pinned Memory
    std::vector<float*> PinnedMem;

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