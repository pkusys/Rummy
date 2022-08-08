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
#include <pthread.h>  

namespace faiss {

// This class controls the memory management on CPU side
class PipeCluster {
public:

    // Construct the cluster info
    PipeCluster(int nlist, int d, std::vector<int> & sizes,
            std::vector<float *> & pointers, std::vector<int*> & indexes, bool interleaved_);

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

    // Set the balanced clusters's status on computation 
    // (avl means if you want to change the avl tree)
    void setComDevice(int clu_id, int page_id, bool b, bool avl = true);

    // Return if the cluster is on device
    bool readonDevice(int id);

    // Return if the cluster is on device
    bool readComDevice(int id);

    // Add the Global count by reference number (If not in GPU pageid = -1)
    void addGlobalCount(int id, int pageid, int num);

    // Check the Global count by reference number
    int readGlobalCount(int id);

public:
    /// AVL Tree to manage the allocated clusters 
    /// (Key-Value: LRU count -> page id) 
    /// (only contains no-pinned clusters on device)
    std::unique_ptr<PipeAVLTree<int,int> > LRUtree_;

public: // For convenient, may change the mode to public later
// private
    
    // The minmal block number for a kernel launch
    int Min_Block = -1;

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

    // Whether the code should be interleaved
    bool interleaved;

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

    // Each cluster's index id on noPinned Memory before balance
    std::vector<int*> noBalan_ids;

    // Each balanced cluster's storage
    std::vector<float*> Mem;

    // Each balanced cluster's idx
    std::vector<int*> Balan_ids;

    // Each cluster's storage on Device
    std::vector<int> DeviceMem;

    // Check if the cluster is resident on the device
    std::vector<bool> isonDevice;

    // Check if the cluster is pinned on the device
    std::vector<bool> isPinnedDevice;

    // Check if the cluster is currently executed by one query on device
    std::vector<bool> isComDevice;

    /// LRU count
    std::vector<int> GlobalCount;

    /// Map Cluster -> page id
    std::vector<int> clu_page;

    // mutex to lock pipecluster and pipegpuresource
    pthread_mutex_t resource_mutex;

    // mutex to guarantee there is only one thread exec in computation
    pthread_mutex_t com_mutex;

    // the computation threads
    std::vector<pthread_t> com_threads;

protected:
    struct PinStack{
        /// Constructor that allocates memory
        PinStack(size_t size);

        ~PinStack();

        /// Returns how much size is available 
        /// (existing free pin memory) for an allocation
        size_t getSizeAvailable() const;

        /// Obtains an allocation; all allocations must be 16
        /// byte aligned 
        char* getAlloc(size_t size);

        /// Returns an allocation
        void deAlloc(char* p, size_t size);

        /// Returns the stack state
        std::string toString() const;

        /// Where our temporary memory buffer is allocated starts; we allocate starting
        /// 16 bytes into this pointer
        char* alloc_;

        /// Pinstack size
        size_t allocSize_;

        /// Valid emporary memory region; [start_, end_)
        char* start_;
        char* end_;

        /// Stack head within [start, end)
        char* head_;

        /// What's the high water mark in terms of memory used from the
        /// temporary buffer?
        size_t highWaterMemoryUsed_;

    };
private:

    /// Pin memory stack
    PinStack stack_;

public:
    /// Return the available Pin Temp Size
    size_t getPinTempAvail() const;

    /// Display the Pin Temp Status
    std::string PinTempStatus() const;

    /// Allocate a block of continuous memory
    void* allocPinTemp(size_t size);

    /// Free a block allocPinTemp(size_t size) memory
    void freePinTemp(void* p, size_t size);
};

}