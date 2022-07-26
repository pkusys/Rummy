/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StackDeviceMemory.h>
#include <faiss/pipe/PipeCluster.h>
#include <faiss/pipe/PipeStructure.h>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include <list>
#include <memory>
#include <tuple>

namespace faiss {
namespace gpu {

class PipeTempMemory;

struct MemBlock{
    
    //The allocate pages
    std::vector<int> pages;
    
    // The LRU count
    int count = -1;
    bool valid = true;
};

/// Implementation of the GpuResources object that provides for
/// pipelined and oversubscribed memory management.
class PipeGpuResources {
public:
    PipeGpuResources();

    ~PipeGpuResources();

    /// Specify that we wish to use a certain fixed size of memory on
    /// all devices as temporary memory.
    void setTempMemory(size_t size);

    /// Specify that we wish to use a maximum fixed size of memory on
    /// all devices as device memory.
    void setMaxDeviceSize(size_t size);

    /// Specify the fixed page size to allocate device memory
    void setPageSize(size_t size);

    /// Set the Execute stream.
    void setExecuteStream(int device, cudaStream_t stream);

    /// Set the Async Copy Host to Device stream.
    void setCopyH2DStream(int device, cudaStream_t stream);

    /// Set the Async Copy Device to Host stream.
    void setCopyD2HStream(int device, cudaStream_t stream);

    /// Returns the execute stream for the given device on which all Faiss GPU work is
    /// ordered
    cudaStream_t getExecuteStream(int device);

    /// Returns the copy H2D stream for the given device
    cudaStream_t getCopyH2DStream(int device);

    /// Returns the copy D2H stream for the given device
    cudaStream_t getCopyD2HStream(int device);

    /// If enabled, will print every GPU memory allocation and deallocation to
    /// standard output in detail
    void setLogMemoryAllocations(bool enable);

public:

    /// Internal system calls

    /// Initialize resources for this device
    void initializeForDevice(int device, PipeCluster * pc);

    cublasHandle_t getBlasHandle(int device);

    /// Allocate free GPU memory for size pages (Already cudamalloc, just return a pointer here)
    MemBlock allocMemory(int size);

    /// Allocate temp device memory
    void* allocTemMemory(size_t size);

    /// DeAllocate temp device memory
    void deallocTemMemory(void *p, size_t size);

    /// Update the pages info according to responding allocated clusters
    void updatePages(const std::vector<int> &pages, const std::vector<int> &clus);

    // /// Allocate non-free GPU memory since GPU memory is not enough
    // void* reallocMemory(size_t size);

    /// Free a set of pages
    void deallocMemory(int sta, int num);

    void* getPageAddress(int pageid);

    void memcpyh2d(int pageid);

private:

    /// Have GPU resources been initialized for this device yet?
    bool isInitialized(int device) const;

public: // For debug
// private:
    
    /// Temporary memory provider, per each device
    std::unordered_map<int, std::unique_ptr<PipeTempMemory>> tempMemory_;

    /// Our execute stream that work is ordered on, one per each device
    std::unordered_map<int, cudaStream_t> executeStreams_;

    /// Our Copy from host to device stream
    std::unordered_map<int, cudaStream_t> copyH2DStreams_;

    /// Our Copy from device to host stream
    std::unordered_map<int, cudaStream_t> copyD2HStreams_;

    /// cuBLAS handle for each device
    std::unordered_map<int, cublasHandle_t> blasHandles_;

    /// Another option is to use a specified amount of memory on all
    /// devices
    size_t tempMemSize_;

    /// Use Max fixed page size to allocate device memory
    size_t MaxDeviceSize_;

    /// Intilized
    std::unordered_map<int, bool> isini;

    /// Use a fixed page size to allocate device memory
    size_t pageSize_;

    // Bytes of a page
    size_t pageNum_;

    /// Whether or not we log every GPU memory allocation and deallocation
    bool allocLogging_;

    /// Where the allocated memory is resident on
    char *p_ = nullptr;

    /// Record which cluster the ith page is affiliated to
    std::vector<int> pageinfo;

    /// AVL Tree to manage the free pages
    std::unique_ptr<PipeAVLTree<int,int> > freetree_;

    /// The cluster info on CPU side
    PipeCluster *pc_;
};

// Typical strorage for query vectors
class PipeTempMemory {
public:
    /// Allocate a new region of GPU memory that we are reponsible to manage
    PipeTempMemory(int device, size_t size);

    /// Destructor function
    ~PipeTempMemory();

    /// Get the device on which the TempMemory is resident
    int getDevice() const;

    /// All allocations requested should be a multiple of 16
    /// bytes (already malloc, just return the pointer)
    void* allocMemory(size_t size);

    /// Free a block of memory (deprecated !!!)
    void deallocMemory(void* p, size_t size);

    size_t getSizeAvailable() const;

    /// Returns the current state
    std::string toString() const;

protected:
    
    struct PipeStack{
        /// Constructor that allocates memory
        PipeStack(int device, size_t size);

        ~PipeStack();

        /// Returns how much size is available (existing malloc) for an allocation
        size_t getSizeAvailable() const;

        /// Obtains an allocation; all allocations must be 16
        /// byte aligned 
        char* getAlloc(size_t size);

        /// Returns an allocation
        void deAlloc(char* p, size_t size);

        /// Returns the stack state
        std::string toString() const;

        /// Which device this allocation is on
        int device_;

        /// Where our temporary memory buffer is allocated starts; we allocate starting
        /// 16 bytes into this pointer
        char* alloc_;

        /// Pipestack size
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
    /// The device
    int device_;

    /// Pipe memory stack
    PipeStack stack_;
};

} // namespace gpu
} // namespace faiss