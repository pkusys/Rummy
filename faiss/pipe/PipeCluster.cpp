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
#include <sstream>

namespace faiss {

namespace {

// Default expected standard variance ratio
const float StdVarRation = 0.2;

// Default Temp PinMemory Size (64 MiB)
const int PinTempSize = 64 * 1024 * 1024;

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
            std::vector<float *> & pointers, std::vector<int*> & indexes, 
            bool interleaved_) : stack_(PinTempSize){
    
    // Initialize some attributes
    nlist = nlist_;

    d = d_;

    CluSize = sizes;

    interleaved = interleaved_;

    noPinnedMem = pointers;

    noBalan_ids = indexes;

    Mem.resize(nlist);

    // Balance the clusters
    balance(StdVarRation);

    // Initialize the device memory info
    isonDevice.resize(BCluSize.size());
    isPinnedDevice.resize(BCluSize.size());
    isComDevice.resize(BCluSize.size());
    Mem.resize(BCluSize.size());
    Balan_ids.resize(BCluSize.size());
    GlobalCount.resize(BCluSize.size());
    DeviceMem.resize(BCluSize.size());
    clu_page.resize(BCluSize.size());

    std::fill(isonDevice.begin(), isonDevice.end(), false);
    std::fill(isPinnedDevice.begin(), isPinnedDevice.end(), false);
    std::fill(isComDevice.begin(), isComDevice.end(), false);
    std::fill(GlobalCount.begin(), GlobalCount.end(), 0);
    std::fill(clu_page.begin(), clu_page.end(), -1);

    // Construct the OnDevice cluster
    auto tmptree = std::unique_ptr<PipeAVLTree<int,int> >
            (new PipeAVLTree<int, int>());
    
    LRUtree_ = std::move(tmptree);

    mallocPinnedMem();
    // mallocNoPinnedMem();

    pthread_mutex_init(&resource_mutex, 0);

    pthread_mutex_init(&com_mutex, 0);

}

PipeCluster::~PipeCluster(){
    // Free the allocated memory if necessary
    if (!Mem.empty())
        freeMem();
    
    // Free the mutex
    pthread_mutex_destroy(&resource_mutex);

    pthread_mutex_destroy(&com_mutex);
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
        int *index_nop = noBalan_ids[i];
        
        if (!interleaved) {
            for(int j = 0; j < num; j++){
                size_t bytes = BCluSize[index] * d * sizeof(float);
                float *p;

                size_t index_bytes = BCluSize[index] * sizeof(int);
                int *index_p;

                // Malloc pinned memory
                auto error = cudaMallocHost((void **) &p, bytes);
                FAISS_ASSERT_FMT(
                        error == cudaSuccess,
                        "Failed to malloc pinned memory: %d vectors (error %d %s)",
                        BCluSize[index],
                        (int)error,
                        cudaGetErrorString(error));

                error = cudaMallocHost((void **) &index_p, index_bytes);
                FAISS_ASSERT_FMT(
                        error == cudaSuccess,
                        "Failed to malloc pinned memory: %d vectors (error %d %s)",
                        BCluSize[index],
                        (int)error,
                        cudaGetErrorString(error));

                // Substitute pinned memory for nopinned
                memcpy(p, nop, bytes);
                memcpy(index_p, index_nop, index_bytes);

                Mem[index] = p;
                Balan_ids[index] = index_p;

                // ADD the original memory address
                nop = nop + bytes/sizeof(float);
                index_nop = index_nop + index_bytes/sizeof(int);
                index++;
            }
        }
        else{
            for(int j = 0; j < num; j++){
                size_t nobytes = BCluSize[index] * d * sizeof(float);
                size_t bytes = BCluSize[index] % 32 == 0 ? BCluSize[index] : 
                    BCluSize[index] / 32 * 32 + 32 ;
                bytes *= d * sizeof(float);
                float *p;

                size_t index_bytes = BCluSize[index] * sizeof(int);
                int *index_p;

                // Malloc pinned memory
                auto error = cudaMallocHost((void **) &p, bytes);
                FAISS_ASSERT_FMT(
                        error == cudaSuccess,
                        "Failed to malloc pinned memory: %d vectors (error %d %s)",
                        BCluSize[index],
                        (int)error,
                        cudaGetErrorString(error));

                error = cudaMallocHost((void **) &index_p, index_bytes);
                FAISS_ASSERT_FMT(
                        error == cudaSuccess,
                        "Failed to malloc pinned memory: %d vectors (error %d %s)",
                        BCluSize[index],
                        (int)error,
                        cudaGetErrorString(error));
                memcpy(index_p, index_nop, index_bytes);

                // Substitute pinned memory for nopinned
                int nt = std::min(omp_get_max_threads(), BCluSize[index]);
#pragma omp parallel for if (nt > 1)
                for (int l = 0; l < BCluSize[index]; l++) {
                    for (int m = 0; m < d; m++) {
                        int oldidx = l * d + m;
                        int newidx = m * 32 + (int)(l / 32) * (32 * d) + l % 32;
                        p[newidx] = nop[oldidx]; 
                    }
                }

                Mem[index] = p;
                Balan_ids[index] = index_p;

                // ADD the original memory address
                nop = nop + nobytes/sizeof(float);
                index_nop = index_nop + index_bytes/sizeof(int);
                index++;
            }
        }
        
        // Free the nopinned memory
        free(noPinnedMem[i]);
        free(noBalan_ids[i]);
    }
    // Check the correctness of pinned malloc
    FAISS_ASSERT(index == BCluSize.size());

    pinned = true;
}

void PipeCluster::mallocNoPinnedMem(){
    // Malloc the no pinned memory and free the nopinned memory
    int index = 0;

    for (int i = 0; i < nlist; i++){
        auto balaClu = O2Bmap[i];
        int num = balaClu.size();
        float *nop = noPinnedMem[i];
        int *index_nop = noBalan_ids[i];
        
        for(int j = 0; j < num; j++){
            size_t bytes = BCluSize[index] * d * sizeof(float);
            float *p;

            size_t index_bytes = BCluSize[index] * sizeof(int);
            int *index_p;

            // Malloc no pinned memory
            p = (float *)malloc(bytes);
            index_p = (int *)malloc(index_bytes);

            // Substitute pinned memory for nopinned
            memcpy(p, nop, bytes);
            memcpy(index_p, index_nop, index_bytes);

            Mem[index] = p;
            Balan_ids[index] = index_p;

            // ADD the original memory address
            nop = nop + bytes/sizeof(float);
            index_nop = index_nop + index_bytes/sizeof(int);
            index++;
        }
        // Free the nopinned memory
        free(noPinnedMem[i]);
        free(noBalan_ids[i]);
    }
    // Check the correctness of pinned malloc
    FAISS_ASSERT(index == BCluSize.size());

    pinned = false;
}

void PipeCluster::freeMem(){
    // Free the cudaMallocHost memory
    if (pinned){
        for (int i = 0; i < Mem.size(); i++){
            auto error = cudaFreeHost(Mem[i]);
            FAISS_ASSERT_FMT(
                    error == cudaSuccess,
                    "Failed to free pinned memory (error %d %s)",
                    (int)error,
                    cudaGetErrorString(error));
        }
    }
    else{
        for (int i = 0; i < Mem.size(); i++){
            free(Mem[i]);
        }
    }
}

void PipeCluster::setPinnedonDevice(int id, int page_id, bool b, bool avl){
    // Set the status
    isPinnedDevice[id] = b;
    if (avl){
        if (b){
            LRUtree_->remove(readGlobalCount(id), page_id);
        }
        else{
            LRUtree_->insert(readGlobalCount(id), page_id);
        }
    }
}

void PipeCluster::setComDevice(int id, int page_id, bool b, bool avl){
    // Set the status
    isComDevice[id] = b;
    if (avl){
        if (b){
            LRUtree_->remove(readGlobalCount(id), page_id);
        }
        else{
            LRUtree_->insert(readGlobalCount(id), page_id);
        }
    }
}

bool PipeCluster::readComDevice(int id){
    // Read the status
    return isComDevice[id];
}

bool PipeCluster::readPinnedonDevice(int id){
    // Read the status
    return isPinnedDevice[id];
}

void PipeCluster::setonDevice(int id, int page_id, bool b, bool avl){
    // Set the status
    isonDevice[id] = b;
    if (avl){
        if (b){
            LRUtree_->insert(readGlobalCount(id), page_id);
        }
        else{
            LRUtree_->remove(readGlobalCount(id), page_id);
        }
    }
}

bool PipeCluster::readonDevice(int id){
    return isonDevice[id];
}

void PipeCluster::addGlobalCount(int id, int page_id, int num){
    // ADD the count
    GlobalCount[id] += num;

    if (page_id != -1){
        // Sync with LRU tree
        auto node = LRUtree_->Search(GlobalCount[id] - num, page_id);
        // Not pinned on device
        if (node){
            int newkey = node->key + num;
            int newval = node->val;
            LRUtree_->remove(node->key, node->val);
            LRUtree_->insert(newkey, newval);
        }
    }
}

int PipeCluster::readGlobalCount(int id){
    // Read Count
    return GlobalCount[id];
}

//
// PinTempMemory & PinStack
//

PipeCluster::PinStack::PinStack(size_t size)
        : alloc_(nullptr),
          allocSize_(size),
          start_(nullptr),
          end_(nullptr),
          head_(nullptr),
          highWaterMemoryUsed_(0) {
    if (allocSize_ == 0) {
        return;
    }

    // Temp Pinned page must be aligned to 256 bytes
    size_t adjsize = ((allocSize_ + 255) / 256 ) * 256;

    void* p = nullptr;
    auto err = cudaMallocHost(&p, adjsize);

    // Fail to alloc stack memory
    if (err != cudaSuccess) {
        // FIXME: as of CUDA 11, a memory allocation error appears to be
        // presented via cudaGetLastError as well, and needs to be cleared.
        // Just call the function to clear it
        cudaGetLastError();

        std::stringstream ss;
        ss << "PinStack: alloc fail: size " << allocSize_
            << " (cudaMallocHost error " << cudaGetErrorString(err) << " ["
            << (int)err << "])\n";
        auto str = ss.str();

        FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "%s", str.c_str());
    }

    alloc_ = (char*) p;
    FAISS_ASSERT_FMT(
            alloc_,
            "could not reserve temporary memory region of size %zu",
            allocSize_);

    // Initialize the stack info
    start_ = alloc_;
    head_ = start_;
    end_ = alloc_ + adjsize;
}

PipeCluster::PinStack::~PinStack() {

    if (alloc_) {
        auto err = cudaFreeHost(alloc_);
        FAISS_ASSERT_FMT(
                err == cudaSuccess,
                "Failed to cudaFreeHost pointer %p (error %d %s)",
                alloc_,
                (int)err,
                cudaGetErrorString(err));
    }
}

size_t PipeCluster::PinStack::getSizeAvailable() const {
    return (end_ - head_);
}

char* PipeCluster::PinStack::getAlloc(size_t size){
    // The allocation fails if the remaining memory isnot enough
    auto sizeRemaining = getSizeAvailable();

    FAISS_ASSERT_FMT(size <= sizeRemaining, "The PinStackis not enough for size: %zu", size);

    // All allocations should have been adjusted to a multiple of 16 bytes
    FAISS_ASSERT(size % 16 == 0);

    char* startAlloc = head_;
    char* endAlloc = head_ + size;

    head_ = endAlloc;
    FAISS_ASSERT(head_ <= end_);

    highWaterMemoryUsed_ =
            std::max(highWaterMemoryUsed_, (size_t)(head_ - start_));
    FAISS_ASSERT(startAlloc);
    return startAlloc;

}

void PipeCluster::PinStack::deAlloc(
        char* p,
        size_t size) {
    // This allocation should be within ourselves
    FAISS_ASSERT(p >= start_ && p < end_);

    // All allocations should have been adjusted to a multiple of 16 bytes
    FAISS_ASSERT(size % 16 == 0);

    // This is on our stack
    // Allocations should be freed in the reverse order they are made
    if (p + size != head_) {
        FAISS_ASSERT(p + size == head_);
    }

    head_ = p;
}

std::string PipeCluster::PinStack::toString() const {
    std::stringstream s;

    s << "Temp Pin Host Memory " << ": Total memory " << allocSize_ << " ["
      << (void*)start_ << ", " << (void*)end_ << ")\n";
    s << "     Available memory " << (size_t)(end_ - head_) << " ["
      << (void*)head_ << ", " << (void*)end_ << ")\n";
    s << "     High water temp alloc " << highWaterMemoryUsed_ << "\n";

    return s.str();
}

size_t PipeCluster::getPinTempAvail() const {
    return stack_.getSizeAvailable();
}

std::string PipeCluster::PinTempStatus() const {
    return stack_.toString();
}

void* PipeCluster::allocPinTemp(size_t size) {
    // All allocations should have been adjusted to a multiple of 16 bytes
    FAISS_ASSERT(size % 16 == 0);
    return (void*)stack_.getAlloc(size);
}

void PipeCluster::freePinTemp(
        void* p,
        size_t size) {
    FAISS_ASSERT(p);

    stack_.deAlloc((char*)p, size);
}

}