/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/PipeGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <algorithm>

//For debug
#include <unistd.h>

namespace faiss {
namespace gpu {

namespace {

// Default temporary memory allocation
constexpr size_t kTempMem = (size_t)256 * 1024 * 1024;

// Default Max Device memory allocation (15Gib)
constexpr size_t kMaxDeviceMem = (size_t)4 * 1024 * 1024 * 1024ll;

// Default page size allocation (1 MB)
constexpr size_t kPagesize = (size_t)1 * 1024 * 1024;

size_t adjustStackSize(size_t size) {
    if (size == 0) {
        return 0;
    } else {
        // ensure that we have at least 16 bytes, as all allocations are bumped
        // up to 16
        return utils::roundUp(size, (size_t)16);
    }
}


}

//
// PipeGpuResources
//

PipeGpuResources::PipeGpuResources()
        : tempMemSize_(kTempMem),
          MaxDeviceSize_(kMaxDeviceMem),
          pageSize_(kPagesize),
          allocLogging_(false) {}

PipeGpuResources::~PipeGpuResources() {
    // The temporary memory allocator has allocated memory this class, so clean
    // that up before we finish fully de-initializing
    tempMemory_.clear();

    // Destroy the created streams
    for (auto& entry : executeStreams_) {
        DeviceScope scope(entry.first);

        CUDA_VERIFY(cudaStreamDestroy(entry.second));
    }

    for (auto& entry : copyH2DStreams_) {
        DeviceScope scope(entry.first);

        CUDA_VERIFY(cudaStreamDestroy(entry.second));
    }

    for (auto& entry : copyD2HStreams_) {
        DeviceScope scope(entry.first);

        CUDA_VERIFY(cudaStreamDestroy(entry.second));
    }

    for (auto& entry : blasHandles_) {
        DeviceScope scope(entry.first);

        auto blasStatus = cublasDestroy(entry.second);
        FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
    }

    // Free the device memory & temp memory
    // Temp memory will automatically free
    if (p_){
        auto err = cudaFree(p_);
            FAISS_ASSERT_FMT(
                    err == cudaSuccess,
                    "Failed to cudaFree pointer %p (error %d %s)",
                    p_,
                    (int)err,
                    cudaGetErrorString(err));
    }
}

void PipeGpuResources::setTempMemory(size_t size) {
    if (tempMemSize_ != size) {
        // adjust based on general limits
        tempMemSize_ = std::min(size, kTempMem);

        // We need to re-initialize memory resources for all current devices
        // that have been initialized. This should be safe to do, even if we are
        // currently running work, because the cudaFree call that this implies
        // will force-synchronize all GPUs with the CPU
        for (auto& p : tempMemory_) {
            int device = p.first;
            // Free the existing memory first
            p.second.reset();

            // Allocate new
            p.second = std::unique_ptr<PipeTempMemory>(new PipeTempMemory(
                    p.first,
                    // adjust for this specific device
                    tempMemSize_));
        }
    }
}

void PipeGpuResources::setMaxDeviceSize(size_t size) {
    FAISS_ASSERT(size % pageSize_ == 0);
    MaxDeviceSize_ = size;
}

void PipeGpuResources::setPageSize(size_t size) {
    FAISS_ASSERT(size % 256 == 0);
    pageSize_ = size;
}

void PipeGpuResources::setExecuteStream(
        int device,
        cudaStream_t stream) {
    if (isInitialized(device)) {
        // A new series of calls may not be ordered with what was the previous
        // stream, so if the stream being specified is different, then we need
        // to ensure ordering between the two (new stream waits on old)
        auto it = executeStreams_.find(device);
        cudaStream_t prevStream = nullptr;

        // Extract the previous stream
        FAISS_ASSERT(executeStreams_.count(device));
        prevStream = executeStreams_[device];

        if (prevStream != stream) {
            streamWait({stream}, {prevStream});
        }
    }

    executeStreams_[device] = stream;
}

void PipeGpuResources::setCopyH2DStream(
        int device,
        cudaStream_t stream) {
    if (isInitialized(device)) {
        // A new series of calls may not be ordered with what was the previous
        // stream, so if the stream being specified is different, then we need
        // to ensure ordering between the two (new stream waits on old)
        auto it = copyH2DStreams_.find(device);
        cudaStream_t prevStream = nullptr;

        // Extract the previous stream
        FAISS_ASSERT(copyH2DStreams_.count(device));
        prevStream = copyH2DStreams_[device];

        if (prevStream != stream) {
            streamWait({stream}, {prevStream});
        }
    }

    copyH2DStreams_[device] = stream;
}

void PipeGpuResources::setCopyD2HStream(
        int device,
        cudaStream_t stream) {
    if (isInitialized(device)) {
        // A new series of calls may not be ordered with what was the previous
        // stream, so if the stream being specified is different, then we need
        // to ensure ordering between the two (new stream waits on old)
        auto it = copyD2HStreams_.find(device);
        cudaStream_t prevStream = nullptr;

        // Extract the previous stream
        FAISS_ASSERT(copyD2HStreams_.count(device));
        prevStream = copyD2HStreams_[device];

        if (prevStream != stream) {
            streamWait({stream}, {prevStream});
        }
    }

    copyD2HStreams_[device] = stream;
}

cudaStream_t PipeGpuResources::getExecuteStream(int device) {
    initializeForDevice(device, nullptr);

    auto it = executeStreams_.find(device);
    FAISS_ASSERT(it != executeStreams_.end());

    // Return the execute stream
    return executeStreams_[device];
}

cudaStream_t PipeGpuResources::getCopyH2DStream(int device) {
    initializeForDevice(device, nullptr);

    auto it = copyH2DStreams_.find(device);
    FAISS_ASSERT(it != copyH2DStreams_.end());

    // Return the copy Host to Device stream
    return copyH2DStreams_[device];
}

cudaStream_t PipeGpuResources::getCopyD2HStream(int device) {
    initializeForDevice(device, nullptr);

    auto it = copyD2HStreams_.find(device);
    FAISS_ASSERT(it != copyD2HStreams_.end());

    // Return the cpoy Host to Device stream
    return copyD2HStreams_[device];
}

void PipeGpuResources::setLogMemoryAllocations(bool enable) {
    allocLogging_ = enable;
}

bool PipeGpuResources::isInitialized(int device) const {
    // Use default streams as a marker for whether or not a certain
    // device has been initialized
    return isini.count(device) != 0;
}

// Internal system call

cublasHandle_t PipeGpuResources::getBlasHandle(int device) {
    initializeForDevice(device, nullptr);
    return blasHandles_[device];
}

void PipeGpuResources::initializeForDevice(int device, PipeCluster *pc){
    if (isInitialized(device)) {
        return;
    }
    pc_ = pc;

    // Set the page size to the number of bytes of a balanced cluster
    setPageSize(pc->bcs * (pc->d + 1) * sizeof(float));
    
    // switch to the "device"
    FAISS_ASSERT(device < getNumDevices());
    DeviceScope scope(device);

    // Make sure that device properties for all devices are cached
    auto& prop = getDeviceProperties(device);

    // Check to make sure we meet our minimum compute capability (3.0)
    FAISS_ASSERT_FMT(
            prop.major >= 3,
            "Device id %d with CC %d.%d not supported, "
            "need 3.0+ compute capability",
            device,
            prop.major,
            prop.minor);

    // Check to make sure warpSize == 32
    FAISS_ASSERT_FMT(
            prop.warpSize == 32,
            "Device id %d does not have expected warpSize of 32",
            device);

    // Create main streams
    cudaStream_t execStream = 0;
    CUDA_VERIFY(
            cudaStreamCreateWithFlags(&execStream, cudaStreamNonBlocking));

    executeStreams_[device] = execStream;

    cudaStream_t h2dStream = 0;
    CUDA_VERIFY(
            cudaStreamCreateWithFlags(&h2dStream, cudaStreamNonBlocking));

    copyH2DStreams_[device] = h2dStream;

    cudaStream_t d2hStream = 0;
    CUDA_VERIFY(
            cudaStreamCreateWithFlags(&d2hStream, cudaStreamNonBlocking));

    copyD2HStreams_[device] = d2hStream;

    // Create cuBLAS handle
    cublasHandle_t blasHandle = 0;
    auto blasStatus = cublasCreate(&blasHandle);
    FAISS_ASSERT(blasStatus == CUBLAS_STATUS_SUCCESS);
    blasHandles_[device] = blasHandle;

    // No precision reduction
#if CUDA_VERSION >= 11000
    cublasSetMathMode(
            blasHandle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
#endif

    // Allocate Temp memory
    FAISS_ASSERT(tempMemory_.count(device) == 0);
    auto mem = std::unique_ptr<PipeTempMemory>(new PipeTempMemory(
            device,
            tempMemSize_));

    tempMemory_.emplace(device, std::move(mem));

    // Allocate Device memory
    void* p = nullptr;
    auto err = cudaMalloc(&p, MaxDeviceSize_);

    // Throw if we fail to allocate
    if (err != cudaSuccess) {
        cudaGetLastError();

        std::stringstream ss;
        ss << "PipeStack: alloc fail: size " << MaxDeviceSize_
            << " (cudaMalloc error " << cudaGetErrorString(err) << " ["
            << (int)err << "])\n";
        auto str = ss.str();

        if (allocLogging_) {
            std::cout << str;
        }
        FAISS_THROW_IF_NOT_FMT(err == cudaSuccess, "%s", str.c_str());
    }
    p_ = (char *)p;

    // Initialize the pageinfo
    size_t pageNum = MaxDeviceSize_ / pageSize_;
    pageNum_ = pageNum;
    // std::cout << "Device Memory layout (page number):" << pageNum << "\n";
    pageinfo.resize(pageNum);

    std::fill(pageinfo.begin(), pageinfo.end(), -1);

    // Construct the avl tree of free pages
    auto tmptree = std::unique_ptr<PipeAVLTree<int,int> >
            (new PipeAVLTree<int, int>());
    for (int i = 0 ;i < pageNum; i++)
        tmptree->insert(i, i);
    
    freetree_ = std::move(tmptree);

    isini[device] = true;
}

MemBlock PipeGpuResources::allocMemory(int size){

    // Check if the current allocate request size <= Total page size
    FAISS_THROW_IF_NOT(size <= pageinfo.size());

    MemBlock best;

    int cnt = 0;
    int freecnt = 0;

    std::vector<int> alloc(size);

    // Try to allocate free pages
    for(int i = 0; i < size;i++){
        // No free pages left
        if (freetree_->size == 0)
            break;
        
        std::pair<int, int> tmppair = freetree_->minimum();
        // std::cout << tmppair.second << "\n";
        int offset = tmppair.first;
        alloc[cnt] = offset;

        // Delete the free pages in avl tree
        freetree_->remove(offset, offset);

        cnt++;
    }

    // Calculate the realloc pages
    freecnt = cnt;
    cnt = size - freecnt;

    if (cnt == 0){
        // no need to sort
        best.pages = std::move(alloc);
        return best;
    }
    cnt = 0;

    std::vector<std::pair<int, int> > tmppairs(size);
    // Try to reallocate pages
    for (int i= 0; i < size - freecnt; i++){
        if(pc_->LRUtree_->size == 0)
            break;
        
        std::pair<int, int> tmppair = pc_->LRUtree_->minimum();
        tmppairs[cnt] = tmppair;
        alloc[freecnt+cnt] = tmppair.second;

        // Delete the free pages in avl tree
        pc_->LRUtree_->remove(tmppair.first, tmppair.second);

        cnt++;
    }
    if (cnt + freecnt == size){
        // Free these pages
        for (int i = freecnt; i < size; i++){
            int id = alloc[i];
            pc_->clu_page[pageinfo[id]] = -1;
            pc_->setonDevice(pageinfo[id], id, false, false);
            pageinfo[id] = -1;
        }
        std::sort(alloc.begin(), alloc.end());
        best.pages = std::move(alloc);
        return best;
    }
    // No enough pages to allcate
    // Reset the allocated pages in trees
    else{
        int i = 0;
        for (; i < freecnt; i++){
            pc_->LRUtree_->insert(alloc[i], alloc[i]);
        }
        for (; i < freecnt + cnt; i++){
            pc_->LRUtree_->insert(tmppairs[i - freecnt].first, tmppairs[i - freecnt].second);
        }
    }

    best.valid = false;
    return best;
    
}

void* PipeGpuResources::allocTemMemory(size_t size){
    // Default is zero
    int dev = 0;
    return tempMemory_[dev]->allocMemory(size);
}

void PipeGpuResources::deallocTemMemory(void *p, size_t size){
    // Default is zero
    int dev = 0;
    tempMemory_[dev]->deallocMemory(p, size);
}

void PipeGpuResources::updatePages(const std::vector<int> &pages, 
        const std::vector<int> &clus){

    int len = pages.size();
    // Check the length
    FAISS_THROW_IF_NOT_FMT(len == clus.size(), "%s", 
            "The length of pages and clusters is not matched");

    for (int i = 0; i < len; i++){
        // Update pageinfo
        pageinfo[i] = clus[i];

        // Update pointer pipecluster
        pc_->setonDevice(clus[i], pages[i], true);
    }
}

void PipeGpuResources::deallocMemory(int sta, int num){
    // for (int i = sta; i < num + sta; i++){
    //     pc_->setonDevice(i, false);
    //     pc_->setPinnedonDevice(i, false);
    //     pageinfo[i] = -1;
    // }
}

void* PipeGpuResources::getPageAddress(int pageid){
    FAISS_ASSERT(pageid >=0 && pageid < pageNum_);
    // printf("%d\n", pageid);

    // Return a pointer
    return (void*)(p_ + pageSize_ * pageid);
}

void PipeGpuResources::memcpyh2d(int pageid){
    float *target = (float*)getPageAddress(pageid);

    int cluid = pageinfo[pageid];
    FAISS_ASSERT(cluid >= 0);
    size_t bytes = pc_->BCluSize[cluid] * sizeof(float) * pc_->d;
    size_t index_bytes = pc_->BCluSize[cluid] * sizeof(int);

    float *index_target = target + bytes / sizeof(float);

    // This is a sync version copy for profiler
    cudaMemcpy((void*)target , pc_->Mem[cluid], 
            bytes, cudaMemcpyHostToDevice);

    cudaMemcpy((void*)index_target , pc_->Balan_ids[cluid], 
            index_bytes, cudaMemcpyHostToDevice);
}

//
// PipeTempMemory & PipeStack
//

PipeTempMemory::PipeStack::PipeStack(int d, size_t size)
        : device_(d),
          alloc_(nullptr),
          allocSize_(adjustStackSize(size)),
          start_(nullptr),
          end_(nullptr),
          head_(nullptr),
          highWaterMemoryUsed_(0) {
    if (allocSize_ == 0) {
        return;
    }

    DeviceScope s(device_);

    // Each GPU memory page must be aligned to 256 bytes
    size_t adjsize = utils::roundUp(allocSize_, (size_t)256);

    void* p = nullptr;
    auto err = cudaMalloc(&p, adjsize);

    // Fail to alloc stack memory
    if (err != cudaSuccess) {
        // FIXME: as of CUDA 11, a memory allocation error appears to be
        // presented via cudaGetLastError as well, and needs to be cleared.
        // Just call the function to clear it
        cudaGetLastError();

        std::stringstream ss;
        ss << "PipeStack: alloc fail: size " << allocSize_
            << " (cudaMalloc error " << cudaGetErrorString(err) << " ["
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

PipeTempMemory::PipeStack::~PipeStack() {
    DeviceScope s(device_);

    if (alloc_) {
        auto err = cudaFree(alloc_);
        FAISS_ASSERT_FMT(
                err == cudaSuccess,
                "Failed to cudaFree pointer %p (error %d %s)",
                alloc_,
                (int)err,
                cudaGetErrorString(err));
    }
}

size_t PipeTempMemory::PipeStack::getSizeAvailable() const {
    return (end_ - head_);
}

char* PipeTempMemory::PipeStack::getAlloc(size_t size){
    // The allocation fails if the remaining memory isnot enough
    auto sizeRemaining = getSizeAvailable();

    FAISS_ASSERT_FMT(size <= sizeRemaining, "The PipeStack is not enough for size: %zu", size);

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

void PipeTempMemory::PipeStack::deAlloc(
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

std::string PipeTempMemory::PipeStack::toString() const {
    std::stringstream s;

    s << "SDM device " << device_ << ": Total memory " << allocSize_ << " ["
      << (void*)start_ << ", " << (void*)end_ << ")\n";
    s << "     Available memory " << (size_t)(end_ - head_) << " ["
      << (void*)head_ << ", " << (void*)end_ << ")\n";
    s << "     High water temp alloc " << highWaterMemoryUsed_ << "\n";

    return s.str();
}

PipeTempMemory::PipeTempMemory(
        int device,
        size_t sz)
        : device_(device), stack_(device, sz) {}

PipeTempMemory::~PipeTempMemory() {}

int PipeTempMemory::getDevice() const {
    return device_;
}

size_t PipeTempMemory::getSizeAvailable() const {
    return stack_.getSizeAvailable();
}

std::string PipeTempMemory::toString() const {
    return stack_.toString();
}

void* PipeTempMemory::allocMemory(size_t size) {
    // All allocations should have been adjusted to a multiple of 16 bytes
    FAISS_ASSERT(size % 16 == 0);
    return (void*)stack_.getAlloc(size);
}

void PipeTempMemory::deallocMemory(
        void* p,
        size_t size) {
    FAISS_ASSERT(p);

    stack_.deAlloc((char*)p, size);
}


} // namespace gpu
} // namespace faiss