/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuFaissAssert.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <cstring>
#include <limits>
#include <utility> // std::move

namespace faiss {
namespace gpu {

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::PipeTensor()
        : hostdata_(nullptr), devicedata_(nullptr) {
    static_assert(Dim > 0, "must have > 0 dimensions");

    for (int i = 0; i < Dim; i++) {
        size_[i] = 0;
        stride_[i] = (IndexT)1;
    }
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::~PipeTensor() {
    if (state_ == AllocState::HostOwner) {
        FAISS_ASSERT(this->hostdata_ != nullptr);
        // delete[] this->data_;
        pc->freePinTemp(this->hostdata_, this->getSizeInBytes());
        this->hostdata_ = nullptr;
    }
    else if (state_ == AllocState::DeviceOwner) {
        FAISS_ASSERT(this->devicedata_ != nullptr);
        pgr->deallocTemMemory(this->devicedata_, this->getSizeInBytes());
        this->devicedata_ = nullptr;
    }
    else if (state_ == AllocState::BothOwner){
        FAISS_ASSERT(this->hostdata_ != nullptr);
        // delete[] this->data_;
        pc->freePinTemp(this->hostdata_, this->getSizeInBytes());
        this->hostdata_ = nullptr;

        FAISS_ASSERT(this->devicedata_ != nullptr);
        pgr->deallocTemMemory(this->devicedata_, this->getSizeInBytes());
        this->devicedata_ = nullptr;
    }
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::PipeTensor(
        PipeTensor<T, Dim, InnerContig, IndexT>& t) {
    this->operator=(t);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::PipeTensor(
        PipeTensor<T, Dim, InnerContig, IndexT>&& t) {
    this->operator=(std::move(t));
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>& PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::operator=(PipeTensor<T, Dim, InnerContig, IndexT>&
                                      t) {
    hostdata_ = t.hostdata_;
    devicedata_ = t.devicedata_;
    for (int i = 0; i < Dim; i++) {
        size_[i] = t.size_[i];
        stride_[i] = t.stride_[i];
    }

    return *this;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ __device__ PipeTensor<T, Dim, InnerContig, IndexT>& PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::operator=(PipeTensor<T, Dim, InnerContig, IndexT>&&
                                      t) {
    hostdata_ = t.hostdata_;
    t.hostdata_ = nullptr;
    devicedata_ = t.devicedata_;
    t.devicedata_ = nullptr;
    t.state_ = AllocState::NotOwner;
    for (int i = 0; i < Dim; i++) {
        stride_[i] = t.stride_[i];
        t.stride_[i] = 0;
        size_[i] = t.size_[i];
        t.size_[i] = 0;
    }

    return *this;
}


template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::PipeTensor(
        DataPtrType data,
        const IndexT sizes[Dim])
        : hostdata_(data), devicedata_(nullptr) {
    
    static_assert(Dim > 0, "PipeTensor must have > 0 dimensions");

    for (int i = 0; i < Dim; i++) {
        size_[i] = sizes[i];
    }

    stride_[Dim - 1] = (IndexT)1;
    for (int i = Dim - 2; i >= 0; i--) {
        stride_[i] = stride_[i + 1] * sizes[i + 1];
    }
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::PipeTensor(
        DataPtrType data,
        std::initializer_list<IndexT> sizes)
        : hostdata_(data), devicedata_(nullptr) {
    
    GPU_FAISS_ASSERT(sizes.size() == Dim);
    static_assert(Dim > 0, "PipeTensor must have > 0 dimensions");

    int i = 0;
    for (auto s : sizes) {
        size_[i++] = s;
    }

    stride_[Dim - 1] = (IndexT)1;
    for (int j = Dim - 2; j >= 0; j--) {
        stride_[j] = stride_[j + 1] * size_[j + 1];
    }
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::PipeTensor(
        const IndexT sizes[Dim], PipeCluster* pc_)
        : state_(AllocState::HostOwner) {
    static_assert(Dim > 0, "PipeTensor must have > 0 dimensions");

    for (int i = 0; i < Dim; i++) {
        size_[i] = sizes[i];
    }

    stride_[Dim - 1] = (IndexT)1;
    for (int i = Dim - 2; i >= 0; i--) {
        stride_[i] = stride_[i + 1] * sizes[i + 1];
    }
    // this->hostdata_ = new T[this->numElements()];
    this->hostdata_ = (DataPtrType)pc_->allocPinTemp(this->getSizeInBytes());
    FAISS_ASSERT(this->hostdata_ != nullptr);
    this->devicedata_ = nullptr;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::PipeTensor( 
        std::initializer_list<IndexT> sizes, PipeCluster* pc_)
        : state_(AllocState::HostOwner) {
    GPU_FAISS_ASSERT(sizes.size() == Dim);
    static_assert(Dim > 0, "PipeTensor must have > 0 dimensions");

    int i = 0;
    for (auto s : sizes) {
        size_[i++] = s;
    }

    stride_[Dim - 1] = (IndexT)1;
    for (int j = Dim - 2; j >= 0; j--) {
        stride_[j] = stride_[j + 1] * size_[j + 1];
    }
    // this->hostdata_ = new T[this->numElements()];
    this->hostdata_ = (DataPtrType)pc_->allocPinTemp(this->getSizeInBytes());
    FAISS_ASSERT(this->hostdata_ != nullptr);
    this->devicedata_ = nullptr;
}


template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::PipeTensor(
        DataPtrType data,
        const IndexT sizes[Dim],
        const IndexT strides[Dim])
        : hostdata_(data), devicedata_(nullptr) {
    
    static_assert(Dim > 0, "PipeTensor must have > 0 dimensions");

    for (int i = 0; i < Dim; i++) {
        size_[i] = sizes[i];
        stride_[i] = strides[i];
    }
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT>::PipeTensor(
        DataPtrType data, DataPtrType data2, 
        const IndexT sizes[Dim],
        const IndexT strides[Dim])
        : hostdata_(data), devicedata_(data2) {
    
    static_assert(Dim > 0, "PipeTensor must have > 0 dimensions");

    for (int i = 0; i < Dim; i++) {
        size_[i] = sizes[i];
        stride_[i] = strides[i];
    }
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ void PipeTensor<T, Dim, InnerContig, IndexT>::copyFrom(
        const PipeTensor<T, Dim, InnerContig, IndexT>& t,
        cudaStream_t stream) {
    // The tensor must be fully contiguous
    GPU_FAISS_ASSERT(this->isContiguous());

    // Size must be the same (since dimensions are checked and
    // continuity is assumed, we need only check total number of
    // elements
    GPU_FAISS_ASSERT(this->numElements() == t.numElements());

    // Handle Device data
    if (t.numElements() > 0 && t.devicedata()) {
        GPU_FAISS_ASSERT(this->devicedata_);

        int ourDev = getDeviceForAddress(this->devicedata_);
        GPU_FAISS_ASSERT(ourDev >= 0);
        int tDev = getDeviceForAddress(t.devicedata());
        GPU_FAISS_ASSERT(tDev >= 0);

        CUDA_VERIFY(cudaMemcpyAsync(
                this->devicedata_,
                t.devicedata(),
                this->getSizeInBytes(),
                cudaMemcpyDeviceToDevice,
                stream));
    }

    // Handle Host data
    if (t.numElements() > 0 && t.hostdata()) {
        GPU_FAISS_ASSERT(this->hostdata_);

        int ourDev = getDeviceForAddress(this->hostdata_);
        GPU_FAISS_ASSERT(ourDev == -1);
        int tDev = getDeviceForAddress(t.hostdata());
        GPU_FAISS_ASSERT(tDev == -1);

        CUDA_VERIFY(cudaMemcpyAsync(
                this->hostdata_,
                t.hostdata(),
                this->getSizeInBytes(),
                cudaMemcpyHostToHost,
                stream));
    }
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ void PipeTensor<T, Dim, InnerContig, IndexT>::copyTo(
        PipeTensor<T, Dim, InnerContig, IndexT>& t,
        cudaStream_t stream) {
    // The tensor must be fully contiguous
    GPU_FAISS_ASSERT(this->isContiguous());

    // Size must be the same (since dimensions are checked and
    // continuity is assumed, we need only check total number of
    // elements
    GPU_FAISS_ASSERT(this->numElements() == t.numElements());

    // Handle Device data
    if (t.numElements() > 0 && this->devicedata_) {
        GPU_FAISS_ASSERT(this->devicedata_);

        int ourDev = getDeviceForAddress(this->devicedata_);
        GPU_FAISS_ASSERT(ourDev >= 0);
        int tDev = getDeviceForAddress(t.devicedata());
        GPU_FAISS_ASSERT(tDev >= 0);

        CUDA_VERIFY(cudaMemcpyAsync(
                t.devicedata(),
                this->devicedata_,
                this->getSizeInBytes(),
                cudaMemcpyDeviceToDevice,
                stream));
    }

    // Handle Host data
    if (t.numElements() > 0 && this->hostdata_) {
        GPU_FAISS_ASSERT(this->hostdata_);

        int ourDev = getDeviceForAddress(this->hostdata_);
        GPU_FAISS_ASSERT(ourDev == -1);
        int tDev = getDeviceForAddress(t.hostdata());
        GPU_FAISS_ASSERT(tDev == -1);

        CUDA_VERIFY(cudaMemcpyAsync(
                t.hostdata(),
                this->hostdata_,
                this->getSizeInBytes(),
                cudaMemcpyHostToHost,
                stream));
    }
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ void PipeTensor<T, Dim, InnerContig, IndexT>::memh2d(cudaStream_t stream) {
    if (!devicedata_){
        this->devicedata_ = (DataPtrType)pgr->allocTemMemory(this->getSizeInBytes());
        this->state_ = AllocState::BothOwner;
        FAISS_ASSERT(this->devicedata_ != nullptr);
    }

    CUDA_VERIFY(cudaMemcpyAsync(
                this->devicedata_,
                this->hostdata_,
                this->getSizeInBytes(),
                cudaMemcpyHostToDevice,
                stream));
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ void PipeTensor<T, Dim, InnerContig, IndexT>::memd2h(cudaStream_t stream) {
    FAISS_ASSERT(this->devicedata_ != nullptr);

    CUDA_VERIFY(cudaMemcpyAsync(
                this->hostdata_,
                this->devicedata_,
                this->getSizeInBytes(),
                cudaMemcpyDeviceToHost,
                stream));
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ void PipeTensor<T, Dim, InnerContig, IndexT>::copyFrom(
        const std::vector<T>& v,
        cudaStream_t stream) {
    // The tensor must be fully contiguous
    GPU_FAISS_ASSERT(this->isContiguous());

    // Size must be the same
    GPU_FAISS_ASSERT(this->numElements() == v.size());

    if (v.size() > 0) {
        GPU_FAISS_ASSERT(this->hostdata_);
        int ourDev = getDeviceForAddress(this->hostdata_);
        GPU_FAISS_ASSERT(ourDev == -1);

        std::memcpy(
                this->hostdata_, v.data(), this->numElements() * sizeof(T));
    }
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ void PipeTensor<T, Dim, InnerContig, IndexT>::copyFrom(
        const DataPtrType v,
        cudaStream_t stream) {
    // The tensor must be fully contiguous
    GPU_FAISS_ASSERT(this->isContiguous());

    // Size must be the same
    // GPU_FAISS_ASSERT(this->numElements() == v.size());

    GPU_FAISS_ASSERT(this->hostdata_);
    int ourDev = getDeviceForAddress(this->hostdata_);
    GPU_FAISS_ASSERT(ourDev == -1);

    std::memcpy(
            this->hostdata_, v, this->numElements() * sizeof(T));
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ std::vector<T> PipeTensor<T, Dim, InnerContig, IndexT>::
        copyToVector(cudaStream_t stream) {
    // The tensor must be fully contiguous
    GPU_FAISS_ASSERT(this->isContiguous());

    std::vector<T> out(this->numElements());

    if (!out.empty()) {
        GPU_FAISS_ASSERT(this->hostdata_);
        int ourDev = getDeviceForAddress(this->hostdata_);
        GPU_FAISS_ASSERT(ourDev == -1);

        std::memcpy(
                out.data(), this->hostdata_, this->numElements() * sizeof(T));
    }

    return out;
}


template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <typename OtherT, int OtherDim>
__host__ bool PipeTensor<T, Dim, InnerContig, IndexT>::isSame(
        const PipeTensor<OtherT, OtherDim, InnerContig, IndexT>& rhs)
        const {
    if (Dim != OtherDim) {
        return false;
    }

    for (int i = 0; i < Dim; ++i) {
        if (this->getSize(i) != rhs.getSize(i)) {
            return false;
        }

        if (this->getStride(i) != rhs.getStride(i)) {
            return false;
        }
    }

    return true;
}


template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <typename OtherT, int OtherDim>
__host__ bool PipeTensor<T, Dim, InnerContig, IndexT>::
        isSameSize(const PipeTensor<OtherT, OtherDim, InnerContig, IndexT>&
                        rhs) const {
    if (Dim != OtherDim) {
        return false;
    }

    for (int i = 0; i < Dim; ++i) {
        if (this->getSize(i) != rhs.getSize(i)) {
            return false;
        }
    }

    return true;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <typename U>
__host__ PipeTensor<U, Dim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::cast() {
    static_assert(sizeof(U) == sizeof(T), "cast must be to same size object");

    return PipeTensor<U, Dim, InnerContig, IndexT>(
            reinterpret_cast<U*>(hostdata_), 
            reinterpret_cast<U*>(devicedata_),size_, stride_);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <typename U>
__host__ const PipeTensor<U, Dim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::cast() const {
    static_assert(sizeof(U) == sizeof(T), "cast must be to same size object");

    return PipeTensor<U, Dim, InnerContig, IndexT>(
            reinterpret_cast<U*>(hostdata_), 
            reinterpret_cast<U*>(devicedata_),size_, stride_);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <typename U>
__host__ PipeTensor<U, Dim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::castResize() {
    static_assert(sizeof(U) >= sizeof(T), "only handles greater sizes");
    constexpr int kMultiple = sizeof(U) / sizeof(T);

    GPU_FAISS_ASSERT(canCastResize<U>());

    IndexT newSize[Dim];
    IndexT newStride[Dim];

    for (int i = 0; i < Dim - 1; ++i) {
        newSize[i] = size_[i];
        newStride[i] = stride_[i] / kMultiple;
    }

    newStride[Dim - 1] = 1; // this is the same as the old stride
    newSize[Dim - 1] = size_[Dim - 1] / kMultiple;

    return PipeTensor<U, Dim, InnerContig, IndexT>(
            reinterpret_cast<U*>(hostdata_), 
            reinterpret_cast<U*>(devicedata_), newSize, newStride);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <typename U>
__host__ const PipeTensor<U, Dim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::castResize() const {
    return const_cast<PipeTensor<T, Dim, InnerContig, IndexT>*>(this)
            ->castResize<U>();
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <typename U>
__host__ bool PipeTensor<T, Dim, InnerContig, IndexT>::
        canCastResize() const {
    static_assert(sizeof(U) >= sizeof(T), "only handles greater sizes");
    constexpr int kMultiple = sizeof(U) / sizeof(T);

    // Ensure that the base pointer is sizeof(U) aligned
    if (((uintptr_t)hostdata_) % sizeof(U) != 0) {
        return false;
    }

    // Check all outer strides
    for (int i = 0; i < Dim - 1; ++i) {
        if (stride_[i] % kMultiple != 0) {
            return false;
        }
    }

    // Check inner size
    if (size_[Dim - 1] % kMultiple != 0) {
        return false;
    }

    if (stride_[Dim - 1] != 1) {
        return false;
    }

    return true;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <typename NewIndexT>
__host__ PipeTensor<T, Dim, InnerContig, NewIndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::castIndexType() const {
    if (sizeof(NewIndexT) < sizeof(IndexT)) {
        GPU_FAISS_ASSERT(this->canUseIndexType<NewIndexT>());
    }

    NewIndexT newSize[Dim];
    NewIndexT newStride[Dim];
    for (int i = 0; i < Dim; ++i) {
        newSize[i] = (NewIndexT)size_[i];
        newStride[i] = (NewIndexT)stride_[i];
    }

    return PipeTensor<T, Dim, InnerContig, NewIndexT>(
            hostdata_, devicedata_, newSize, newStride);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <typename NewIndexT>
__host__ bool PipeTensor<T, Dim, InnerContig, IndexT>::canUseIndexType()
        const {
    static_assert(sizeof(size_t) >= sizeof(IndexT), "index size too large");
    static_assert(
            sizeof(size_t) >= sizeof(NewIndexT), "new index size too large");

    // Find maximum offset that can be calculated
    // FIXME: maybe also consider offset in bytes? multiply by sizeof(T)?
    size_t maxOffset = 0;

    for (int i = 0; i < Dim; ++i) {
        size_t curMaxOffset = (size_t)size_[i] * (size_t)stride_[i];
        if (curMaxOffset > maxOffset) {
            maxOffset = curMaxOffset;
        }
    }

    if (maxOffset > (size_t)std::numeric_limits<NewIndexT>::max()) {
        return false;
    }

    return true;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ __device__ size_t
PipeTensor<T, Dim, InnerContig, IndexT>::numElements() const {
    size_t size = (size_t)getSize(0);

    for (int i = 1; i < Dim; ++i) {
        size *= (size_t)getSize(i);
    }

    return size;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ bool PipeTensor<T, Dim, InnerContig, IndexT>::
        isContiguous() const {
    long prevSize = 1;

    for (int i = Dim - 1; i >= 0; --i) {
        if (getSize(i) != (IndexT)1) {
            if (getStride(i) == prevSize) {
                prevSize *= getSize(i);
            } else {
                return false;
            }
        }
    }

    return true;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ bool PipeTensor<T, Dim, InnerContig, IndexT>::
        isConsistentlySized(int i) const {
    if (i == 0 && getStride(i) > 0 && getSize(i) > 0) {
        return true;
    } else if (
            (i > 0) && (i < Dim) && (getStride(i) > 0) &&
            ((getStride(i - 1) / getStride(i)) >= getSize(i))) {
        return true;
    }

    return false;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ bool PipeTensor<T, Dim, InnerContig, IndexT>::
        isConsistentlySized() const {
    for (int i = 0; i < Dim; ++i) {
        if (!isConsistentlySized(i)) {
            return false;
        }
    }

    return true;
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ bool PipeTensor<T, Dim, InnerContig, IndexT>::
        isContiguousDim(int i) const {
    return (i == Dim - 1) || // just in case
            ((i < Dim - 1) &&
             ((getStride(i) / getStride(i + 1)) == getSize(i + 1)));
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::transpose(int dim1, int dim2) const {
    GPU_FAISS_ASSERT(dim1 >= 0 && dim1 < Dim);
    GPU_FAISS_ASSERT(dim2 >= 0 && dim2 < Dim);

    // If a tensor is innermost contiguous, one cannot transpose the innermost
    // dimension
    if (InnerContig) {
        GPU_FAISS_ASSERT(dim1 != Dim - 1 && dim2 != Dim - 1);
    }

    IndexT newSize[Dim];
    IndexT newStride[Dim];

    for (int i = 0; i < Dim; ++i) {
        newSize[i] = size_[i];
        newStride[i] = stride_[i];
    }

    IndexT tmp = newSize[dim1];
    newSize[dim1] = newSize[dim2];
    newSize[dim2] = tmp;

    tmp = newStride[dim1];
    newStride[dim1] = newStride[dim2];
    newStride[dim2] = tmp;

    return PipeTensor<T, Dim, true, IndexT>
        (hostdata_, devicedata_, newSize, newStride);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ PipeTensor<T, Dim, false, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::transposeInnermost(int dim1) const {
    GPU_FAISS_ASSERT(dim1 >= 0 && dim1 < Dim);

    // We are exchanging with the innermost dimension
    int dim2 = 1;

    IndexT newSize[Dim];
    IndexT newStride[Dim];

    for (int i = 0; i < Dim; ++i) {
        newSize[i] = size_[i];
        newStride[i] = stride_[i];
    }

    IndexT tmp = newSize[dim1];
    newSize[dim1] = newSize[dim2];
    newSize[dim2] = tmp;

    tmp = newStride[dim1];
    newStride[dim1] = newStride[dim2];
    newStride[dim2] = tmp;

    return PipeTensor<T, Dim, false, IndexT>
        (hostdata_, devicedata_ ,newSize, newStride);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <int NewDim>
__host__ PipeTensor<T, NewDim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::upcastOuter() {
    // Can only create tensors of greater dimension
    static_assert(NewDim > Dim, "Can only upcast to greater dim");

    IndexT newSize[NewDim];
    IndexT newStride[NewDim];

    int shift = NewDim - Dim;

    for (int i = 0; i < NewDim; ++i) {
        if (i < shift) {
            // These are the extended dimensions
            newSize[i] = (IndexT)1;
            newStride[i] = size_[0] * stride_[0];
        } else {
            // Shift the remaining dimensions
            newSize[i] = size_[i - shift];
            newStride[i] = stride_[i - shift];
        }
    }

    return PipeTensor<T, NewDim, InnerContig, IndexT>(
            hostdata_, devicedata_, newSize, newStride);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <int NewDim>
__host__ PipeTensor<T, NewDim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::upcastInner() {
    // Can only create tensors of greater dimension
    static_assert(NewDim > Dim, "Can only upcast to greater dim");

    IndexT newSize[NewDim];
    IndexT newStride[NewDim];

    for (int i = 0; i < NewDim; ++i) {
        if (i < Dim) {
            // Existing dimensions get copied over
            newSize[i] = size_[i];
            newStride[i] = stride_[i];
        } else {
            // Extended dimensions
            newSize[i] = (IndexT)1;
            newStride[i] = (IndexT)1;
        }
    }

    return PipeTensor<T, NewDim, InnerContig, IndexT>(
            hostdata_, devicedata_, newSize, newStride);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <int NewDim>
__host__ PipeTensor<T, NewDim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::downcastOuter() {
    // Can only create tensors of lesser dimension
    static_assert(NewDim < Dim, "Can only downcast to lesser dim");

    // We can't downcast non-contiguous tensors, since it leaves
    // garbage data in the tensor. The tensor needs to be contiguous
    // in all of the dimensions we are collapsing (no padding in
    // them).
    for (int i = 0; i < Dim - NewDim; ++i) {
        bool cont = isContiguousDim(i);
        GPU_FAISS_ASSERT(cont);
    }

    IndexT newSize[NewDim];
    IndexT newStride[NewDim];

    int ignoredDims = Dim - NewDim;
    IndexT collapsedSize = 1;

    for (int i = 0; i < Dim; ++i) {
        if (i < ignoredDims) {
            // Collapse these dimensions
            collapsedSize *= getSize(i);
        } else {
            // Non-collapsed dimensions
            if (i == ignoredDims) {
                // This is the first non-collapsed dimension
                newSize[i - ignoredDims] = collapsedSize * getSize(i);
            } else {
                // Subsequent non-collapsed dimensions
                newSize[i - ignoredDims] = getSize(i);
            }

            newStride[i - ignoredDims] = getStride(i);
        }
    }

    return PipeTensor<T, NewDim, InnerContig, IndexT>(
            hostdata_, devicedata_, newSize, newStride);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <int NewDim>
__host__ PipeTensor<T, NewDim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::downcastInner() {
    // Can only create tensors of lesser dimension
    static_assert(NewDim < Dim, "Can only downcast to lesser dim");

    // We can't downcast non-contiguous tensors, since it leaves
    // garbage data in the tensor. The tensor needs to be contiguous
    // in all of the dimensions we are collapsing (no padding in
    // them).
    for (int i = NewDim; i < Dim; ++i) {
        GPU_FAISS_ASSERT(isContiguousDim(i));
    }

    IndexT newSize[NewDim];
    IndexT newStride[NewDim];

    IndexT collapsedSize = 1;

    for (int i = Dim - 1; i >= 0; --i) {
        if (i >= NewDim) {
            // Collapse these dimensions
            collapsedSize *= getSize(i);
        } else {
            // Non-collapsed dimensions
            if (i == NewDim - 1) {
                // This is the first non-collapsed dimension
                newSize[i] = collapsedSize * getSize(i);
                newStride[i] = getStride(Dim - 1);
            } else {
                // Subsequent non-collapsed dimensions
                newSize[i] = getSize(i);
                newStride[i] = getStride(i);
            }
        }
    }

    return PipeTensor<T, NewDim, InnerContig, IndexT>(
            hostdata_, devicedata_, newSize, newStride);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <int SubDim>
__host__ __device__ PipeTensor<T, SubDim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::hostview(DataPtrType at) {
    static_assert(
            SubDim >= 1 && SubDim < Dim, "can only create view of lesser dim");

    IndexT viewSizes[SubDim];
    IndexT viewStrides[SubDim];

    for (int i = 0; i < SubDim; ++i) {
        viewSizes[i] = size_[Dim - SubDim + i];
        viewStrides[i] = stride_[Dim - SubDim + i];
    }

    return PipeTensor<T, SubDim, InnerContig, IndexT>(
            at, nullptr, viewSizes, viewStrides);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <int SubDim>
__host__ __device__ PipeTensor<T, SubDim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::deviceview(DataPtrType at) {
    static_assert(
            SubDim >= 1 && SubDim < Dim, "can only create view of lesser dim");

    IndexT viewSizes[SubDim];
    IndexT viewStrides[SubDim];

    for (int i = 0; i < SubDim; ++i) {
        viewSizes[i] = size_[Dim - SubDim + i];
        viewStrides[i] = stride_[Dim - SubDim + i];
    }

    return PipeTensor<T, SubDim, InnerContig, IndexT>(
            nullptr, at, viewSizes, viewStrides);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <int SubDim>
__host__ __device__ PipeTensor<T, SubDim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::hostview() {
    return hostview<SubDim>(hostdata_);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <int SubDim>
__host__ __device__ PipeTensor<T, SubDim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::deviceview() {
    return deviceview<SubDim>(devicedata_);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ __device__ PipeTensor<T, Dim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::narrowOutermost(IndexT start, IndexT size) {
    return this->narrow(0, start, size);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ __device__ PipeTensor<T, Dim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::narrow(int dim, IndexT start, IndexT size) {
    DataPtrType hostData = hostdata_;
    DataPtrType deviceData = devicedata_;

    GPU_FAISS_ASSERT(
            start >= 0 && start < size_[dim] && (start + size) <= size_[dim]);

    if (start > 0) {
        if (hostData)
            hostData += (size_t)start * stride_[dim];
        if (deviceData)
            deviceData += (size_t)start * stride_[dim];
    }

    IndexT newSize[Dim];
    for (int i = 0; i < Dim; ++i) {
        if (i == dim) {
            GPU_FAISS_ASSERT(start + size <= size_[dim]);
            newSize[i] = size;
        } else {
            newSize[i] = size_[i];
        }
    }

    // If we were innermost contiguous before, we are still innermost contiguous
    return PipeTensor<T, Dim, InnerContig, IndexT>(
            hostData, deviceData, newSize, stride_);
}

template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
template <int NewDim>
__host__ __device__ PipeTensor<T, NewDim, InnerContig, IndexT> PipeTensor<
        T,
        Dim,
        InnerContig,
        IndexT>::hostview(std::initializer_list<IndexT> sizes) {
    GPU_FAISS_ASSERT(this->isContiguous());

    GPU_FAISS_ASSERT(sizes.size() == NewDim);

    // The total size of the new view must be the same as the total size
    // of the old view
    size_t curSize = numElements();
    size_t newSize = 1;

    for (auto s : sizes) {
        newSize *= s;
    }

    GPU_FAISS_ASSERT(curSize == newSize);
    return PipeTensor<T, NewDim, true, IndexT>(hostdata(), sizes);
}

} // namespace gpu
} // namespace faiss