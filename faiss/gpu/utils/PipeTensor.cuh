/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <initializer_list>
#include <vector>

#include <faiss/pipe/PipeCluster.h>
#include <faiss/gpu/PipeGpuResources.h>

/// Multi-dimensional array class for unified usage (Pipe transmission use PipeGpuResource).
/// Originally from Facebook's fbcunn, since added to the Torch GPU
/// library cutorch as well.

namespace faiss {
namespace gpu {

/// Pipe tensor type (Only use default PtrTraits)
template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
class PipeTensor;

/// Type of a subspace of a tensor
namespace detail {
template <
        typename TensorType,
        int SubDim,
        typename U>
class PipeSubTensor;
}


namespace traits {

template <typename T>
struct PipeRestrictPtrTraits {
    typedef T* __restrict__ PtrType;
};

template <typename T>
struct PipeDefaultPtrTraits {
    typedef T* PtrType;
};
}

/**
   Templated multi-dimensional array that supports strided access of
   elements. Main access is through `operator[]`; e.g.,
   `tensor[x][y][z]`.

   - `T` is the contained type (e.g., `float`)
   - `Dim` is the tensor rank
   - If `InnerContig` is true, then the tensor is assumed to be innermost
   - contiguous, and only operations that make sense on contiguous
   - arrays are allowed (e.g., no transpose). Strides are still
   - calculated, but innermost stride is assumed to be 1.
   - `IndexT` is the integer type used for size/stride arrays, and for
   - all indexing math. Default is `int`, but for large tensors, `long`
   - can be used instead.
*/
template <
        typename T,
        int Dim,
        bool InnerContig = false,
        typename IndexT = int>
class PipeTensor {

public:
    enum { NumDim = Dim };
    typedef T DataType;
    typedef IndexT IndexType;
    enum { IsInnerContig = InnerContig };
    typedef typename traits::PipeDefaultPtrTraits<T>::PtrType DataPtrType;
    typedef PipeTensor<T, Dim, InnerContig, IndexT> PipeTensorType;

    /// Set the two resources
    __host__ void setResources(PipeCluster *p1, PipeGpuResources *p2)
    {pc = p1; pgr = p2;}

    /// Default constructor
    __host__  PipeTensor();

    /// Destructor
    __host__ ~PipeTensor();

    /// Copy constructor
    __host__
    PipeTensor(PipeTensor<T, Dim, InnerContig, IndexT>& t);

    /// Move constructor
    __host__
    PipeTensor(PipeTensor<T, Dim, InnerContig, IndexT>&& t);

    /// Assignment
    __host__ __device__ PipeTensor<T, Dim, InnerContig, IndexT>&
    operator=(PipeTensor<T, Dim, InnerContig, IndexT>& t);

    /// Move assignment
    __host__ __device__ PipeTensor<T, Dim, InnerContig, IndexT>&
    operator=(PipeTensor<T, Dim, InnerContig, IndexT>&& t);

    /// Constructor that calculates strides with no padding
    /// zili doesn't recommend you to use the following two API to create PipeTensor straightly
    __host__
    PipeTensor(DataPtrType data, const IndexT sizes[Dim]);
    __host__
    PipeTensor(DataPtrType data, std::initializer_list<IndexT> sizes);

    /// Constructs a tensor of the given size, allocating memory for it
    /// locally
    __host__ PipeTensor(const IndexT sizes[Dim], PipeCluster* p);
    __host__ PipeTensor(std::initializer_list<IndexT> sizes, PipeCluster* p);

    /// Constructor that takes arbitrary size/stride arrays.
    /// Errors if you attempt to pass non-contiguous strides to a
    /// contiguous tensor.
    __host__
    PipeTensor(DataPtrType data,
           const IndexT sizes[Dim],
           const IndexT strides[Dim]);

    /// Constructor that takes arbitrary size/stride arrays.
    /// Errors if you attempt to pass non-contiguous strides to a
    /// contiguous tensor.
    __host__ __device__
    PipeTensor(DataPtrType data, DataPtrType data2,
           const IndexT sizes[Dim],
           const IndexT strides[Dim]);

    /// Copies a tensor into ourselves; sizes must match
    __host__ void copyFrom(
            const PipeTensor<T, Dim, InnerContig, IndexT>& t,
            cudaStream_t stream = 0);

    /// Copies ourselves into a tensor; sizes must match
    __host__ void copyTo(
            PipeTensor<T, Dim, InnerContig, IndexT>& t,
            cudaStream_t stream = 0);

    /// Data transfer
    __host__ void memh2d(cudaStream_t stream = 0);

    /// Data transfer
    __host__ void memd2h(cudaStream_t stream = 0);

    /// Reserve device temp memory
    __host__ void reserve();


    /// Copies a CPU std::vector<T> into ourselves, allocating memory for it.
    /// The total size of our Tensor must match vector<T>::size(), though
    /// we are not restricted to 1D Tensors to match the 1D vector<T>.
    /// `stream` specifies the stream of the copy and thus the stream on which
    /// the memory will initially be used.
    __host__ void copyFrom(const std::vector<T>& v, cudaStream_t stream = 0);

    /// Copies a CPU Pointer (no pin memory) to us with pinned memory
    __host__ void copyFrom(const DataPtrType v, cudaStream_t stream = 0);

    /// Copies ourselves into a flattened (1D) std::vector, using the given
    /// stream
    __host__ std::vector<T> copyToVector(cudaStream_t stream = 0);

    /// Returns true if the two tensors are of the same dimensionality,
    /// size and stride.
    template <typename OtherT, int OtherDim>
    __host__ bool isSame(
            const PipeTensor<OtherT, OtherDim, InnerContig, IndexT>& rhs)
            const;

    /// Returns true if the two tensors are of the same dimensionality and size(exclude stride)
    template <typename OtherT, int OtherDim>
    __host__ bool isSameSize(
            const PipeTensor<OtherT, OtherDim, InnerContig, IndexT>& rhs)
            const;

    /// Cast to a tensor of a different type of the same size and
    /// stride. U and our type T must be of the same size
    template <typename U>
    __host__ PipeTensor<U, Dim, InnerContig, IndexT> cast();

    /// Const version of `cast`
    template <typename U>
    __host__ const PipeTensor<U, Dim, InnerContig, IndexT>
    cast() const;

    /// Cast to a tensor of a different type which is potentially a
    /// different size than our type T. Tensor must be aligned and the
    /// innermost dimension must be a size that is a multiple of
    /// sizeof(U) / sizeof(T), and the stride of the innermost dimension
    /// must be contiguous. The stride of all outer dimensions must be a
    /// multiple of sizeof(U) / sizeof(T) as well.
    template <typename U>
    __host__ PipeTensor<U, Dim, InnerContig, IndexT>
    castResize();

    /// Const version of `castResize`
    template <typename U>
    __host__ const PipeTensor<U, Dim, InnerContig, IndexT>
    castResize() const;

    /// Returns true if we can castResize() this tensor to the new type
    template <typename U>
    __host__ bool canCastResize() const;

    /// Attempts to cast this tensor to a tensor of a different IndexT.
    /// Fails if size or stride entries are not representable in the new
    /// IndexT.
    template <typename NewIndexT>
    __host__ PipeTensor<T, Dim, InnerContig, NewIndexT> castIndexType()
            const;

    /// Returns true if we can use this indexing type to access all elements
    /// index type
    template <typename NewIndexT>
    __host__ bool canUseIndexType() const;

    /// Returns a raw pointer to the start of our data.
    __host__ __device__ inline DataPtrType hostdata() {
        return hostdata_;
    }

    /// Returns a raw pointer to the end of our data, assuming
    /// continuity
    __host__ __device__ inline DataPtrType hostend() {
        return hostdata() + numElements();
    }

    /// Returns a raw pointer to the start of our data (const).
    __host__ __device__ inline const DataPtrType hostdata() const {
        return hostdata_;
    }

    /// Returns a raw pointer to the end of our data, assuming
    /// continuity (const)
    __host__ __device__ inline DataPtrType hostend() const {
        return hostdata() + numElements();
    }

    /// Returns a raw pointer to the start of our data.
    __host__ __device__ inline DataPtrType devicedata() {
        GPU_FAISS_ASSERT(this->devicedata_ != nullptr);
        return devicedata_;
    }

    /// Returns a raw pointer to the end of our data, assuming
    /// continuity
    __host__ __device__ inline DataPtrType deviceend() {
        GPU_FAISS_ASSERT(this->devicedata_ != nullptr);
        return devicedata() + numElements();
    }

    /// Returns a raw pointer to the start of our data (const).
    __host__ __device__ inline const DataPtrType devicedata() const {
        GPU_FAISS_ASSERT(this->devicedata_ != nullptr);
        return devicedata_;
    }

    /// Returns a raw pointer to the end of our data, assuming
    /// continuity (const)
    __host__ __device__ inline DataPtrType deviceend() const {
        GPU_FAISS_ASSERT(this->devicedata_ != nullptr);
        return devicedata() + numElements();
    }

    /// Cast to a different datatype
    template <typename U>
    __host__ inline U* hostdataAs() {
        return reinterpret_cast<U*>(hostdata_);
    }

    /// Cast to a different datatype
    template <typename U>
    __host__ inline const U*
    hostdataAs() const {
        return reinterpret_cast<U*>(hostdata_);
    }

    /// Cast to a different datatype
    template <typename U>
    __host__ inline U* devicedataAs() {
        return reinterpret_cast<U*>(devicedata_);
    }

    /// Cast to a different datatype
    template <typename U>
    __host__ inline const U* 
    devicedataAs() const {
        return reinterpret_cast<U*>(devicedata_);
    }

    /// Returns a read/write view of a portion of our tensor, hostdata_.
    __host__ __device__ inline detail::PipeSubTensor<PipeTensorType, Dim - 1, T>
    operator[](IndexT);

    /// Returns a read/write view of a portion of our tensor, devicedata_.
    __host__ __device__ inline detail::PipeSubTensor<PipeTensorType, Dim - 1, T>
    operator()(IndexT);


    /// Returns a read/write view of a portion of our tensor, hostdata_.
    __host__ __device__ inline const detail::
            PipeSubTensor<PipeTensorType, Dim - 1, T>
            operator[](IndexT) const;

    /// Returns a read/write view of a portion of our tensor, devicedata_.
    __host__ __device__ inline const detail::
            PipeSubTensor<PipeTensorType, Dim - 1, T>
            operator()(IndexT) const;


    /// Returns the size of a given dimension, `[0, Dim - 1]`. No bounds
    /// checking.
    __host__ __device__ inline IndexT getSize(int i) const {
        return size_[i];
    }

    /// Returns the stride of a given dimension, `[0, Dim - 1]`. No bounds
    /// checking.
    __host__ __device__ inline IndexT getStride(int i) const {
        return stride_[i];
    }

    /// Returns the total number of elements contained within our data
    /// (product of `getSize(i)`)
    __host__ __device__ size_t numElements() const;

    /// If we are contiguous, returns the total size in bytes of our
    /// data
    __host__ __device__ size_t getSizeInBytes() const {
        return numElements() * sizeof(T);
    }

    /// Returns the size array.
    __host__ __device__ inline const IndexT* sizes() const {
        return size_;
    }

    /// Returns the stride array.
    __host__ __device__ inline const IndexT* strides() const {
        return stride_;
    }

    /// Returns true if there is no padding within the tensor and no
    /// re-ordering of the dimensions.
    /// ~~~
    /// (stride(i) == size(i + 1) * stride(i + 1)) && stride(dim - 1) == 0
    /// ~~~
    __host__ bool isContiguous() const;

    /// Returns whether a given dimension has only increasing stride
    /// from the previous dimension. A tensor that was permuted by
    /// exchanging size and stride only will fail this check.
    /// If `i == 0` just check `size > 0`. Returns `false` if `stride` is `<=
    /// 0`.
    __host__ bool isConsistentlySized(int i) const;

    // Returns whether at each dimension `stride <= size`.
    // If this is not the case then iterating once over the size space will
    // touch the same memory locations multiple times.
    __host__ bool isConsistentlySized() const;

    /// Returns true if the given dimension index has no padding
    __host__ bool isContiguousDim(int i) const;

    /// Returns a tensor of the same dimension after transposing the two
    /// dimensions given. Does not actually move elements; transposition
    /// is made by permuting the size/stride arrays.
    /// If the dimensions are not valid, asserts.
    __host__ PipeTensor<T, Dim, InnerContig, IndexT> transpose(
            int dim1,
            int dim2) const;

    /// Transpose a tensor, exchanging a non-innermost dimension with the
    /// innermost dimension, returning a no longer innermost contiguous tensor
    __host__ PipeTensor<T, Dim, false, IndexT>
    transposeInnermost(int dim1) const;

    /// Upcast a tensor of dimension `D` to some tensor of dimension
    /// D' > D by padding the leading dimensions by 1
    /// e.g., upcasting a 2-d tensor `[2][3]` to a 4-d tensor `[1][1][2][3]`
    template <int NewDim>
    __host__ PipeTensor<T, NewDim, InnerContig, IndexT>
    upcastOuter();

    /// Upcast a tensor of dimension `D` to some tensor of dimension
    /// D' > D by padding the lowest/most varying dimensions by 1
    /// e.g., upcasting a 2-d tensor `[2][3]` to a 4-d tensor `[2][3][1][1]`
    template <int NewDim>
    __host__ PipeTensor<T, NewDim, InnerContig, IndexT>
    upcastInner();

    /// Downcast a tensor of dimension `D` to some tensor of dimension
    /// D' < D by collapsing the leading dimensions. asserts if there is
    /// padding on the leading dimensions.
    template <int NewDim>
    __host__ PipeTensor<T, NewDim, InnerContig, IndexT>
    downcastOuter();

    /// Downcast a tensor of dimension `D` to some tensor of dimension
    /// D' < D by collapsing the leading dimensions. asserts if there is
    /// padding on the leading dimensions.
    template <int NewDim>
    __host__ PipeTensor<T, NewDim, InnerContig, IndexT>
    downcastInner();

    /// Returns a tensor that is a view of the `SubDim`-dimensional slice
    /// of this tensor, starting at `at`.
    template <int SubDim>
    __host__ __device__ PipeTensor<T, SubDim, InnerContig, IndexT> hostview(
            DataPtrType at);

    /// Returns a tensor that is a view of the `SubDim`-dimensional slice
    /// of this tensor, starting at `at`.
    template <int SubDim>
    __host__ __device__ PipeTensor<T, SubDim, InnerContig, IndexT> deviceview(
            DataPtrType at);

    /// Returns a tensor that is a view of the `SubDim`-dimensional slice
    /// of this tensor, starting where our data begins
    template <int SubDim>
    __host__ __device__ PipeTensor<T, SubDim, InnerContig, IndexT> hostview();

    /// Returns a tensor that is a view of the `SubDim`-dimensional slice
    /// of this tensor, starting where our data begins
    template <int SubDim>
    __host__ __device__ PipeTensor<T, SubDim, InnerContig, IndexT> deviceview();

    /// Returns a tensor of the same dimension that is a view of the
    /// original tensor with the specified dimension restricted to the
    /// elements in the range [start, start + size)
    __host__ __device__ PipeTensor<T, Dim, InnerContig, IndexT>
    narrowOutermost(IndexT start, IndexT size);

    /// Returns a tensor of the same dimension that is a view of the
    /// original tensor with the specified dimension restricted to the
    /// elements in the range [start, start + size).
    /// Can occur in an arbitrary dimension
    __host__ __device__ PipeTensor<T, Dim, InnerContig, IndexT> narrow(
            int dim,
            IndexT start,
            IndexT size);

    /// Returns a view of the given tensor expressed as a tensor of a
    /// different number of dimensions.
    /// Only works if we are contiguous.
    template <int NewDim>
    __host__ __device__ PipeTensor<T, NewDim, InnerContig, IndexT> hostview(
            std::initializer_list<IndexT> sizes);

    /// Returns a view of the given tensor expressed as a tensor of a
    /// different number of dimensions.
    /// Only works if we are contiguous.
    template <int NewDim>
    __host__ __device__ PipeTensor<T, NewDim, InnerContig, IndexT> deviceview(
            std::initializer_list<IndexT> sizes);

protected:
    /// Raw pointer to where the tensor data begins in host memory
    DataPtrType hostdata_;

    /// Raw pointer to where the tensor data begins in device memory
    DataPtrType devicedata_;

    /// Array of strides (in sizeof(T) terms) per each dimension
    IndexT stride_[Dim];

    /// Size per each dimension
    IndexT size_[Dim];

    /// Resource to allocate host memory
    PipeCluster *pc = nullptr;

    /// Resource to allocate device memory
    PipeGpuResources *pgr = nullptr;

private:
    enum AllocState {
        /// This tensor itself owns the host and device memory, which must be freed via
        /// cudaFree and free
        BothOwner,

        /// This tensor itself only owns the host memory, which must be freed via
        /// free
        HostOwner,

        /// This tensor itself only owns the device memory, which must be freed via
        /// PipeResource->freeTemp()
        DeviceOwner,

        /// This tensor itself is not an owner of the either memory; there is
        /// nothing to free
        NotOwner,
    };

    AllocState state_ = AllocState::NotOwner;

};

namespace detail {

/// Specialization for a view of a single value (0-dimensional)


/// Specialization for a view of a single value (0-dimensional)
template <typename TensorType, typename U>
class PipeSubTensor<TensorType, 0, U> {
   public:
    __host__ __device__ PipeSubTensor<TensorType, 0, U> operator=(
            typename TensorType::DataType val) {
        *data_ = val;
        return *this;
    }

    // operator T&
    __host__ __device__ operator typename TensorType::DataType &() {
        return *data_;
    }

    // const operator T& returning const T&
    __host__ __device__ operator const typename TensorType::DataType &() const {
        return *data_;
    }

    // operator& returning T*
    __host__ __device__ typename TensorType::DataType* operator&() {
        return data_;
    }

    // const operator& returning const T*
    __host__ __device__ const typename TensorType::DataType* operator&() const {
        return data_;
    }

    /// Returns a raw accessor to our slice.
    __host__ __device__ inline typename TensorType::DataPtrType data() {
        return data_;
    }

    /// Returns a raw accessor to our slice (const).
    __host__ __device__ inline const typename TensorType::DataPtrType data()
            const {
        return data_;
    }

    /// Cast to a different datatype.
    template <typename T>
    __host__ __device__ T& as() {
        return *dataAs<T>();
    }

    /// Cast to a different datatype (const).
    template <typename T>
    __host__ __device__ const T& as() const {
        return *dataAs<T>();
    }

    /// Cast to a different datatype
    template <typename T>
    __host__ __device__ inline T* dataAs() {
        return reinterpret_cast<T*>(data_);
    }

    /// Cast to a different datatype (const)
    template <typename T>
    __host__ __device__ inline T* dataAs()
            const {
        return reinterpret_cast<T*>(data_);
    }

    /// Use the texture cache for reads
    __device__ inline typename TensorType::DataType ldg() const {
#if __CUDA_ARCH__ >= 350
        return __ldg(data_);
#else
        return *data_;
#endif
    }

    /// Use the texture cache for reads; cast as a particular type
    template <typename T>
    __device__ inline T ldgAs() const {
#if __CUDA_ARCH__ >= 350
        return __ldg(dataAs<T>());
#else
        return as<T>();
#endif
    }

   protected:
    /// One dimension greater can create us
    friend class PipeSubTensor<TensorType, 1, U>;

    /// Our parent tensor can create us
    friend class PipeTensor<
            typename TensorType::DataType,
            1,
            TensorType::IsInnerContig,
            typename TensorType::IndexType>;

    __host__ __device__ inline PipeSubTensor(
            TensorType& t,
            typename TensorType::DataPtrType data)
            : tensor_(t), data_(data) {}

    /// The tensor we're referencing
    TensorType& tensor_;

    /// Where our value is located
    typename TensorType::DataPtrType const data_;
};





/// PipeSubTensor is either a GPU tensor or a CPU tensor
template <typename TensorType, int SubDim, typename U>
class PipeSubTensor{
public:
    /// Returns a view of the data located at our offset (the dimension
    /// `SubDim` - 1 tensor).
    __host__ __device__ inline PipeSubTensor<TensorType, SubDim - 1, U>
    operator[](typename TensorType::IndexType index) {
        if (TensorType::IsInnerContig && SubDim == 1) {
            // Innermost dimension is stride 1 for contiguous arrays
            return PipeSubTensor<TensorType, SubDim - 1, U>(
                    tensor_, data_ + index);
        } else {
            return PipeSubTensor<TensorType, SubDim - 1, U>(
                    tensor_,
                    data_ +
                            index *
                                    tensor_.getStride(
                                            TensorType::NumDim - SubDim));
        }
    }

    /// Returns a view of the data located at our offset (the dimension
    /// `SubDim` - 1 tensor) (const).
    __host__ __device__ inline const PipeSubTensor<
            TensorType,
            SubDim - 1,
            U>
    operator[](typename TensorType::IndexType index) const {
        if (TensorType::IsInnerContig && SubDim == 1) {
            // Innermost dimension is stride 1 for contiguous arrays
            return PipeSubTensor<TensorType, SubDim - 1, U>(
                    tensor_, data_ + index);
        } else {
            return PipeSubTensor<TensorType, SubDim - 1, U>(
                    tensor_,
                    data_ +
                            index *
                                    tensor_.getStride(
                                            TensorType::NumDim - SubDim));
        }
    }

    // operator& returning T*
    __host__ __device__ typename TensorType::DataType* operator&() {
        return data_;
    }

    // const operator& returning const T*
    __host__ __device__ const typename TensorType::DataType* operator&() const {
        return data_;
    }

    /// Returns a raw accessor to our slice.
    __host__ __device__ inline typename TensorType::DataPtrType data() {
        return data_;
    }

    /// Returns a raw accessor to our slice (const).
    __host__ __device__ inline const typename TensorType::DataPtrType data()
            const {
        return data_;
    }

    /// Cast to a different datatype.
    template <typename T>
    __host__ __device__ T& as() {
        return *dataAs<T>();
    }

    /// Cast to a different datatype (const).
    template <typename T>
    __host__ __device__ const T& as() const {
        return *dataAs<T>();
    }

    /// Cast to a different datatype
    template <typename T>
    __host__ __device__ inline T* dataAs() {
        return reinterpret_cast<T*>(data_);
    }

    /// Cast to a different datatype (const)
    template <typename T>
    __host__ __device__ inline T* dataAs()
            const {
        return reinterpret_cast<T*>(data_);
    }

    /// Use the texture cache for reads
    __device__ inline typename TensorType::DataType ldg() const {
#if __CUDA_ARCH__ >= 350
        return __ldg(data_);
#else
        return *data_;
#endif
    }

    /// Use the texture cache for reads; cast as a particular type
    template <typename T>
    __device__ inline T ldgAs() const {
#if __CUDA_ARCH__ >= 350
        return __ldg(dataAs<T>());
#else
        return as<T>();
#endif
    }

    /// Returns a tensor that is a view of the SubDim-dimensional slice
    /// of this tensor, starting where our data begins
    PipeTensor<typename TensorType::DataType,
           SubDim,
           TensorType::IsInnerContig,
           typename TensorType::IndexType>
    view() {
        return tensor_.template view<SubDim>(data_);
    }

    protected:
    /// One dimension greater can create us
    friend class PipeSubTensor<TensorType, SubDim + 1, U>;

    /// Our parent tensor can create us
    friend class PipeTensor<
            typename TensorType::DataType,
            TensorType::NumDim,
            TensorType::IsInnerContig,
            typename TensorType::IndexType>;

    __host__ __device__ inline PipeSubTensor(
            TensorType& t,
            typename TensorType::DataPtrType data)
            : tensor_(t), data_(data) {}

    /// The tensor we're referencing
    TensorType& tensor_;

    /// The start of our sub-region
    typename TensorType::DataPtrType const data_;

};

} // namespace detail

/// [] can only acess host data
template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ __device__ inline detail::PipeSubTensor<
        PipeTensor<T, Dim, InnerContig, IndexT>,
        Dim - 1,
        T>
PipeTensor<T, Dim, InnerContig, IndexT>::operator[](IndexT index) {
    return detail::PipeSubTensor<PipeTensorType, Dim - 1, T>(
            detail::PipeSubTensor<PipeTensorType, Dim, T>(*this, hostdata_)[index]);
}

/// () can only acess device data
template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ __device__ inline detail::PipeSubTensor<
        PipeTensor<T, Dim, InnerContig, IndexT>,
        Dim - 1,
        T>
PipeTensor<T, Dim, InnerContig, IndexT>::operator()(IndexT index) {
    return detail::PipeSubTensor<PipeTensorType, Dim - 1, T>(
            detail::PipeSubTensor<PipeTensorType, Dim, T>(*this, devicedata_)[index]);
}


template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ __device__ inline const detail::PipeSubTensor<
        PipeTensor<T, Dim, InnerContig, IndexT>,
        Dim - 1,
        T>
PipeTensor<T, Dim, InnerContig, IndexT>::operator[](IndexT index) const {
    return detail::PipeSubTensor<PipeTensorType, Dim - 1, T>(
            detail::PipeSubTensor<PipeTensorType, Dim, T>(
                    const_cast<PipeTensorType&>(*this), hostdata_)[index]);
}


template <
        typename T,
        int Dim,
        bool InnerContig,
        typename IndexT>
__host__ __device__ inline const detail::PipeSubTensor<
        PipeTensor<T, Dim, InnerContig, IndexT>,
        Dim - 1,
        T>
PipeTensor<T, Dim, InnerContig, IndexT>::operator()(IndexT index) const {
    return detail::PipeSubTensor<PipeTensorType, Dim - 1, T>(
            detail::PipeSubTensor<PipeTensorType, Dim, T>(
                    const_cast<PipeTensorType&>(*this), devicedata_)[index]);
}




} // namespace gpu
} // namespace faiss

#include <faiss/gpu/utils/PipeTensor-inl.cuh>