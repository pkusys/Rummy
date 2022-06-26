/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <omp.h>
#include <cinttypes>
#include <stdint.h>
#include <algorithm>
#include <mutex>
#include <string.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/pipe/IndexIVFPipe.h>
#include <faiss/utils/utils.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/Heap.h>


namespace faiss {


/*************************************************************************
 * IndexIVFStats
 *************************************************************************/

void IndexIVFStats::reset() {
    memset((void*)this, 0, sizeof(*this));
}

void IndexIVFStats::add(const IndexIVFStats& other) {
    nq += other.nq;
    nlist += other.nlist;
    ndis += other.ndis;
    nheap_updates += other.nheap_updates;
    quantization_time += other.quantization_time;
    search_time += other.search_time;
}

IndexIVFStats indexIVF_stats;



/*****************************************
 * IndexIVFPipe implementation
 ******************************************/

using ScopedIds = InvertedLists::ScopedIds;
using ScopedCodes = InvertedLists::ScopedCodes;

IndexIVFPipe::IndexIVFPipe(
            size_t d_,
            size_t nlist_,
            MetricType metric_type_,
            IndexIVFPipeConfig config_)
            : d(d_), Index(d_, metric), nprobe (1), ivfPipeConfig_(config_),\
            verbose(false), balanced(false), code_size(d * sizeof(float)) {

                //initialize GPU resources.
                FAISS_THROW_IF_NOT_FMT(
                        config_.device < getNumDevices(),
                        "Invalid GPU device %d",
                        config_.device);

                FAISS_THROW_IF_NOT_MSG(dims > 0, "Invalid number of dimensions");

                FAISS_THROW_IF_NOT_FMT(
                        config_.memorySpace == MemorySpace::Device ||
                                (config_.memorySpace == MemorySpace::Unified &&
                                getFullUnifiedMemSupport(config_.device)),
                        "Device %d does not support full CUDA 8 Unified Memory (CC 6.0+)",
                        config.device);

                resources_ = provider.getResources();
                FAISS_ASSERT((bool)resources_);
                resources_->initializeForDevice(config_.device);

                //initialize invlists and related params.
                FAISS_THROW_IF_NOT_MSG(nlist > 0, "nlist must be > 0");
                direct_map = new DirectMap();
                nlist = nlist_;
                quantizer = nullptr;
                invlists = new ArrayInvertedLists(nlist, code_size);
                pipe_cluster = nullptr;

                // Spherical by default if the metric is inner_product
                if (metric_type == METRIC_INNER_PRODUCT) {
                    cp.spherical = true;
                }
                // here we set a low # iterations because this is typically used
                // for large clusterings
                cp.niter = 10;
                cp.verbose = verbose;

                metric_type = metric_type_;
                metric_arg = 0;


                // Construct an empty quantizer
                GpuIndexFlatConfig config = ivfConfig_.flatConfig;
                // FIXME: inherit our same device
                config.device = config_.device;
                if (metric_type == faiss::METRIC_L2) {
                    quantizer = new GpuIndexFlatL2(resources_, d, config);
                } else if (metric_type == faiss::METRIC_INNER_PRODUCT) {
                    quantizer = new GpuIndexFlatIP(resources_, d, config);
                } else {
                    // unknown metric type
                    FAISS_THROW_FMT("unsupported metric type %d", (int)metric_type);
                }

                //not trained when initialized
                is_trained = false;

            }


IndexIVFPipe::~IndexIVFPipe() {
    delete quantizer;
    delete direct_map;
    if (!balanced) {
        delete invlists;
    }
    else{
        delete pipe_cluster;
    }
}

void IndexIVFPipe::reset() {
    FAISS_ASSERT(!balanced);
    direct_map->clear();
    invlists->reset();
    ntotal = 0;
}

void IndexIVFPipe::train(idx_t n, const float* x) {

    if (this->is_trained) {
        FAISS_ASSERT(quantizer->is_trained);
        FAISS_ASSERT(quantizer->ntotal == nlist);
        return;
    }

    if (verbose)
        printf("Training quantizer\n");
    
    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    DeviceScope scope(config_.device);

    // FIXME: GPUize more of this
    // First, make sure that the data is resident on the CPU, if it is not on
    // the CPU, as we depend upon parts of the CPU code
    auto hostData = toHost<float, 2>(
            (float*)x,
            resources_->getDefaultStream(config_.device),
            {(int)n, (int)this->d});

    const float* gpu_x = hostData.data();

    if (n == 0) {
        // nothing to do
        return;
    }

    // leverage the CPU-side k-means code, which works for the GPU
    // flat index as well
    Clustering clus(d, nlist, cp);
    quantizer->reset();
    clus.train(n, x, *quantizer);
    quantizer->is_trained = true;
    FAISS_ASSERT(quantizer->ntotal == nlist);
    is_trained = true;

}



void IndexIVFPipe::assignPaged_(int n, const float* x, int* coarse_ids) {
    if (n > 0) {
        size_t totalSize = (size_t)n * this->d * sizeof(float);

        if (totalSize > kAddPageSize || n > kAddVecSize) {
            // How many vectors fit into kAddPageSize?
            size_t maxNumVecsForPageSize =
                    kAddPageSize / ((size_t)this->d * sizeof(float));

            // Always add at least 1 vector, if we have huge vectors
            maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, (size_t)1);

            size_t tileSize = std::min((size_t)n, maxNumVecsForPageSize);
            tileSize = std::min(tileSize, kSearchVecSize);

            for (size_t i = 0; i < (size_t)n; i += tileSize) {
                size_t curNum = std::min(tileSize, n - i);

                assignPage_(
                        curNum,
                        x + i * (size_t)this->d,
                        coarse_ids + i);
            }
        } else {
            assignPage_(n, x, coarse_ids);
        }
    }
}

void IndexIVFPipe::assignPage_(int n, const float* x, int* coarse_ids) {
    // At this point, `x` can be resident on CPU or GPU, and `ids` may be
    // resident on CPU, GPU or may be null.
    //
    // Before continuing, we guarantee that all data will be resident on the
    // GPU.
    auto stream = resources_->getDefaultStreamCurrentDevice();

    auto vecs = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(x),
            stream,
            {n, this->d});

    const float* x_gpu = vecs.data();

    FAISS_ASSERT(n > 0);

    // Not all vectors may be able to be added (some may contain NaNs etc)
    FAISS_ASSERT(vecs.getSize(0) == indices.getSize(0));
    FAISS_ASSERT(vecs.getSize(1) == dim_);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    // Determine which IVF lists we need to append to

    // We don't actually need this
    DeviceTensor<float, 2, true> listDistance(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecs.getSize(0), 1});
    // We use this
    DeviceTensor<int, 2, true> listIds2d(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecs.getSize(0), 1});

    quantizer_->query(
            vecs, 1, metric_, metricArg_, listDistance, listIds2d, false);

    //copy coarse ids back to CPU
    fromDevice<int, 2>(listIds2d, coarse_ids, stream);

    // but keep the ntotal based on the total number of vectors that we
    // attempted to add
    ntotal += n;


}


void IndexIVFPipe::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}

void IndexIVFPipe::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    if (n == 0) {
        // nothing to add
        return;
    }

    DeviceScope scope(config_.device);
    std::unique_ptr<int[]> coarse_idx(new int[n]);
    assignPaged_((int)n, x, coarse_idx.get());

    add_core(n, x, xids, coarse_idx.get());
}

void IndexIVFPipe::add_core(
        idx_t n,
        const float* x,
        const int64_t* xids,
        const int* coarse_idx)

{
    FAISS_ASSERT(!balanced);
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_ASSERT(invlists);
    direct_map->check_can_add(xids);

    int64_t n_add = 0;

    DirectMapAdd dm_adder(*direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = (idx_t)coarse_idx[i];

            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                size_t offset =
                        invlists->add_entry(list_no, id, (const uint8_t*)xi);
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("IndexIVFFlat::add_core: added %" PRId64 " / %" PRId64
               " vectors\n",
               n_add,
               n);
    }
    ntotal += n;
}

void IndexIVFPipe::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    if (!include_listnos) {
        memcpy(codes, x, code_size * n);
    } else {
        size_t coarse_size = coarse_code_size();
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            uint8_t* code = codes + i * (code_size + coarse_size);
            const float* xi = x + i * d;
            if (list_no >= 0) {
                encode_listno(list_no, code);
                memcpy(code + coarse_size, xi, code_size);
            } else {
                memset(code, 0, code_size + coarse_size);
            }
        }
    }
}

void IndexIVFPipe::add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids) {
    FAISS_ASSERT(!balanced);
    size_t coarse_size = coarse_code_size();
    DirectMapAdd dm_adder(*direct_map, n, xids);

    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = codes + (code_size + coarse_size) * i;
        idx_t list_no = decode_listno(code);
        idx_t id = xids ? xids[i] : ntotal + i;
        size_t ofs = invlists->add_entry(list_no, id, code + coarse_size);
        dm_adder.add(i, list_no, ofs);
    }
    ntotal += n;
}

void IndexIVFPipe::train_residual(idx_t /*n*/, const float* /*x*/) {
    if (verbose)
        printf("IndexIVF: no residual training\n");
    // does nothing by default
}

void IndexIVFPipe::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}

void IndexIVFPipe::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    size_t coarse_size = coarse_code_size();
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = bytes + i * (code_size + coarse_size);
        float* xi = x + i * d;
        memcpy(xi, code + coarse_size, code_size);
    }
}


/** It is a sad fact of software that a conceptually simple function like this
 * becomes very complex when you factor in several ways of parallelizing +
 * interrupt/error handling + collecting stats + min/max collection. The
 * codepath that is used 95% of time is the one for parallel_mode = 0 */
void IndexIVFPipe::sample_list(
        idx_t n,
        const float* x,
        float** coarse_dis,
        int** ori_idx,
        idx_t** ori_offset,
        size_t** bcluster_cnt,
        size_t *actual_nprobe,
        int** pipe_cluster_idx,
        size_t* batch_width) {


    FAISS_THROW_IF_NOT_MSG(this->is_trained, "Index not trained");

    // For now, only support <= max int results
    FAISS_THROW_IF_NOT_FMT(
            n <= (Index::idx_t)std::numeric_limits<int>::max(),
            "GPU index only supports up to %d indices",
            std::numeric_limits<int>::max());

    if (n == 0) {
        // nothing to search
        return;
    }
    
    const size_t nprobe = std::min(nlist, this->nprobe);
    *actual_nprobe = nprobe;
    FAISS_THROW_IF_NOT(nprobe > 0);

    *coarse_dis = new float[n * nprobe];
    *ori_idx = new int[n * nprobe];
    *ori_offset = new idx_t[n * nprobe];
    *bcluster_cnt = new size_t[n];

    if (!balanced) {
        balance();
    }

    int nt = std::min(omp_get_max_threads(), int(n));
    std::mutex exception_mutex;
    std::string exception_string;


    DeviceScope scope(config_.device);
    auto stream = resources_->getDefaultStream(config_.device);

    bool usePaged = false;

    if (getDeviceForAddress(x) == -1) {
        // It is possible that the user is querying for a vector set size
        // `x` that won't fit on the GPU.
        // In this case, we will have to handle paging of the data from CPU
        // -> GPU.
        // Currently, we don't handle the case where the output data won't
        // fit on the GPU (e.g., n * k is too large for the GPU memory).
        size_t dataSize = (size_t)n * this->d * sizeof(float);

        if (dataSize >= minPagedSize_) {
            sampleFromCpuPaged_(n, x, nprobe, *coarse_dis, *ori_idx);
            usePaged = true;
        }
    }

    if (!usePaged) {
        sampleNonPaged_(n, x, nprobe, *coarse_dis, *ori_idx);
    }



    std::vector<std::vector<int>> clusters_queries;
    clusters_queries.resize (n);

    #pragma omp parallel for if (nt > 1)
    for (size_t i = 0; i < n; i++) {
        std::vector<int> clusters_query;
        size_t offset = 0;
        for (size_t j = 0; j < nprobe; j++) {
            (*ori_offset)[i * nprobe + j] = offset;
            std::vector<int> &bclusters_probe =
                         pipe_cluster->O2Bmap[(*ori_idx)[i * nprobe + j]];

            clusters_query.insert (clusters_query.end(),
                                bclusters_probe.begin(), bclusters_probe.end());
            offset += bclusters_probe.size();                                
        }
        clusters_queries[i] = clusters_query;
        (*bcluster_cnt)[i] = offset;
    }

    size_t max_bclusters_cnt = 0;
    for (int i = 0; i < n; i++){
        max_bclusters_cnt = std::max (max_bclusters_cnt, (*bcluster_cnt)[i]);
    }
    *batch_width = max_bclusters_cnt;
    *pipe_cluster_idx = new int[n * max_bclusters_cnt];
    auto p = *pipe_cluster_idx;


    #pragma omp parallel for if (nt > 1)
    for (int i = 0; i < n; i++) {
        size_t real_bclusters = (*bcluster_cnt)[i];
        memcpy ((void *)&p[i * max_bclusters_cnt], clusters_queries[i].data(), real_bclusters * sizeof(int));
        std::fill (p + i * max_bclusters_cnt + real_bclusters, p + (i + 1) * max_bclusters_cnt, -1);
    }

    return;
}


void IndexIVFPipe::sampleFromCpuPaged_(
        int n,
        const float* x,
        int nprobe,
        float* coarse_dis,
        int* coarse_ids) const {

    // Just page without overlapping copy with compute
    int batchSize = utils::nextHighestPowerOf2(
            (int)((size_t)kNonPinnedPageSize / (sizeof(float) * this->d)));

    for (int cur = 0; cur < n; cur += batchSize) {
        int num = std::min(batchSize, n - cur);

        float* coarse_dis_slice = coarse_dis + cur;
        int* coarse_ids_slice = coarse_ids + cur;

        sampleNonPaged_(
                num,
                x + (size_t)cur * this->d,
                k,
                coarse_dis_slice,
                coarse_ids_slice);
    }

    return;

    // maybe we can consider overlapping computing with copying.

}

void IndexIVFPipe::sampleNonPaged_(
        int n,
        const float* x,
        int nprobe,
        float* coarse_dis,
        int* coarse_ids) const {
    auto stream = resources_->getDefaultStream(config_.device);

    // Make sure arguments are on the device we desire; use temporary
    // memory allocations to move it if necessary
    auto vecs = toDeviceTemporary<float, 2>(
            resources_.get(),
            config_.device,
            const_cast<float*>(x),
            stream,
            {n, (int)this->d});

    // Device is already set in GpuIndex::search
    FAISS_THROW_IF_NOT(nprobe > 0 && nprobe <= nlist);

    // Data is already resident on the GPU
    Tensor<float, 2, true> queries(const_cast<float*>(vecs.data()), {n, (int)this->d});


    auto stream = resources_->getDefaultStreamCurrentDevice();

    // These are caught at a higher level
    FAISS_ASSERT(nprobe <= GPU_MAX_SELECTION_K);
    FAISS_ASSERT(queries.getSize(1) == dim_);

    // Reserve space for the quantized information
    DeviceTensor<float, 2, true> coarseDistances(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe});
    DeviceTensor<int, 2, true> coarseIndices(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {queries.getSize(0), nprobe});

    // Find the `nprobe` closest lists; we can use int indices both
    // internally and externally
    quantizer_->query(
            queries,
            nprobe,
            metric_,
            metricArg_,
            coarseDistances,
            coarseIndices,
            false);

    fromDevice<int, 2>(coarseIndices, coarse_ids, stream);
    fromDevice<int, 2>(coarseDistances, coarse_dis, stream);

}


void IndexIVFPipe::search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const{
                FAISS_THROW_FMT("search not implemented!%d\n",1);
            }

void IndexIVFPipe::range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result) const{
                FAISS_THROW_FMT("search not implemented!%d\n",2);
            }


void IndexIVFPipe::reconstruct(idx_t key, float* recons) const {
    FAISS_ASSERT(!balanced);
    idx_t lo = direct_map->get(key);
    reconstruct_from_offset(lo_listno(lo), lo_offset(lo), recons);
}


void IndexIVFPipe::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_ASSERT(!balanced);
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        ScopedIds idlist(invlists, list_no);

        for (idx_t offset = 0; offset < list_size; offset++) {
            idx_t id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }

            float* reconstructed = recons + (id - i0) * d;
            reconstruct_from_offset(list_no, offset, reconstructed);
        }
    }
}

/* standalone codec interface */
size_t IndexIVFPipe::sa_code_size() const {
    size_t coarse_size = coarse_code_size();
    return code_size + coarse_size;
}

void IndexIVFPipe::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    std::unique_ptr<int64_t[]> idx(new int64_t[n]);
    quantizer->assign(n, x, idx.get());
    encode_vectors(n, x, idx.get(), bytes, true);
}

size_t IndexIVFPipe::remove_ids(const IDSelector& sel) {
    FAISS_ASSERT(!balanced);
    size_t nremove = direct_map->remove_ids(sel, invlists);
    ntotal -= nremove;
    return nremove;
}


void IndexIVFPipe::update_vectors(int n, const idx_t* new_ids, const float* x) {
    FAISS_ASSERT(!balanced);

    if (direct_map->type == DirectMap::Hashtable) {
        // just remove then add
        IDSelectorArray sel(n, new_ids);
        size_t nremove = remove_ids(sel);
        FAISS_THROW_IF_NOT_MSG(
                nremove == n, "did not find all entries to remove");
        add_with_ids(n, x, new_ids);
        return;
    }

    FAISS_THROW_IF_NOT(direct_map->type == DirectMap::Array);
    // here it is more tricky because we don't want to introduce holes
    // in continuous range of ids

    FAISS_THROW_IF_NOT(is_trained);
    std::vector<idx_t> assign(n);
    quantizer->assign(n, x, assign.data());

    std::vector<uint8_t> flat_codes(n * code_size);
    encode_vectors(n, x, assign.data(), flat_codes.data());

    direct_map->update_codes(
            invlists, n, new_ids, assign.data(), flat_codes.data());
}


void IndexIVFPipe::check_compatible_for_merge(const IndexIVFPipe& other) const {
    // minimal sanity checks
    FAISS_THROW_IF_NOT(other.d == d);
    FAISS_THROW_IF_NOT(other.nlist == nlist);
    FAISS_THROW_IF_NOT(other.code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(other),
            "can only merge indexes of the same type");
    FAISS_THROW_IF_NOT_MSG(
            this->direct_map->no() && other.direct_map->no(),
            "merge direct_map not implemented");
}

void IndexIVFPipe::merge_from(IndexIVFPipe& other, idx_t add_id) {
    check_compatible_for_merge(other);

    invlists->merge_from(other.invlists, add_id);

    ntotal += other.ntotal;
    other.ntotal = 0;
}

void IndexIVFPipe::replace_invlists(InvertedLists* il, bool own) {
    if (own_invlists) {
        delete invlists;
        invlists = nullptr;
    }
    // FAISS_THROW_IF_NOT (ntotal == 0);
    if (il) {
        FAISS_THROW_IF_NOT(il->nlist == nlist);
        FAISS_THROW_IF_NOT(
                il->code_size == code_size ||
                il->code_size == InvertedLists::INVALID_CODE_SIZE);
    }
    invlists = il;
    own_invlists = own;
}

void IndexIVFPipe::copy_subset_to(
        IndexIVFPipe& other,
        int subset_type,
        idx_t a1,
        idx_t a2) const {
    FAISS_THROW_IF_NOT(nlist == other.nlist);
    FAISS_THROW_IF_NOT(code_size == other.code_size);
    FAISS_THROW_IF_NOT(other.direct_map->no());
    FAISS_THROW_IF_NOT_FMT(
            subset_type == 0 || subset_type == 1 || subset_type == 2,
            "subset type %d not implemented",
            subset_type);

    size_t accu_n = 0;
    size_t accu_a1 = 0;
    size_t accu_a2 = 0;

    InvertedLists* oivf = other.invlists;

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t n = invlists->list_size(list_no);
        ScopedIds ids_in(invlists, list_no);

        if (subset_type == 0) {
            for (idx_t i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (a1 <= id && id < a2) {
                    oivf->add_entry(
                            list_no,
                            invlists->get_single_id(list_no, i),
                            ScopedCodes(invlists, list_no, i).get());
                    other.ntotal++;
                }
            }
        } else if (subset_type == 1) {
            for (idx_t i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (id % a1 == a2) {
                    oivf->add_entry(
                            list_no,
                            invlists->get_single_id(list_no, i),
                            ScopedCodes(invlists, list_no, i).get());
                    other.ntotal++;
                }
            }
        } else if (subset_type == 2) {
            // see what is allocated to a1 and to a2
            size_t next_accu_n = accu_n + n;
            size_t next_accu_a1 = next_accu_n * a1 / ntotal;
            size_t i1 = next_accu_a1 - accu_a1;
            size_t next_accu_a2 = next_accu_n * a2 / ntotal;
            size_t i2 = next_accu_a2 - accu_a2;

            for (idx_t i = i1; i < i2; i++) {
                oivf->add_entry(
                        list_no,
                        invlists->get_single_id(list_no, i),
                        ScopedCodes(invlists, list_no, i).get());
            }

            other.ntotal += i2 - i1;
            accu_a1 = next_accu_a1;
            accu_a2 = next_accu_a2;
        }
        accu_n += n;
    }
    FAISS_ASSERT(accu_n == ntotal);
}


//* previous L1Quantizer

size_t IndexIVFPipe::coarse_code_size() const {
    size_t nl = nlist - 1;
    size_t nbyte = 0;
    while (nl > 0) {
        nbyte++;
        nl >>= 8;
    }
    return nbyte;
}

void IndexIVFPipe::encode_listno(Index::idx_t list_no, uint8_t* code) const {
    // little endian
    size_t nl = nlist - 1;
    while (nl > 0) {
        *code++ = list_no & 0xff;
        list_no >>= 8;
        nl >>= 8;
    }
}

Index::idx_t IndexIVFPipe::decode_listno(const uint8_t* code) const {
    size_t nl = nlist - 1;
    int64_t list_no = 0;
    int nbit = 0;
    while (nl > 0) {
        list_no |= int64_t(*code++) << nbit;
        nbit += 8;
        nl >>= 8;
    }
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return list_no;
}


void IndexIVFPipe::balance() {

    //The size of each origional cluster.
    std::vector<int> sizes;

    //The data pointer of each origional cluster.
    std::vector<float*> pointers;

    for (size_t i = 0; i < nlist; i++) {

        //copy the data of origional cluster to a malloced memory.
        const uint8_t* codes_list = invlists->get_codes(i);
        size_t list_size = invlists->list_size(i);
        size_t bytes = list_size * code_size;
        float *codes_list_float = (float*)malloc(bytes);
        memcpy(codes_list_float, codes_list, bytes);
        invlists->release_codes(i);
        
        sizes.push_back((int)list_size);
        pointers.push_back(codes_list_float);
    }

    //construct PipeCluster from the origional clusters' data.
    pipe_cluster = new PipeCluster(nlist, d, sizes, pointers);
    delete invlists;
    balanced = true;

}

void IndexIVFPipe::set_nprobe(size_t nprobe_) {
        nprobe = nprobe_;
    }


}