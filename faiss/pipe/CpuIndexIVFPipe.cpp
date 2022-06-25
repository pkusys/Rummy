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
#include <faiss/pipe/CpuIndexIVFPipe.h>
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
            MetricType metric_type_)
            : d(d_), nlist(nlist_), metric_type(metric_type_), Index(d_) {
                verbose = false;
                balanced = false;
                code_size = d * sizeof(float);
                direct_map = new DirectMap();
                quantizer = new IndexFlatPipe (d_, nlist_, metric_type);
                invlists = new ArrayInvertedLists(nlist, code_size);
                pipe_cluster = nullptr;
                is_trained = false;
                nprobe = 1;
                if (metric_type == METRIC_INNER_PRODUCT) {
                    cp.spherical = true;
                }
                
            }

IndexIVFPipe::~IndexIVFPipe() {
    delete quantizer;
    delete direct_map;
    if(!balanced) {
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
        //FAISS_ASSERT(index_);already has data vectors
        return;
    }

    //FAISS_ASSERT(!index_);no data vectors yet

    if (verbose)
        printf("Training quantizer\n");
    
    Clustering clus(d, nlist, cp);
    quantizer->reset();
    clus.train(n, x, *quantizer);
    quantizer->is_trained = true;
    quantizer->pin();
    is_trained = true;
}

void IndexIVFPipe::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}

void IndexIVFPipe::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    quantizer->assign(n, x, coarse_idx.get());
    add_core(n, x, xids, coarse_idx.get());
}


void IndexIVFPipe::add_core(
        idx_t n,
        const float* x,
        const int64_t* xids,
        const int64_t* coarse_idx)

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
            idx_t list_no = coarse_idx[i];

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
        idx_t** ori_idx,
        idx_t** ori_offset,
        size_t** bcluster_cnt,
        size_t *actual_nprobe,
        int** pipe_cluster_idx,
        size_t* batch_width) {
    
    const size_t nprobe = std::min(nlist, this->nprobe);
    *actual_nprobe = nprobe;
    FAISS_THROW_IF_NOT(nprobe > 0);

    *coarse_dis = new float[n * nprobe];
    *ori_idx = new idx_t[n * nprobe];
    *ori_offset = new idx_t[n * nprobe];
    *bcluster_cnt = new size_t[n];

    if (!balanced) {
        balance();
    }

    int nt = std::min(omp_get_max_threads(), int(n));
    std::mutex exception_mutex;
    std::string exception_string;

    #pragma omp parallel for if (nt > 1)
    for (idx_t slice = 0; slice < nt; slice++) {
        idx_t i0 = n * slice / nt;
        idx_t i1 = n * (slice + 1) / nt;
        if (i1 > i0) {
            quantizer->search (i1 - i0, x + i0 * d, nprobe, (*coarse_dis) + i0 * nprobe, (*ori_idx) + i0 * nprobe);
        }
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