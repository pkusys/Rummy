/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <typeinfo>
#include <faiss/MetricType.h>
#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/pipe/IndexFlatPipe.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/DirectMap.h>

namespace faiss {

/* Piped IndexIVF, with CPU quantizer */
struct CpuIndexIVFPipe: Index {

    int d;
    size_t nlist;
    MetricType metric_type;
    using idx_t = int64_t;
    bool verbose;

    IndexFlatPipe* quantizer; ///< quantizer that maps vectors to inverted lists
    InvertedLists* invlists;

    ClusteringParameters cp; ///< to override default clustering params


    bool own_invlists;

    size_t code_size; ///< code size per vector in bytes

    size_t nprobe;    ///< number of probes at query time
    size_t max_codes; ///< max nb of codes to visit to do a query

    /** Parallel mode determines how queries are parallelized with OpenMP
     *
     * 0 (default): split over queries
     * 1: parallelize over inverted lists
     * 2: parallelize over both
     * 3: split over queries with a finer granularity
     *
     * PARALLEL_MODE_NO_HEAP_INIT: binary or with the previous to
     * prevent the heap to be initialized and finalized
     */
    int parallel_mode;
    const int PARALLEL_MODE_NO_HEAP_INIT = 1024;

    /** optional map that maps back ids to invlist entries. This
     *  enables reconstruct() */
    DirectMap *direct_map;

    bool balanced;

    PipeCluster *pipe_cluster;

    CpuIndexIVFPipe(
            size_t d_,
            size_t nlist_,
            MetricType = METRIC_L2);

    ~CpuIndexIVFPipe();


    //* IndexIVF functions:

    void reset() override;

    /// Trains the quantizer and calls train_residual to train sub-quantizers
    void train(idx_t n, const float* x) override;

    /// Calls add_with_ids with NULL ids
    void add(idx_t n, const float* x) override;

    //reimplmented by IndexIVFFlat
    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx);

    /** Add vectors that are computed with the standalone codec
     *
     * @param codes  codes to add size n * sa_code_size()
     * @param xids   corresponding ids, size n
     */
    void add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids);

    /// default implementation that calls encode_vectors
    void add_with_ids(idx_t n, const float* x, const idx_t* xids) override;    

    void balance();

    //Level1_Quantizer
    size_t coarse_code_size() const;

    /** check that the two indexes are compatible (ie, they are
     * trained in the same way and have the same
     * parameters). Otherwise throw. */
    void check_compatible_for_merge(const CpuIndexIVFPipe& other) const;


    //* Level1Quantizier functions
    Index::idx_t decode_listno(const uint8_t* code) const;
    
    void encode_listno(Index::idx_t list_no, uint8_t* code) const;

    //reimplmented by IndexIVFFlat
    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const;

    //reimplmented by IndexIVFFlat
    //get_InvertedListScanner()  removed
     
    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result) const override;

    /** reconstruct a vector. Works only if maintain_direct_map is set to 1 or 2
     */
    void reconstruct(idx_t key, float* recons) const override;

    /** Reconstruct a subset of the indexed vectors.
     *
     * Overrides default implementation to bypass reconstruct() which requires
     * direct_map to be maintained.
     *
     * @param i0     first vector to reconstruct
     * @param ni     nb of vectors to reconstruct
     * @param recons output array of reconstructed vectors, size ni * d
     */
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    //search_and_reconstruct removed

    /** Reconstruct a vector given the location in terms of (inv list index +
     * inv list offset) instead of the id.
     *
     * Useful for reconstructing when the direct_map is not maintained and
     * the inv list offset is computed by search_preassigned() with
     * `store_pairs` set.
     *///reimplmented by IndexIVFFlat  
    virtual void reconstruct_from_offset(
            int64_t list_no,
            int64_t offset,
            float* recons) const;

    /// Dataset manipulation functions

    size_t remove_ids(const IDSelector& sel) override;

    /// replace the inverted lists, old one is deallocated if own_invlists
    void replace_invlists(InvertedLists* il, bool own = false);

    /** sample the balanced clusters to be searched on
     * @param n number of queries
     * @param x vectors to query
     * @param coarse_dis[n][nprobe]  the matrix of distance between origional clusters and queries
     * @param ori_idx[n][nprobe] the matrix of id of origional clusters
     * @param ori_offset[n][nprobe] the matrix of offset of origional cluster[i][j] in pipe_cluster_idx[i] 
     * @param bcluster_cnt[n] the number of balanced clusters to be searched for query i.
     * @param actual_nprobe actual number of probes
     * @param pipe_cluster_idx[n][batch_width] the matrix of id of balanced clusters
     * @param batch_width the maximum numbers of balanced vectors for one query in this case
     */
    void sample_list(
        idx_t n,
        const float* x,
        float** coarse_dis,
        idx_t** ori_idx,
        idx_t** ori_offset,
        size_t** bcluster_cnt,
        size_t *actual_nprobe,
        int** pipe_cluster_idx,
        size_t* batch_width);

    
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /* The standalone codec interface (except sa_decode that is specific) */
    size_t sa_code_size() const override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;


    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    //search_preassigned removed

    /// Sub-classes that encode the residuals can train their encoders here
    /// does nothing by default
    void train_residual(idx_t n, const float* x);

    /** Update a subset of vectors.
     *
     * The index must have a direct_map
     *
     * @param nv     nb of vectors to update
     * @param idx    vector indices to update, size nv
     * @param v      vectors of new values, size nv*d
     */
    virtual void update_vectors(int nv, const idx_t* idx, const float* v);


    /** moves the entries from another dataset to self. On output,
     * other is empty. add_id is added to all moved ids (for
     * sequential ids, this would be this->ntotal */
    virtual void merge_from(CpuIndexIVFPipe& other, idx_t add_id);

    /** copy a subset of the entries index to the other index
     *
     * if subset_type == 0: copies ids in [a1, a2)
     * if subset_type == 1: copies ids if id % a1 == a2
     * if subset_type == 2: copies inverted lists such that a1
     *                      elements are left before and a2 elements are after
     */
    virtual void copy_subset_to(
            CpuIndexIVFPipe& other,
            int subset_type,
            idx_t a1,
            idx_t a2) const;


    size_t get_list_size(size_t list_no) const {
        return invlists->list_size(list_no);
    }

    /** intialize a direct map
     *
     * @param new_maintain_direct_map    if true, create a direct map,
     *                                   else clear it
     */
    void make_direct_map(bool new_maintain_direct_map = true);

    void set_direct_map_type(DirectMap::Type type);

    void set_nprobe(size_t nprobe_);

};


struct CpuIndexIVFStats {
    size_t nq;                // nb of queries run
    size_t nlist;             // nb of inverted lists scanned
    size_t ndis;              // nb of distances computed
    size_t nheap_updates;     // nb of times the heap was updated
    double quantization_time; // time spent quantizing vectors (in ms)
    double search_time;       // time spent searching lists (in ms)

    CpuIndexIVFStats() {
        reset();
    }
    void reset();
    void add(const CpuIndexIVFStats& other);
};

}