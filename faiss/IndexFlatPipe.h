/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <vector>
#include <faiss/pipe/PipeCluster.h>
#include <faiss/impl/AuxIndexStructures.h>

namespace faiss
{

struct IndexFlatPipe: Index
{
    IndexFlatPipe(
        size_t d_,
        size_t nlist_,
        MetricType metric);

    ~IndexFlatPipe();

    void add(idx_t n, const float* x) override;

    void reset() override;


    size_t sa_code_size() const override;

    /** remove some ids. NB that Because of the structure of the
     * indexing structure, the semantics of this operation are
     * different from the usual ones: the new ids are shifted */
    size_t remove_ids(const IDSelector& sel) override;



    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result) const override;

    void reconstruct(idx_t key, float* recons) const override;

    /** compute distance with a subset of vectors
     *
     * @param x       query vectors, size n * d
     * @param labels  indices of the vectors that should be compared
     *                for each query vector, size n * k
     * @param distances
     *                corresponding output distances, size n * k
     */
    void compute_distance_subset(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            const idx_t* labels) const;

    // get pointer to the floating point data
    float* get_xb() {
        if (pinned) {
            return (float*)pinned_codes;
        }
        else {
            return (float*)codes.data();
        }
    }
    const float* get_xb() const {
        if (pinned) {
            return (float*)pinned_codes;
        }
        else {
            return (float*)codes.data();
        }
    }


    DistanceComputer* get_distance_computer() const override;

    /* The stanadlone codec interface (just memcopies in this case) */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void pin();

    void unpin();
    


    size_t code_size;
    std::vector<uint8_t> codes;
    uint8_t* pinned_codes;
    bool pinned;


};



}
