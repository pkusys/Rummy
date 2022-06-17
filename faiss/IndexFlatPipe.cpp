#include <IndexFlatPipe.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <cuda_runtime.h>

namespace faiss
{

IndexFlatPipe::IndexFlatPipe(
        size_t d_,
        size_t nlist_,
        MetricType metric)
        : Index(d, metric) {
            code_size = d * sizeof(float);
        }

IndexFlatPipe::~IndexFlatPipe() {
    if (pinned) {
        auto error = cudaFreeHost(pinned_codes);
        FAISS_ASSERT_FMT(
                    error == cudaSuccess,
                    "Failed to free pinned memory (error %d %s)",
                    (int)error,
                    cudaGetErrorString(error));

        pinned = false;
        return;
    }
}

void IndexFlatPipe::add (idx_t n, const float* x) {
    if (pinned) {
        size_t bytes = (ntotal + n) * d * sizeof(float);
        uint8_t *p;
        auto error = cudaMallocHost((void **) &p, bytes);
        FAISS_ASSERT_FMT(
                    error == cudaSuccess,
                    "Failed to malloc pinned memory: %d vectors (error %d %s)",
                    BCluSize[index],
                    (int)error,
                    cudaGetErrorString(error));
        
        memcpy (p, pinned_codes, ntotal * d * sizeof(float));
        sa_encode(n, x, &codes[ntotal * code_size]);
        ntotal += n;

        auto error = cudaFreeHost(pinned_codes);
        FAISS_ASSERT_FMT(
                error == cudaSuccess,
                "Failed to free pinned memory (error %d %s)",
                (int)error,
                cudaGetErrorString(error));

        pinned_codes = p;

    }
    else {
        FAISS_THROW_IF_NOT(is_trained);
        codes.resize((ntotal + n) * code_size);
        sa_encode(n, x, &codes[ntotal * code_size]);
        ntotal += n;
    }
    return;
}

void IndexFlatPipe::reset() {
    if (pinned) {
        auto error = cudaFreeHost(pinned_codes);
        FAISS_ASSERT_FMT(
                error == cudaSuccess,
                "Failed to free pinned memory (error %d %s)",
                (int)error,
                cudaGetErrorString(error));
    }
    else {
        codes.clear();
    }
    ntotal = 0;
    return;
}

size_t IndexFlatPipe::sa_code_size() const {
    return code_size;
}

size_t IndexFlatPipe::remove_ids(const IDSelector& sel) {
    if (pinned) {
        idx_t nremove = 0;
        for (idx_t i = 0; i < ntotal; i++) {
            if (sel.is_member(i)) {
                nremove++;
            }
        }
        
        if (nremove > 0) {
            uint8_t *p;
            size_t bytes = (ntotal - nremove) * d * sizeof(float);
            auto error = cudaMallocHost((void **) &p, bytes);
            FAISS_ASSERT_FMT(
                        error == cudaSuccess,
                        "Failed to malloc pinned memory: %d vectors (error %d %s)",
                        BCluSize[index],
                        (int)error,
                        cudaGetErrorString(error));

            idx_t j = 0;
            ntotal -= cnt;
            for (idx_t i = 0; i < ntotal; i++) {
                if (sel.is_member(i)) {
                    // should be removed
                } else {
                    if (i > j) {
                        memcpy(&p[code_size * j],
                                &pinned_codes[code_size * i],
                                code_size);
                    }
                    j++;
                }
            }

            auto error = cudaFreeHost(pinned_codes);
            FAISS_ASSERT_FMT(
                    error == cudaSuccess,
                    "Failed to free pinned memory (error %d %s)",
                    (int)error,
                    cudaGetErrorString(error));

            pinned_codes = p;   
        }
        return nremove;     

    }
    else {
        idx_t j = 0;
        for (idx_t i = 0; i < ntotal; i++) {
            if (sel.is_member(i)) {
                // should be removed
            } else {
                if (i > j) {
                    memmove(&codes[code_size * j],
                            &codes[code_size * i],
                            code_size);
                }
                j++;
            }
        }
        size_t nremove = ntotal - j;
        if (nremove > 0) {
            ntotal = j;
            codes.resize(ntotal * code_size);
        }
        return nremove;
    }
    
}

void IndexFlatPipe::reconstruct(idx_t key, float* recons) const {
    if (pinned) {
        sa_decode(1, pinned_codes + key * code_size, recons);
    }
    else {
        sa_decode(1, codes.data() +key * code_size, recons);
    }
}

void IndexFlatPipe::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    if (n > 0) {
        memcpy(bytes, x, sizeof(float) * d * n);
    }
}

void IndexFlatPipe::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    if (n > 0) {
        memcpy(x, bytes, sizeof(float) * d * n);
    }
}

void IndexFlatPipe::pin() {
    
    if (pinned) {
        return;
    }

    size_t bytes = ntotal * d * sizeof(float);
    auto error = cudaMallocHost((void **) &pinned_codes, bytes);
    FAISS_ASSERT_FMT(
                    error == cudaSuccess,
                    "Failed to malloc pinned memory: %d vectors (error %d %s)",
                    BCluSize[index],
                    (int)error,
                    cudaGetErrorString(error));

    memcpy (pinned_codes, codes.data(), bytes);
    codes.clear ();

    pinned = true;
    return;
}

void IndexFlatPipe::unpin() {

    if (!pinned) {
        return;
    }

    size_t bytes = ntotal * d * sizeof(float);
    codes.resize (bytes);
    memcpy (codes.data(), pinned_codes, bytes);

    auto error = cudaFreeHost(pinned_codes);
    FAISS_ASSERT_FMT(
                error == cudaSuccess,
                "Failed to free pinned memory (error %d %s)",
                (int)error,
                cudaGetErrorString(error));

    pinned = false;
    return;

}

void IndexFlatPipe::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    FAISS_THROW_IF_NOT (k > 0);
    FAISS_THROW_IF_NOT (!pinned);//

    // we see the distances and labels as heaps

    if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_inner_product(x, get_xb(), d, n, ntotal, &res);
    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_L2sqr(x, get_xb(), d, n, ntotal, &res);
    } else {
        float_maxheap_array_t res = {size_t(n), size_t(k), labels, distances};
        knn_extra_metrics(
                x, get_xb(), d, n, ntotal, metric_type, metric_arg, &res);
    }
}

void IndexFlatPipe::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result) const {
    FAISS_THROW_IF_NOT (!pinned);//       
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            range_search_inner_product(
                    x, get_xb(), d, n, ntotal, radius, result);
            break;
        case METRIC_L2:
            range_search_L2sqr(x, get_xb(), d, n, ntotal, radius, result);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

void IndexFlatPipe::compute_distance_subset(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        const idx_t* labels) const {
    FAISS_THROW_IF_NOT (!pinned);//
    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            fvec_inner_products_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        case METRIC_L2:
            fvec_L2sqr_by_idx(distances, x, get_xb(), labels, d, n, k);
            break;
        default:
            FAISS_THROW_MSG("metric type not supported");
    }
}

}