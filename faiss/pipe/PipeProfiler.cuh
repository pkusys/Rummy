/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <map>

#include <faiss/impl/FaissAssert.h>
#include <faiss/pipe/PipeKernel.cuh>
#include <faiss/gpu/PipeGpuResources.h>
#include <faiss/gpu/utils/PipeTensor.cuh>
#include <faiss/pipe/IndexIVFPipe.h>

namespace faiss {

struct IndexIVFPipe;
namespace gpu {

/** The class is aimed to profile the overhead of a transmission or computation 
 * which will be used for runtime pipelined algorithm.
 */
struct PipeProfiler{
    
    // The construct function and the two resource must be initialized first
    PipeProfiler(IndexIVFPipe *index);

    ~PipeProfiler(){
        delete trans;
        delete coms;
    }


    void train();

    void save(const char* path);

    void load(const char* path);

    double queryTran(int pageCnt);

    double queryCom(int dataCnt);

    int decideSplit(int queryCnt, int dataCnt);

protected:
    
    struct TranProfiler{
        void train();   

        // Modeling: time = data (page or cluster) size * a + b
        double a, b;
        std::map<int, double> tranTimeDict;

        bool istrained = false;

        PipeProfiler* p;

        TranProfiler(PipeProfiler* p_){
            p = p_;
        }
    };

    struct ComProfiler{

        void train();

        std::map<int, double> computeTimeDict;

        bool istrained = false;

        PipeProfiler* p;

        ComProfiler(PipeProfiler* p_){
            p = p_;
        } 


    };

    enum ProfileType {
        // Profiler for transmission
        Transmission,

        // Profiler for computation
        Computation

    };

public:

    // Profile this memory layout
    PipeGpuResources *pgr_;

    // Profile this pipe cluster info
    PipeCluster *pc_;

    // TranProfiler
    TranProfiler *trans;

    ComProfiler *coms;

    IndexIVFPipe *index_;

    // Record the train status
    bool istrained = false;

    bool verbose = false;

    int maxDataCnt;

    int nqMax;

};

} // namespace gpu
} // namespace faiss