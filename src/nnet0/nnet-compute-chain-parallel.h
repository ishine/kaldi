// nnet0/nnet-compute-chian-parallel.h

// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET_NNET_COMPUTE_CHAIN_H_
#define KALDI_NNET_NNET_COMPUTE_CHAIN_H_

#include "hmm/transition-model.h"

#include <string>
#include <iomanip>
#include <mpi.h>

#include "nnet-trnopts.h"
#include "nnet0/nnet-randomizer.h"
#include "nnet0/nnet-loss.h"
#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-model-sync.h"

#include "cudamatrix/cu-device.h"

#include "nnet0/nnet-compute-parallel.h"

#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-training.h"
#include "chain/chain-training.h"
#include "chain/chain-den-graph.h"

namespace kaldi {
namespace nnet0 {

struct NnetChainUpdateOptions : public NnetUpdateOptions {

	chain::ChainTrainingOptions chain_config;
	bool apply_deriv_weights;

    int32 num_stream;
    int32 batch_size; // batch truncated size
    int32 targets_delay;
    int32 context_left;
    int32 print_interval;

    NnetChainUpdateOptions(const NnetTrainOptions *trn_opts, const NnetDataRandomizerOptions *rnd_opts, LossOptions *loss_opts, const NnetParallelOptions *parallel_opts)
    	: NnetUpdateOptions(trn_opts, rnd_opts, loss_opts, parallel_opts), apply_deriv_weights(true),
		  num_stream(4), batch_size(0), targets_delay(0), context_left(-1), print_interval(100) { }

  	  void Register(OptionsItf *po)
  	  {
			NnetUpdateOptions::Register(po);
			chain_config.Register(po);

		    po->Register("apply-deriv-weights", &apply_deriv_weights,
		                   "If true, apply the per-frame derivative weights stored with "
		                   "the example");
			po->Register("num-stream", &num_stream, "LSTM BPTT multi-stream training");
			po->Register("batch-size", &batch_size, "LSTM BPTT batch size");
			po->Register("targets-delay", &targets_delay, "LSTM BPTT targets delay");
			po->Register("context-left", &context_left, "using number of frames as left context, -1 denote according the chunk egs.");
		    po->Register("print-interval", &print_interval, "Interval (measured in "
		                   "minibatches) after which we print out objective function "
		                   "during training\n");
  	  }
};

struct ChainInfo : kaldi::nnet3::ObjectiveFunctionInfo {

	void Add(ChainInfo &info) {
		tot_weight_this_phase += info.tot_weight_this_phase;
		tot_objf_this_phase += info.tot_objf_this_phase;
		tot_aux_objf_this_phase += info.tot_aux_objf_this_phase;
		tot_weight += info.tot_weight;
		tot_objf += info.tot_objf;
		tot_aux_objf += info.tot_aux_objf;
	}

	void Merge(int myid, int root) {
		MPI_Barrier(MPI_COMM_WORLD);

		void *addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->tot_weight));
		MPI_Reduce(addr, (void*)(&this->tot_weight), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->tot_objf));
		MPI_Reduce(addr, (void*)(&this->tot_objf), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

		addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->tot_aux_objf));
		MPI_Reduce(addr, (void*)(&this->tot_aux_objf), 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	}
};

struct NnetChainStats: NnetStats {

	unordered_map<std::string, ChainInfo, StringHasher> objf_info_;

    NnetChainStats() { }

    void Add(unordered_map<std::string, ChainInfo, StringHasher> &objf_info) {
        auto iter = objf_info.begin(), end = objf_info.end();
        for (; iter != end; ++iter) {
        	std::string name = iter->first;
        	ChainInfo &info = iter->second;
            if (objf_info_.find(name) != objf_info_.end())
        	    objf_info_[name].Add(info);
            else    
                objf_info_[name] = info;
        }
    }

    void MergeStats(NnetUpdateOptions *opts, int root)
    {
    	int myid = opts->parallel_opts->myid;
    	MPI_Barrier(MPI_COMM_WORLD);

    	void *addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->total_frames));
    	MPI_Reduce(addr, (void*)(&this->total_frames), 1, MPI_UNSIGNED_LONG, MPI_SUM, root, MPI_COMM_WORLD);

    	addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->num_done));
    	MPI_Reduce(addr, (void*)(&this->num_done), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

        auto iter = objf_info_.begin(), end = objf_info_.end();
        for (; iter != end; ++iter) {
        	std::string name = iter->first;
        	ChainInfo &info = iter->second;
        	info.Merge(myid, root);
        }
    }

    void Print(NnetUpdateOptions *opts, double time_now)
    {
        KALDI_LOG << "Done " << num_done << " files, "
                  << "[" << (opts->crossvalidate?"CROSS-VALIDATION":"TRAINING")
                  << ", " << (opts->randomize?"RANDOMIZED":"NOT-RANDOMIZED")
                  << ", " << time_now/60 << " min, " << total_frames/time_now << " fps"
                  << "]";

        auto iter = objf_info_.begin(), end = objf_info_.end();
		for (; iter != end; ++iter) {
			std::string name = iter->first;
			ChainInfo &info = iter->second;
			info.PrintTotalStats(name);
		}
    }
};


void NnetChainUpdateParallel(const NnetChainUpdateOptions *opts,
		fst::StdVectorFst *den_fst,
		std::string	model_filename,
		std::string feature_rspecifier,
		Nnet *nnet,
		NnetChainStats *stats);

} // namespace nnet0
} // namespace kaldi

#endif // KALDI_NNET_NNET_COMPUTE_CHAIN_H_
