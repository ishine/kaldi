// lm/rnnt-compute-lstm-parallel.h

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

#ifndef KALDI_LM_RNNT_COMPUTE_LSTM_LM_PARALLEL_H_
#define KALDI_LM_RNNT_COMPUTE_LSTM_LM_PARALLEL_H_

#include "hmm/transition-model.h"

#include <string>
#include <iomanip>
#include <mpi.h>

#include "nnet0/nnet-trnopts.h"
#include "nnet0/nnet-randomizer.h"
#include "nnet0/nnet-loss.h"
#include "nnet0/nnet-nnet.h"

#include "cudamatrix/cu-device.h"

#include "nnet0/nnet-compute-lstm-asgd.h"
#include "lm/lm-model-sync.h"
#include "lm/lm-compute-lstm-parallel.h"

namespace kaldi {
namespace lm {
typedef nnet0::NnetTrainOptions NnetTrainOptions;
typedef nnet0::NnetDataRandomizerOptions NnetDataRandomizerOptions;
typedef nnet0::NnetParallelOptions NnetParallelOptions;
typedef nnet0::NnetUpdateOptions NnetUpdateOptions;
typedef nnet0::NnetStats NnetStats;
typedef nnet0::LossOptions LossOptions;
typedef nnet0::CtcItf CtcItf;

struct RNNTLstmUpdateOptions : public NnetUpdateOptions {

    int32 num_stream;
    int32 max_frames;
    int32 batch_size;
    int32 am_truncated;
    int32 lm_truncated;
    int32 blank_label;
    int32 sos_id;
    bool  freeze_lm;

	RNNTLstmUpdateOptions(const NnetTrainOptions *trn_opts, const NnetDataRandomizerOptions *rnd_opts,
                        LossOptions *loss_opts, const NnetParallelOptions *parallel_opts, const CuAllocatorOptions *cuallocator_opts = NULL)
    	: NnetUpdateOptions(trn_opts, rnd_opts, loss_opts, parallel_opts, cuallocator_opts),
		  num_stream(4), max_frames(25000), batch_size(0), 
          am_truncated(0), lm_truncated(0), blank_label(0), sos_id(0), freeze_lm(false) { 
          objective_function = "rnnt";
    }

  	void Register(OptionsItf *po) {
	  	NnetUpdateOptions::Register(po);
	  	po->Register("num-stream", &num_stream, "---CTC--- BPTT multi-stream training");
	  	po->Register("max-frames", &max_frames, "Max number of frames to be processed");
	  	po->Register("batch-size", &batch_size, "lstm bptt truncated size");
	  	po->Register("am-truncated", &am_truncated, "encoder lstm network bptt truncated size");
	  	po->Register("lm-truncated", &lm_truncated, "predict lstm network bptt truncated size");
	  	po->Register("blank-label", &blank_label, "CTC output bank label id");
	  	po->Register("sos-id", &sos_id, "start of utterance(<s>) id in predict model");
	  	po->Register("freeze-lm", &freeze_lm, "freeze update training predict model");
  	}
};

struct RNNTStats: NnetStats {

    CtcItf rnnt;

    RNNTStats(LossOptions &loss_opts): NnetStats(loss_opts) { }

    void MergeStats(NnetUpdateOptions *opts, int root) {
        int myid = opts->parallel_opts->myid;
        MPI_Barrier(MPI_COMM_WORLD);

        void *addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->total_frames));
        MPI_Reduce(addr, (void*)(&this->total_frames), 1, MPI_UNSIGNED_LONG, MPI_SUM, root, MPI_COMM_WORLD);

        addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->num_done));
        MPI_Reduce(addr, (void*)(&this->num_done), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

        addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->num_no_tgt_mat));
        MPI_Reduce(addr, (void*)(&this->num_no_tgt_mat), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

        addr = (void *) (myid==root ? MPI_IN_PLACE : (void*)(&this->num_other_error));
        MPI_Reduce(addr, (void*)(&this->num_other_error), 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

        if (opts->objective_function == "rnnt") {
        	rnnt.Merge(myid, 0);
        } else {
        	KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }

    }

    void Print(NnetUpdateOptions *opts, double time_now) {
        KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
                  << " with no tgt_mats, " << num_other_error
                  << " with other errors. "
                  << "[" << (opts->crossvalidate?"CROSS-VALIDATION":"TRAINING")
                  << ", " << (opts->randomize?"RANDOMIZED":"NOT-RANDOMIZED")
                  << ", " << time_now/60 << " min, " << total_frames/time_now << " fps"
                  << "]";

        if (opts->objective_function == "rnnt") {
        	KALDI_LOG << rnnt.Report();
        } else {
        	KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }
    }
};

void RNNTLstmUpdateParallel(const RNNTLstmUpdateOptions *opts,
		std::string	model_filename,
        std::string target_model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		RNNTStats *stats);


} // namespace lm
} // namespace kaldi

#endif // KALDI_LM_RNNT_COMPUTE_LSTM_LM_PARALLEL_H_
