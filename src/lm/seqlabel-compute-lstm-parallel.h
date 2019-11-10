// lm/seqlabel-compute-lstm-parallel.h

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

#ifndef KALDI_LM_SEQLABEL_COMPUTE_LSTM_LM_PARALLEL_H_
#define KALDI_LM_SEQLABEL_COMPUTE_LSTM_LM_PARALLEL_H_

#include "nnet2/am-nnet.h"
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

namespace kaldi {
namespace lm {
typedef nnet0::NnetTrainOptions NnetTrainOptions;
typedef nnet0::NnetDataRandomizerOptions NnetDataRandomizerOptions;
typedef nnet0::NnetParallelOptions NnetParallelOptions;
typedef nnet0::LossOptions LossOptions;

struct SeqLabelLstmUpdateOptions : public nnet0::NnetLstmUpdateOptions {


	SeqLabelLstmUpdateOptions(const NnetTrainOptions *trn_opts, const NnetDataRandomizerOptions *rnd_opts,
                                LossOptions *loss_opts, const NnetParallelOptions *parallel_opts)
    	: NnetLstmUpdateOptions(trn_opts, rnd_opts, NULL, loss_opts, parallel_opts) { }

  	  void Register(OptionsItf *po)
  	  {
  		  NnetLstmUpdateOptions::Register(po);

	      //sequence labeling
  	  }
};


struct SeqLabelStats: nnet0::NnetStats {

	nnet0::Xent xent;

    SeqLabelStats(LossOptions &loss_opts):
            NnetStats(loss_opts), xent(loss_opts){}

    void MergeStats(nnet0::NnetUpdateOptions *opts, int root)
    {
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

        if (opts->objective_function == "xent") {
                        xent.Merge(myid, 0);
        }
        else {
        		KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }

    }

    void Print(nnet0::NnetUpdateOptions *opts, double time_now)
    {
        KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
                  << " with no tgt_mats, " << num_other_error
                  << " with other errors. "
                  << "[" << (opts->crossvalidate?"CROSS-VALIDATION":"TRAINING")
                  << ", " << (opts->randomize?"RANDOMIZED":"NOT-RANDOMIZED")
                  << ", " << time_now/60 << " min, " << total_frames/time_now << " fps"
                  << "]";

        if (opts->objective_function == "xent") {
                KALDI_LOG << xent.Report();
        }
        else {
        	KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }
    }
};


void SeqLabelLstmParallel(const SeqLabelLstmUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string label_rspecifier,
		Nnet *nnet,
		SeqLabelStats *stats);


} // namespace nnet0
} // namespace kaldi

#endif // KALDI_LM_SEQLABEL_COMPUTE_LSTM_LM_PARALLEL_H_
