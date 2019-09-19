// lm/slu-compute-lstm-parallel.h

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

#ifndef KALDI_LM_LM_COMPUTE_LSTM_SLU_PARALLEL_H_
#define KALDI_LM_LM_COMPUTE_LSTM_SLU_PARALLEL_H_

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

struct SluLstmUpdateOptions : public nnet0::NnetLstmUpdateOptions {

	// language model
	std::string class_boundary;
	int32 num_class;

	// slot label rspecifier
	std::string slot_rspecifier;
	// intent label rspecifier
	std::string intent_rspecifier;

	int32 slot_delay;
	int32 intent_delay;

    BaseFloat lm_escale;
    BaseFloat slot_escale;
    BaseFloat intent_escale;

	SluLstmUpdateOptions(const NnetTrainOptions *trn_opts, const NnetDataRandomizerOptions *rnd_opts, 
                            LossOptions *loss_opts, const NnetParallelOptions *parallel_opts)
    	: NnetLstmUpdateOptions(trn_opts, rnd_opts, loss_opts, parallel_opts), class_boundary(""), num_class(0),
		  slot_rspecifier(""), intent_rspecifier(""), slot_delay(0), intent_delay(0), lm_escale(1.0), slot_escale(1.0), intent_escale(1.0) { }

  	  void Register(OptionsItf *po)
  	  {
  		  NnetLstmUpdateOptions::Register(po);

	      //lm
  		  po->Register("class-boundary", &class_boundary, "The fist index of each class(and final class class) in class based language model");
  		  po->Register("num-class", &num_class, "The number of class that the language model output");

  		  // slot label
  		  po->Register("slot-label", &slot_rspecifier, "slot classfication label in slu multi-task training");

  		  // intent label
  		  po->Register("intent-label", &intent_rspecifier, "intent classfication label in slu multi-task training");

  		  // slot target delay
  		  po->Register("slot-delay", &slot_delay, "slot target delay in slu multi-task training");
  		  // intent target delay
  		  po->Register("intent-delay", &intent_delay, "intent target delay in slu multi-task training");

  		  po->Register("lm-escale", &lm_escale, "language model error scale in slu parallel component");
  		  po->Register("slot-escale", &slot_escale, "slot error scale in slu parallel component");
  		  po->Register("intent-escale", &intent_escale, "intent error scale in slu parallel component");
  	  }
};


struct SluStats: nnet0::NnetStats {

	nnet0::CBXent cbxent;
	nnet0::Xent xent, slot_xent, intent_xent;

    SluStats(LossOptions &loss_opts):
            NnetStats(loss_opts), xent(loss_opts), slot_xent(loss_opts), intent_xent(loss_opts){}

    void MergeStats(nnet0::NnetUpdateOptions *opts, int root)
    {
        SluLstmUpdateOptions *slu_opts = static_cast<SluLstmUpdateOptions*>(opts);

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
        else if (opts->objective_function == "cbxent") {
                        cbxent.Merge(myid, 0);
        }
        else {
        		KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }

        if (slu_opts->slot_rspecifier != "")
            slot_xent.Merge(myid, 0);
        if (slu_opts->intent_rspecifier != "")
            intent_xent.Merge(myid, 0);

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
                KALDI_LOG << "(LM task report) " << xent.Report();
        }
        else if (opts->objective_function == "cbxent") {
                KALDI_LOG << "(LM task report) " << cbxent.Report();
        }
        else {
        	KALDI_ERR << "Unknown objective function code : " << opts->objective_function;
        }

        SluLstmUpdateOptions *slu_opts = static_cast<SluLstmUpdateOptions*>(opts);
        if (slu_opts->slot_rspecifier != "")
            KALDI_LOG << "(Slot task report) " << slot_xent.Report();
        if (slu_opts->intent_rspecifier != "")
            KALDI_LOG << "(Intent task report) " << intent_xent.Report();
    }
};


void SluLstmUpdateParallel(const SluLstmUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		Nnet *nnet,
		SluStats *stats);


} // namespace nnet0
} // namespace kaldi

#endif // KALDI_LM_LM_COMPUTE_LSTM_SLU_PARALLEL_H_
