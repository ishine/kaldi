// nnet0bin/nnet-train-crfctc-parallel.cc

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

#include "base/kaldi-common.h"
#include "base/timer.h"
#include "util/common-utils.h"
#include "cudamatrix/cu-device.h"
#include "fstext/fstext-lib.h"

#include "nnet0/nnet-trnopts.h"
#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-loss.h"
#include "nnet0/nnet-randomizer.h"
#include "nnet0/nnet-compute-crfctc-parallel.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet0;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
        "Usage:  nnet-train-crfctc-parallel [options] <denominator-fst> <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-crfctc-parallel den.fst scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    SpecAugOptions spec_opts;
    spec_opts.Register(&po);

    NnetParallelOptions parallel_opts;
    parallel_opts.Register(&po);

    LossOptions loss_opts;
    loss_opts.Register(&po);

    CuAllocatorOptions cuallocator_opts;
    cuallocator_opts.Register(&po);

    NnetCrfCtcUpdateOptions opts(&trn_opts, &rnd_opts, &spec_opts,
    		&loss_opts, &parallel_opts, &cuallocator_opts);
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 5-(opts.crossvalidate?1:0)) {
		po.PrintUsage();
		exit(1);
    }

    std::string den_fst_rxfilename = po.GetArg(1),
				feature_rspecifier = po.GetArg(2),
				targets_rspecifier = po.GetArg(3),
				model_filename = po.GetArg(4);
        
    std::string target_model_filename;
    if (!opts.crossvalidate) {
    	target_model_filename = po.GetArg(5);
    }

    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;

// Initialize GPU
#if HAVE_CUDA==1
    if (opts.use_gpu == "yes") {
        CuDevice::Instantiate().AllowMultithreading();
        CuDevice::Instantiate().Initialize();
    }
#endif


    Nnet nnet;
    NnetCrfCtcStats stats(loss_opts);

    fst::StdVectorFst den_fst;
    ReadFstKaldi(den_fst_rxfilename, &den_fst);

    Timer time;
    double time_now = 0;
    KALDI_LOG << "TRAINING STARTED";

    NnetCrfCtcUpdateParallel(&opts,
    		&den_fst,
			model_filename,
			target_model_filename,
			feature_rspecifier,
            targets_rspecifier,
			&nnet,
			&stats);

	/*
    if (!opts.crossvalidate) {
        nnet.Write(target_model_filename, opts.binary);
    }
	*/

    KALDI_LOG << "TRAINING FINISHED; ";
    time_now = time.Elapsed();


    stats.Print(&opts, time_now);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
