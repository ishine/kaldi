// nnet0/nnet-train-chain-parallel.cc

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

#include "nnet0/nnet-trnopts.h"
#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "nnet0/nnet-compute-chain-parallel.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet0;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Train nnet0+chain one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "Minibatches are to be created by nnet3-chain-merge-egs in the input pipeline.\n"
    	"This training program is multi-threaded (best to use it with a GPU).\n"
        "Usage:  nnet-train-chain-parallel [options] <denominator-fst-in> <feature-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-chain-parallel den.fst 'ark:nnet3-merge-egs 1.cegs ark:-|' nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    NnetParallelOptions parallel_opts;
    parallel_opts.Register(&po);

    LossOptions loss_opts;
    loss_opts.Register(&po);

    NnetChainUpdateOptions opts(&trn_opts, &rnd_opts, &loss_opts, &parallel_opts);
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 4-(opts.crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string
	  den_fst_rxfilename = po.GetArg(1),
	  feature_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);
        
    std::string target_model_filename;
    if (!opts.crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().Initialize();
    //CuDevice::Instantiate().DisableCaching();
#endif


    Nnet nnet;
    NnetChainStats stats(loss_opts);

    fst::StdVectorFst den_fst;
    ReadFstKaldi(den_fst_rxfilename, &den_fst);

    Timer time;
    double time_now = 0;
    KALDI_LOG << "TRAINING STARTED";

    NnetChainUpdateParallel(&opts,
    		&den_fst,
    		model_filename,
    		feature_rspecifier,
    		&nnet,
    		&stats);

    if (!opts.crossvalidate) {
      nnet.Write(target_model_filename, opts.binary);
    }

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
