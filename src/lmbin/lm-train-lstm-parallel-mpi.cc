// lm/lm-train-lstm-parallel-mpi.cc

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
#include "nnet0/nnet-loss.h"
#include "nnet0/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"
#include "lm/lm-compute-lstm-parallel.h"

#include <iomanip>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet0;
  using namespace kaldi::lm;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
        "Usage:  nnet-train-lstm-lm-parallel-mpi [options] <feature-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " lm-train-lstm-parallel-mpi scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    NnetParallelOptions parallel_opts;

    //multi-machine
    MPI_Init_thread(&argc,&argv, MPI_THREAD_MULTIPLE, &parallel_opts.thread_level);
    MPI_Comm_size(MPI_COMM_WORLD, &parallel_opts.num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &parallel_opts.myid);

    parallel_opts.Register(&po);

    LossOptions loss_opts;
    loss_opts.Register(&po);

    CuAllocatorOptions cuallocator_opts;
    cuallocator_opts.cache_memory = false;
    cuallocator_opts.Register(&po);

    LstmlmUpdateOptions opts(&trn_opts, &rnd_opts, &loss_opts, &parallel_opts, &cuallocator_opts);
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3-(opts.crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      model_filename = po.GetArg(2);
        
    std::string target_model_filename;
    if (!opts.crossvalidate) {
      target_model_filename = po.GetArg(3);
    }

    //mpi
    	LmParallelUtil util;
	    std::string scpfile;
	    feature_rspecifier = util.AddSuffix(feature_rspecifier, parallel_opts.myid);
	    scpfile = util.GetFilename(feature_rspecifier);
	    if (parallel_opts.myid == 0)
	    {
	    	parallel_opts.num_merge = util.NumofCEMerge(scpfile, parallel_opts.merge_size) / parallel_opts.num_threads;
	    }

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast((void*)(&parallel_opts.merge_size), 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast((void*)(&parallel_opts.num_merge), 1, MPI_INT, 0, MPI_COMM_WORLD);

	    std::string logfn = util.FAddSuffix(parallel_opts.log_file, parallel_opts.myid);

	    //stderr redirect to logfile
	    int    fd;
	    fpos_t pos;

	    fflush(stderr);
	    fgetpos(stderr, &pos);
	    fd = dup(fileno(stderr));
	    FILE * logfile = freopen(logfn.c_str(), "w", stderr);
	    if (NULL == logfile)
	    {
	    	KALDI_ERR << "log path must be specified [--log-file]";
	    }
	    setvbuf(logfile, NULL, _IONBF, 0);

    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    if (opts.use_gpu == "yes") {
        CuDevice::Instantiate().AllowMultithreading();
        CuDevice::Instantiate().Initialize();
    }
#endif


    Nnet nnet;
    LmStats stats(loss_opts);

    Timer time;
    double time_now = 0;
    KALDI_LOG << "TRAINING STARTED";


    LstmlmUpdateParallel(&opts,
					model_filename,
                    target_model_filename,
					feature_rspecifier,
								&nnet,
								&stats);


    if (parallel_opts.myid == 0 && !opts.crossvalidate) {
       //nnet.Write(target_model_filename, opts.binary);
       if (opts.zt_mean_filename != "")
       {
           Vector<BaseFloat> class_zt;
           stats.cbxent.GetConstZtMean(class_zt);
           Output out;
           out.Open(opts.zt_mean_filename, false, false);
           class_zt.Write(out.Stream(), false);
           out.Close();
       }
    }

    KALDI_LOG << "TRAINING FINISHED; ";
    time_now = time.Elapsed();

    stats.Print(&opts, time_now);

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    //restore stderr
    fflush(stderr);
    dup2(fd, fileno(stderr));
    close(fd);
    clearerr(stderr);
    fsetpos(stderr, &pos);

    //merge global statistic data
    stats.MergeStats(&opts, 0);

    if (parallel_opts.myid == 0)
    {
        time_now = time.Elapsed();
        stats.Print(&opts, time_now);
    }

    MPI_Finalize();

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
