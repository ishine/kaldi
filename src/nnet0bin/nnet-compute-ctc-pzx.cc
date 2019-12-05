// nnetbin/nnet-compute-ctc-pzx.cc

// Copyright 2015-2016       Shanghai Jiao Tong University (Author: Wei Deng)

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
#include "util/common-utils.h"
#include "nnet0/nnet-loss.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi;
  using namespace kaldi::nnet0;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute sex vector\n"
        "e.g.:\n"
        " nnet-compute-ctc-pzx ark:am_posterior.ark ark:hyps_results.ark ark,t:pzx.txt \n";

    ParseOptions po(usage);

    int blank_label = 0;
    po.Register("blank-label", &blank_label, "CTC output bank label id");
    std::string use_gpu="yes";

    po.Read(argc, argv);

    CuAllocatorOptions cuallocator_opts;
    cuallocator_opts.cache_memory = true;
    cuallocator_opts.Register(&po);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string am_posterior_rspecifier = po.GetArg(1),
    			hyps_results_rspecifier = po.GetArg(2),
				wspecifier = po.GetArg(3);

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().SetCuAllocatorOptions(cuallocator_opts);
#endif


    SequentialBaseFloatMatrixReader am_posterior_reader(am_posterior_rspecifier);
    RandomAccessInt32VectorReader hyps_reader(hyps_results_rspecifier);
    BaseFloatVectorWriter pzx_writer(wspecifier);

    std::vector<std::vector<int> > labels_utt(1);
    std::vector<int> num_utt_frame_in(1);

    Matrix<BaseFloat> posterior;
    CuMatrix<BaseFloat> cu_posterior, nnet_diff;

	Vector<BaseFloat> pzx(1);
	float num_done = 0, pzx_sum = 0;

	WarpCtc *ctc = new WarpCtc(0);
	for (; !am_posterior_reader.Done(); am_posterior_reader.Next()) {
		const Matrix<BaseFloat> &posterior = am_posterior_reader.Value();
		std::string utt = am_posterior_reader.Key();
		const std::vector<int32> &hpys = hyps_reader.Value(utt);

		num_utt_frame_in[0] = posterior.NumRows();
		cu_posterior = posterior;
		labels_utt[0] = hpys;
		ctc->EvalParallel(num_utt_frame_in, cu_posterior, labels_utt, &nnet_diff, &pzx);
		pzx_writer.Write(utt, pzx);

		num_done++;
		pzx_sum += pzx(0);
	}

	KALDI_LOG << "Computed " << num_done << " utterance, "
			<< "Obj(log[Pzx]) = " << pzx_sum/num_done;
	return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


