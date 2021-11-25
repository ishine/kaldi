// tfnetbin/nnet-tfnet-forward.cc

// Copyright 2021-2022 	Wei Deng

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

#include <limits>

#include "tfnet/tensorflow-transformer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::tfnet;

  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Perform forward pass through tensorflow Neural Network.\n"
        "\n"
        "Usage:  nnet-tfnet-forward [options] <tfmodel-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " nnnet-tfnet-forward tfnnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    TFTransformerWrapperOpts tfnet_opts;
    tfnet_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
    		feature_rspecifier = po.GetArg(2),
			feature_wspecifier = po.GetArg(3);

    TFTransformerWrapper tfnet(tfnet_opts, model_filename);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    std::vector<Matrix<BaseFloat> > input_features;
    std::vector<Matrix<BaseFloat> > loglikes;
    std::vector<std::string> keys;
    bool isfinished = false;
    int batch_size = tfnet_opts.batch_size;
    int max_feature_length = tfnet_opts.max_feature_length;
    int ret;

    Timer time;
    double time_now = 0;
    int32 num_done = 0, batch = 0;
    kaldi::int64 tot_t = 0;

    // iterate over all feature files
    while (true) {
    	if (feature_reader.Done())
    		isfinished = true;

    	if (!isfinished) {
    		// read
    		keys.push_back(feature_reader.Key());
    		input_features.push_back(feature_reader.Value());
    		feature_reader.Next();
    		batch++;
    	}

    	if (batch%batch_size != 0 && !isfinished)
    		continue;

    	if (input_features.size() == 0)
    		break;

    	ret = tfnet.Forward(input_features, loglikes);

    	for (int i = 0; i < loglikes.size(); i++) {
    		feature_writer.Write(keys[i], loglikes[i]);
    		tot_t += input_features[i].NumRows();
    		num_done++;
    	}

    	input_features.clear();
    	batch = 0;

    	// progress log
		if (num_done % 100 == 0) {
			time_now = time.Elapsed();
			KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
						  << time_now/60 << " min; processed " << tot_t/time_now
						  << " frames per second.";
		}
		if (isfinished) break;
    }

    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
