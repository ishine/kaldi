// tfnet/tensorflow-transformer.h

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

#ifndef KALDI_TFNET_TENSORFLOW_TRANSFORMER_H_
#define KALDI_TFNET_TENSORFLOW_TRANSFORMER_H_

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <string>
#include <vector>
#include <unordered_map>
#include "util/stl-utils.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"

using tensorflow::Session;
using tensorflow::Tensor;
using tensorflow::Status;

namespace kaldi {
namespace tfnet {

struct TFTransformerWrapperOpts {
	int batch_size;
	int feature_dim;
	int max_feature_length;
	int num_threads;  // 0 means unlimited
	int ntopk;

	TFTransformerWrapperOpts(): batch_size(1), feature_dim(40),
			max_feature_length(960), num_threads(1), ntopk(35) {}

	void Register(OptionsItf *opts) {
		opts->Register("batch-size", &batch_size, "Forward batch size;");
		opts->Register("feature-dim", &feature_dim, "TF model input feature dim;");
		opts->Register("max-feature-length", &max_feature_length, "TF model input feature length;");
		opts->Register("num-threads", &num_threads, "Number of threads for TF computation in cpu; "
	                   "0 means unlimited.");
		opts->Register("ntopk", &ntopk, "TF model topk output;");
	}
};

class TFTransformerWrapper {
public:
	TFTransformerWrapper(const TFTransformerWrapperOpts &opts,
							const std::string &tf_model_path);
	virtual ~TFTransformerWrapper() { Destory();}

	int Forward(std::vector<Matrix<BaseFloat> > &input_features,
			std::vector<Matrix<BaseFloat> >  &loglikes);

private:
	void Destory();

	/// read the TensorFlow model and create the session for computation
	/// num-threads need to be specified in creating the session
	void LoadTFModel(const std::string &tf_model_path, int num_threads);
	void AcquireInitialTensors(int batch_size);


	TFTransformerWrapperOpts opts_;
	Session* session_;  // for TF computation; pointer owned here

	Tensor encoder_feature_;
	Tensor encoder_feature_length_;
	std::vector<std::pair<std::string, Tensor> > inputs_;
	std::vector<Tensor> outputs_;

	std::vector<std::string> 	output_tensor_names_;
	std::vector<float> 			intput_buffer_;
	std::vector<int> 			intput_length_buffer_;
};

}  // namespace tfnet
}  // namespace kaldi

#endif /* KALDI_TFNET_TENSORFLOW_TRANSFORMER_H_ */
