// tfnet/tensorflow-transformer.cc

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



#include "tfnet/tensorflow-transformer.h"

namespace kaldi {
namespace tfnet {

void TFTransformerWrapper::LoadTFModel(const std::string &tf_model_path, int num_threads) {
	using namespace tensorflow;
	tensorflow::SessionOptions sess_options;
	tensorflow::ConfigProto &config = sess_options.config;
	config.set_inter_op_parallelism_threads(num_threads);
	config.set_intra_op_parallelism_threads(num_threads);
	//config.add_session_inter_op_thread_pool()->set_num_threads(1);
	config.set_use_per_session_threads(false);

    // gpu option
    config.mutable_gpu_options()->set_allow_growth(true);

	KALDI_LOG << "starting initalize session";
	Status status = tensorflow::NewSession(sess_options, &session_);

	GraphDef graphdef;
    //tensorflow::graph::SetDefaultDevice("/gpu:0", &graph_def);
    Status status_load = ReadBinaryProto(Env::Default(), tf_model_path, &graphdef); //从pb文件中读取图模型;
    if (!status_load.ok()) {
        KALDI_ERR << "ERROR: Loading model failed..." << tf_model_path;
        KALDI_ERR << status_load.ToString();
        return ;
    }
    KALDI_LOG << "loading acoustic model: " << tf_model_path << " successfully.";

    Status status_create = session_->Create(graphdef); //将模型导入会话Session中;
    if (!status_create.ok()) {
        KALDI_ERR << "ERROR: Creating graph in session failed..." << status_create.ToString();
        return ;
    }
    KALDI_LOG << "session create successfully.";
}

void TFTransformerWrapper::AcquireInitialTensors(int batch_size) {
	if (encoder_feature_.shape().dim_size(0) == batch_size)
		return;

	Tensor encoder_feature(tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_size, opts_.max_feature_length, opts_.feature_dim}));
	Tensor encoder_feature_length(tensorflow::DT_INT32, tensorflow::TensorShape({batch_size}));
	encoder_feature_ = encoder_feature;
	encoder_feature_length_ = encoder_feature_length;
}

TFTransformerWrapper::TFTransformerWrapper(const TFTransformerWrapperOpts &opts,
											const std::string &tf_model_path):
		opts_(opts) {
	LoadTFModel(tf_model_path, opts.num_threads);

	//定义模型输出
	output_tensor_names_ = {
		"bpe_topk",
		"bpe_output_length"
	};

	intput_buffer_.reserve(opts.batch_size * opts.feature_dim * opts.max_feature_length);
	intput_length_buffer_.reserve(opts.batch_size);
}

int TFTransformerWrapper::Forward(std::vector<Matrix<BaseFloat> > &input_features,
		std::vector<Matrix<BaseFloat> > &loglikes) {

	int batch_size = input_features.size(), size, out_dim;
    int nframes = 0, n;
    intput_buffer_.clear();
    intput_length_buffer_.clear();
	intput_buffer_.resize(batch_size * opts_.feature_dim * opts_.max_feature_length, 0);
	intput_length_buffer_.resize(batch_size);
	auto begin = intput_buffer_.begin();
	for (int i = 0; i < batch_size; i++) {
        nframes = input_features[i].NumRows();
		n = nframes > opts_.max_feature_length ? opts_.max_feature_length : nframes;
		std::copy_n(input_features[i].Data(), n*opts_.feature_dim, begin);
		begin += opts_.max_feature_length*opts_.feature_dim;
		intput_length_buffer_[i] = n;
	}

	AcquireInitialTensors(batch_size);
	std::copy_n(intput_buffer_.begin(), intput_buffer_.size(), encoder_feature_.flat<float>().data());
	std::copy_n(intput_length_buffer_.begin(), intput_length_buffer_.size(), encoder_feature_length_.flat<int>().data());

	//定义模型输入
	inputs_ = {
		{"encoder_feature_data_placeholder", encoder_feature_},
		{"encoder_feature_length_placeholder", encoder_feature_length_}
	};

	Status status_run = session_->Run(inputs_, output_tensor_names_, {}, &outputs_);
	if (!status_run.ok())
		return -1;

	int *bpe_output_length = outputs_[1].flat<int32_t>().data();
	assert(batch_size == outputs_[1].shape().dim_size(0));

	out_dim = outputs_[0].shape().dim_size(2);
	loglikes.resize(batch_size);
	for (int i = 0; i < batch_size; i++) {
		Tensor loglike = outputs_[0].SubSlice(i);
		size = out_dim*bpe_output_length[i];
		loglikes[i].Resize(bpe_output_length[i], out_dim, kUndefined, kStrideEqualNumCols);
		std::copy_n(loglike.unaligned_flat<float>().data(), size, loglikes[i].Data());
	}
	return 0;
}

void TFTransformerWrapper::Destory() {
	if (session_ != NULL) {
		session_->Close();
		session_ = NULL;
	}
}

}  // namespace tfnet
}  // namespace kaldi
