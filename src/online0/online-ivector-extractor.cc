// online0/online-ivector-extractor.cc
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

#include "online0/online-ivector-extractor.h"

namespace kaldi {

OnlineIvectorExtractor::OnlineIvectorExtractor(std::string cfg) :
		feature_cfg_(NULL), feature_opts_(NULL), ivector_config_(NULL), ivector_info_(NULL),
		base_feature_pipeline_(NULL), ivector_feature_(NULL) {

	// base feature pipline config
	feature_cfg_ = new OnlineNnetFeaturePipelineConfig;
	ReadConfigFromFile(cfg, feature_cfg_);

	// main config
	ivector_config_ = new OnlineStreamIvectorExtractionConfig;
	ReadConfigFromFile(cfg, ivector_config_);
}

void OnlineIvectorExtractor::InitExtractor() {
	feature_opts_ = new OnlineNnetFeaturePipelineOptions(feature_cfg_);
	// base feature pipeline
	base_feature_pipeline_ = new OnlineNnetFeaturePipeline(*feature_opts_);

	ivector_info_ = new OnlineStreamIvectorExtractionInfo(*ivector_config_);
	// ivector feature pipeline
	ivector_feature_ = new OnlineStreamIvectorFeature(*ivector_info_, base_feature_pipeline_);

}

int OnlineIvectorExtractor::FeedData(void *data, int nbytes) {

	if (nbytes <= 0)
		return 0;

	int size = nbytes/sizeof(float);
	Vector<BaseFloat> wave_part(size, kSetZero);
	memcpy((char*)(wave_part.Data()), (char*)data, nbytes);
	ivector_feature_->AcceptWaveform(ivector_info_->samp_freq, wave_part);
	return 0;
}

Ivector* OnlineIvectorExtractor::GetCurrentIvector() {

	int num_frame_ready = ivector_feature_->NumFramesReady();

	ivector.ivector_.Resize(ivector_feature_->Dim());
	ivector_feature_->GetFrame(num_frame_ready-1, &ivector.ivector_);
	ivector.tot_frames_ = num_frame_ready;
	ivector.tot_ubm_loglike_ = ivector_feature_->UbmLogLikePerFrame() * num_frame_ready;
	ivector.tot_objf_impr_ = ivector_feature_->ObjfImprPerFrame() * num_frame_ready;
	return &ivector;
}

void OnlineIvectorExtractor::Reset() {

	base_feature_pipeline_->Reset();
	ivector.clear();

	if (ivector_feature_ != NULL) {
		delete ivector_feature_;
	}
	ivector_feature_ = new OnlineStreamIvectorFeature(*ivector_info_, base_feature_pipeline_);
}

void OnlineIvectorExtractor::Destory() {
	if (ivector_feature_ != NULL) {
		delete feature_cfg_; feature_cfg_ = NULL;
		delete feature_opts_; feature_opts_ = NULL;
		delete base_feature_pipeline_; base_feature_pipeline_ = NULL;
		delete ivector_feature_;	  ivector_feature_ = NULL;
		delete ivector_info_;	ivector_info_= NULL;
		delete ivector_config_;	ivector_config_ = NULL;
	}
}

}	// namespace kaldi




