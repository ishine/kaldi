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
		feature_opts_(NULL), ivector_config_(NULL), ivector_info_(NULL),
		base_feature_pipeline_(NULL), ivector_feature_(NULL) {

	// main config
	ivector_config_ = new OnlineStreamIvectorExtractionConfig;
	ReadConfigFromFile(cfg, ivector_config_);
}

void OnlineIvectorExtractor::InitExtractor() {
	feature_opts_ = new OnlineNnetFeaturePipelineOptions(ivector_config_->base_feature_cfg);
	// base feature pipeline
	base_feature_pipeline_ = new OnlineNnetFeaturePipeline(*feature_opts_);

	ivector_info_ = new OnlineStreamIvectorExtractionInfo(*ivector_config_);
	// ivector feature pipeline
	ivector_feature_ = new OnlineStreamIvectorFeature(*ivector_info_, base_feature_pipeline_);
    ivector.clear();
}

int OnlineIvectorExtractor::FeedData(void *data, int nbytes, FeatState state) {

	if (nbytes <= 0)
		return 0;

	int size = nbytes/sizeof(float);
	Vector<BaseFloat> wave_part(size, kSetZero);
	memcpy((char*)(wave_part.Data()), (char*)data, nbytes);
	base_feature_pipeline_->AcceptWaveform(ivector_config_->base_feature_cfg.samp_freq, wave_part);

	if (state == FEAT_END)
		base_feature_pipeline_->InputFinished();

	return 0;
}

Ivector* OnlineIvectorExtractor::GetCurrentIvector(int type) {

	int num_frame_ready = ivector_feature_->NumFramesReady();
	int dim = ivector_feature_->Dim();
	int lda_dim = ivector_feature_->LdaDim();

    if (num_frame_ready > 0) {
    		Vector<BaseFloat> raw_ivec(dim);
		ivector_feature_->GetFrame(num_frame_ready-1, &raw_ivec);
		if (type == 0)
			ivector.ivector_ = raw_ivec;
		else if (type == 1)
			ivector_feature_->LdaTransform(raw_ivec, ivector.ivector_);

		ivector.num_frames_ = num_frame_ready;
		ivector.ubm_loglike_perframe_ = ivector_feature_->UbmLogLikePerFrame();
		ivector.objf_impr_perframe_ = ivector_feature_->ObjfImprPerFrame();
    }
	return &ivector;
}

BaseFloat OnlineIvectorExtractor::GetScore(const VectorBase<BaseFloat> &ivec1, const VectorBase<BaseFloat> &ivec2) {
	KALDI_ASSERT(ivec1.Dim() == ivec2.Dim());

	BaseFloat norm, ratio;
	Vector<BaseFloat> norm_ivec1(ivec1),  norm_ivec2(ivec2);

	// normalize
	norm = norm_ivec1.Norm(2.0);
	ratio = norm / sqrt(norm_ivec1.Dim());
	if (ratio != 0.0) norm_ivec1.Scale(1.0 / ratio);

	norm = norm_ivec2.Norm(2.0);
	ratio = norm / sqrt(norm_ivec2.Dim());
	if (ratio != 0.0) norm_ivec2.Scale(1.0 / ratio);

	// dot product
	return VecVec(norm_ivec1, norm_ivec2);
}

void OnlineIvectorExtractor::GetEnrollSpeakerIvector(const std::vector<Vector<BaseFloat> > &ivectors,
											Vector<BaseFloat> &spk_ivector, int type) {
	int size = ivectors.size();
	BaseFloat norm, ratio;

	if (size > 0) {
		Vector<BaseFloat> mean_ivector(ivectors[0].Dim());

		for (int i = 0; i < size; i++) {
			Vector<BaseFloat> ivector(ivectors[i]);
			// normalize
			norm = ivector.Norm(2.0);
			ratio = norm / sqrt(ivector.Dim());
			if (ratio != 0.0) ivector.Scale(1.0 / ratio);
			// sum
			mean_ivector.AddVec(1.0, ivector);
		}
		// mean
		mean_ivector.Scale(1.0/size);

		if (type == 0) {
			spk_ivector = mean_ivector;
		}
		else if (type == 1) {
			// normalize
			norm = mean_ivector.Norm(2.0);
			ratio = norm / sqrt(mean_ivector.Dim());
			if (ratio != 0.0) mean_ivector.Scale(1.0 / ratio);
			ivector_feature_->LdaTransform(mean_ivector, spk_ivector);
		}
	}
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
		delete feature_opts_; feature_opts_ = NULL;
		delete base_feature_pipeline_; base_feature_pipeline_ = NULL;
		delete ivector_feature_;	  ivector_feature_ = NULL;
		delete ivector_info_;	ivector_info_= NULL;
		delete ivector_config_;	ivector_config_ = NULL;
	}
}

}	// namespace kaldi




