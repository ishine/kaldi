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
	extractor_config_ = new OnlineIvectorExtractorConfig;
	ReadConfigFromFile(cfg, extractor_config_);

	feature_opts_ = new OnlineNnetFeaturePipelineOptions(extractor_config_->ivector_config.base_feature_cfg);

	//vad options
	if (extractor_config_->vad_cfg != "")
		ReadConfigFromFile(extractor_config_->vad_cfg, &vad_opts_);

	// ivector global mean
	if (extractor_config_->mean_filename != "") {
		ReadKaldiObject(extractor_config_->mean_filename, &mean_vec_);
		// lda trainsform matrix
		if (extractor_config_->lda_filename != "")
			ReadKaldiObject(extractor_config_->lda_filename, &lda_transform_);
		// plda model
		if (extractor_config_->plda_filename != "") {
			ReadKaldiObject(extractor_config_->plda_filename, &plda_);
			if (extractor_config_->plda_cfg != "")
				ReadConfigFromFile(extractor_config_->plda_cfg, &plda_config_);
		}
	}
}

void OnlineIvectorExtractor::InitExtractor() {
	// base feature pipeline
	base_feature_pipeline_ = new OnlineNnetFeaturePipeline(*feature_opts_);

	ivector_info_ = new OnlineStreamIvectorExtractionInfo(extractor_config_->ivector_config);
	// ivector feature pipeline
	ivector_feature_ = new OnlineStreamIvectorFeature(*ivector_info_, base_feature_pipeline_);
    ivector.clear();
}

int OnlineIvectorExtractor::FeedData(void *data, int nbytes, FeatState state) {
	int size = nbytes/sizeof(float);

	if (size <= 0)
		return 0;

	Vector<BaseFloat> wav_buffer(size, kSetZero);
	memcpy((char*)(wav_buffer.Data()), (char*)data, nbytes);

	base_feature_pipeline_->AcceptWaveform(extractor_config_->ivector_config.base_feature_cfg.samp_freq, wav_buffer);

	if (state == FEAT_END)
		base_feature_pipeline_->InputFinished();

	return base_feature_pipeline_->NumFramesReady();
}

int OnlineIvectorExtractor::VadProcess(const Matrix<BaseFloat> &vad_feat,
		const Matrix<BaseFloat> &in, Matrix<BaseFloat> &out) {
	Vector<BaseFloat> vad_result(vad_feat.NumRows());
	ComputeVadEnergy(vad_opts_, vad_feat, &vad_result);

	int dim = 0;
	for (int i = 0; i < vad_result.Dim(); i++) {
		if (vad_result(i) != 0)
			dim++;
	}

	if (dim == 0) return dim;

	out.Resize(dim, in.NumCols(), kUndefined);
	int32 index = 0;
	for (int i = 0; i < in.NumRows(); i++) {
        if (vad_result(i) != 0.0) {
          KALDI_ASSERT(vad_result(i) == 1.0); // should be zero or one.
          out.Row(index).CopyFromVec(in.Row(i));
          index++;
        }
	}
	KALDI_ASSERT(index == dim);
	return dim;
}

// ivector-normalize-length
void OnlineIvectorExtractor::IvectorLengthNormalize(Vector<BaseFloat> &ivector) {
	BaseFloat norm = ivector.Norm(2.0);
	BaseFloat ratio = norm / sqrt(ivector.Dim()); // how much larger it is
													// than it would be, in
													// expectation, if normally
	if (ratio == 0.0) {
	  KALDI_WARN << "Zero xVector";
	} else {
		ivector.Scale(1.0 / ratio);
	}
}

// post processing
void OnlineIvectorExtractor::IvectorPostProcess(const VectorBase<BaseFloat> &in, Vector<BaseFloat> &out, int type) {
	int transform_rows, transform_cols, vec_dim;
	Vector<BaseFloat> ivector_raw(in);
	Vector<BaseFloat> ivector_lda;
	Vector<BaseFloat>  *final_ivec = &ivector_raw;

	// lda trainsform
	if (type > 0) {
		// subtract global mean
		ivector_raw.AddVec(-1.0, mean_vec_);

		transform_rows = lda_transform_.NumRows();
		transform_cols = lda_transform_.NumCols();
		vec_dim = ivector_raw.Dim();

        ivector_lda.Resize(transform_rows, kUndefined);
		if (transform_cols == vec_dim) {
			ivector_lda.AddMatVec(1.0, lda_transform_, kNoTrans, ivector_raw, 0.0);
		} else {
			KALDI_ASSERT(transform_cols == vec_dim + 1);
			ivector_lda.CopyColFromMat(lda_transform_, vec_dim);
			ivector_lda.AddMatVec(1.0, lda_transform_.Range(0, transform_rows, 0, vec_dim), kNoTrans, ivector_raw, 1.0);
		}

		final_ivec = &ivector_lda;
	}

	// normalize length
	IvectorLengthNormalize(*final_ivec);
    out = *final_ivec;
}

Ivector* OnlineIvectorExtractor::GetCurrentIvector(int type) {

	int num_frame_ready = ivector_feature_->NumFramesReady();
	int dim = ivector_feature_->Dim();

	if (num_frame_ready <= 0)
		return NULL;

	Vector<BaseFloat> raw_ivec(dim);
	ivector_feature_->GetFrame(num_frame_ready-1, &raw_ivec);

	if (extractor_config_->use_post)
		IvectorPostProcess(raw_ivec, ivector_.ivector_, type);
	else
		ivector_.ivector_ = raw_ivec;

	ivector_.num_frames_ = num_frame_ready;
	ivector_.ubm_loglike_perframe_ = ivector_feature_->UbmLogLikePerFrame();
	ivector_.objf_impr_perframe_ = ivector_feature_->ObjfImprPerFrame();

	return &ivector_;
}
BaseFloat OnlineIvectorExtractor::GetScore(const VectorBase<BaseFloat> &train_ivec, int num_utts,
		const VectorBase<BaseFloat> &test_ivec, int type = 2) {
	KALDI_ASSERT(train_ivec.Dim() == test_ivec.Dim());

	Vector<BaseFloat> train_post, test_post;

	if (!extractor_config_->use_post) {
		IvectorPostProcess(train_ivec, train_post, type);
		IvectorPostProcess(test_ivec, test_post, type);
	} else {
		train_post = train_ivec;
		test_post = test_ivec;
	}

	BaseFloat score = 0;
	if (type < 2) {
		// dot product
		score = VecVec(train_post, test_post);
	} else if(type == 2) {
		int dim = plda_.Dim();
		Vector<BaseFloat> transformed_ivec1(dim);
		Vector<BaseFloat> transformed_ivec2(dim);
		plda_.TransformIvector(plda_config_, train_post, num_utts, &transformed_ivec1);
		plda_.TransformIvector(plda_config_, test_post, 1, &transformed_ivec2);

		Vector<double> train_ivector_dbl(transformed_ivec1), test_ivector_dbl(transformed_ivec2);
		score = plda_.LogLikelihoodRatio(train_ivector_dbl, num_utts, test_ivector_dbl);
	}

	return score;
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
			ivector_feature_->LdaTransform(mean_ivector, spk_ivector);
		}
	}
}

int OnlineIvectorExtractor::GetIvectorDim() {
	int raw_dim = ivector_feature_->Dim();
	int lda_dim = lda_transform_.NumRows();
	int plda_dim = plda_.Dim();

	int ivec_dim = 0;

    if (extractor_config_->use_post) {
	    if (plda_dim > 0)
	    	ivec_dim = plda_dim;
	    else if (lda_dim > 0)
	    	ivec_dim = lda_dim;
    } else
    	ivec_dim = raw_dim;
	return ivec_dim;
}

int OnlineIvectorExtractor::GetAudioFrequency() {
    return feature_opts_->samp_freq;
}

void OnlineIvectorExtractor::Reset() {

	base_feature_pipeline_->Reset();
	ivector_.clear();

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
	}
}

}	// namespace kaldi




