// online0/online-xvector-extractor.cc
// Copyright 2018	Alibaba Inc (author: Wei Deng)

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

#include "online-xvector-extractor.h"

namespace kaldi {

OnlineXvectorExtractor::OnlineXvectorExtractor(std::string cfg) :
		xvector_config_(NULL), feature_opts_(NULL),
		forward_opts_(NULL),
		feature_pipeline_(NULL), forward_(NULL), speex_decoder_(NULL) {
	// main config
	xvector_config_ = new OnlineXvectorExtractorConfig;
	ReadConfigFromFile(cfg, xvector_config_);

	feature_opts_ = new OnlineNnetFeaturePipelineOptions(xvector_config_->feature_cfg);
	// neural network forward options
	forward_opts_ = new OnlineNnet3ForwardOptions;
	ReadConfigFromFile(xvector_config_->forward_cfg, forward_opts_);

	//vad options
	if (xvector_config_->vad_cfg != "")
		ReadConfigFromFile(xvector_config_->vad_cfg, &vad_opts_);

	// speex options
	if (xvector_config_->use_speex) {
        if (xvector_config_->speex_cfg != "")
		    ReadConfigFromFile(xvector_config_->speex_cfg, &speex_opts_);
		speex_decoder_ = new OnlineSpeexDecoder(speex_opts_);
	}

	// xvector global mean
	if (xvector_config_->mean_filename != "") {
		ReadKaldiObject(xvector_config_->mean_filename, &mean_vec_);
		// lda trainsform matrix
		if (xvector_config_->lda_filename != "")
			ReadKaldiObject(xvector_config_->lda_filename, &lda_transform_);
		// plda model
		if (xvector_config_->plda_filename != "") {
			ReadKaldiObject(xvector_config_->plda_filename, &plda_);
			if (xvector_config_->plda_cfg != "")
				ReadConfigFromFile(xvector_config_->plda_cfg, &plda_config_);
		}
	}
}

void OnlineXvectorExtractor::InitExtractor() {
	// base feature pipeline
	feature_pipeline_ = new OnlineNnetFeaturePipeline(*feature_opts_);
	// forward
	forward_ = new OnlineNnet3Forward(forward_opts_);

    xvector_.clear();
}

int OnlineXvectorExtractor::FeedData(void *data, int nbytes, FeatState state) {
	if (nbytes <= 0) return 0;

	Vector<BaseFloat> wav_buffer;
	if (xvector_config_->use_speex) {
		std::vector<char> speex_bits_part(nbytes);
		memcpy((char*)(&speex_bits_part.front()), (char*)data, nbytes);
		speex_decoder_->AcceptSpeexBits(speex_bits_part);
		speex_decoder_->GetWaveform(&wav_buffer);
	} else {
		int size = nbytes/sizeof(float);
		if (size <= 0) return 0;
		wav_buffer.Resize(size, kUndefined);
		memcpy((char*)(wav_buffer.Data()), (char*)data, nbytes);
	}


	feature_pipeline_->AcceptWaveform(feature_opts_->samp_freq, wav_buffer);

	if (state == FEAT_END)
		feature_pipeline_->InputFinished();

	return feature_pipeline_->NumFramesReady();
}

int OnlineXvectorExtractor::VadProcess(const Matrix<BaseFloat> &vad_feat,
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
void OnlineXvectorExtractor::XvectorLengthNormalize(Vector<BaseFloat> &xvector) {
	BaseFloat norm = xvector.Norm(2.0);
	BaseFloat ratio = norm / sqrt(xvector.Dim()); // how much larger it is
													// than it would be, in
													// expectation, if normally
	if (ratio == 0.0) {
	  KALDI_WARN << "Zero xVector";
	} else {
		xvector.Scale(1.0 / ratio);
	}
}

// post processing
void OnlineXvectorExtractor::XvectorPostProcess(const VectorBase<BaseFloat> &in, Vector<BaseFloat> &out, int type) {
	int transform_rows, transform_cols, vec_dim;
	Vector<BaseFloat> xvector_raw(in);
	Vector<BaseFloat> xvector_lda;
	Vector<BaseFloat>  *final_xvec = &xvector_raw;

	// lda trainsform
	if (type > 0) {
		// subtract global mean
		xvector_raw.AddVec(-1.0, mean_vec_);

		transform_rows = lda_transform_.NumRows();
		transform_cols = lda_transform_.NumCols();
		vec_dim = xvector_raw.Dim();

        xvector_lda.Resize(transform_rows, kUndefined);
		if (transform_cols == vec_dim) {
			xvector_lda.AddMatVec(1.0, lda_transform_, kNoTrans, xvector_raw, 0.0);
		} else {
			KALDI_ASSERT(transform_cols == vec_dim + 1);
			xvector_lda.CopyColFromMat(lda_transform_, vec_dim);
			xvector_lda.AddMatVec(1.0, lda_transform_.Range(0, transform_rows, 0, vec_dim), kNoTrans, xvector_raw, 1.0);
		}

		final_xvec = &xvector_lda;
	}

	// normalize length
	XvectorLengthNormalize(*final_xvec);
    out = *final_xvec;
}

Xvector* OnlineXvectorExtractor::GetCurrentXvector(int type) {

	int num_frame_ready, num_rows, feat_dim,
		this_chunk_size, chunk_size, min_chunk_size, xvector_dim;
	bool pad_input = xvector_config_->pad_input;

	num_frame_ready = feature_pipeline_->NumFramesReady();
	if (num_frame_ready <= 0)
		return NULL;

	// input features
	feat_in_.Resize(num_frame_ready, feature_pipeline_->Dim(), kUndefined);
	for (int i = 0; i < num_frame_ready; i++) {
		// feature_pipeline_->GetFrame(frame_offset_+i, &feat_in_.Row(i/in_skip_));
		SubVector<BaseFloat> row(feat_in_, i);
		feature_pipeline_->GetFrame(i, &row);
	}

	// vad
	if (xvector_config_->vad_cfg != "") {
		OnlineStreamBaseFeature *base_feature = feature_pipeline_->GetBaseFeature();
		feat_in_vad_.Resize(num_frame_ready, base_feature->Dim(), kUndefined);
		for (int i = 0; i < num_frame_ready; i++) {
			SubVector<BaseFloat> row(feat_in_vad_, i);
			base_feature->GetFrame(i, &row);
		}
		num_rows = VadProcess(feat_in_vad_, feat_in_, feat_out_);
	} else {
		feat_out_ = feat_in_;
        num_rows = feat_out_.NumRows();
	}


	if (num_rows <= 0)
		return NULL;

	// extract raw xvector
	num_rows = feat_out_.NumRows();
	feat_dim = feat_out_.NumCols();
	chunk_size = xvector_config_->chunk_size;
	min_chunk_size = xvector_config_->min_chunk_size;
	this_chunk_size = xvector_config_->chunk_size;
	if (!pad_input && num_rows < min_chunk_size) {
		KALDI_WARN << "Minimum chunk size of " << min_chunk_size
		                   << " is greater than the number of rows "
		                   << "in utterance.";
		return NULL;
	} else if (num_rows < chunk_size) {
		this_chunk_size = num_rows;
	} else if (chunk_size == -1) {
        this_chunk_size = num_rows;
    }

	int num_chunks = ceil(num_rows / static_cast<BaseFloat>(this_chunk_size));
	xvector_dim = forward_->OutputDim();

	Vector<BaseFloat> xvector_avg(xvector_dim, kSetZero);
	BaseFloat tot_weight = 0.0;

	// Iterate over the feature chunks.
	for (int32 chunk_indx = 0; chunk_indx < num_chunks; chunk_indx++) {
		// If we're nearing the end of the input, we may need to shift the
		// offset back so that we can get this_chunk_size frames of input to
		// the nnet.
		int32 offset = std::min(this_chunk_size, num_rows - chunk_indx * this_chunk_size);
		if (!pad_input && offset < min_chunk_size)
			continue;
		SubMatrix<BaseFloat> sub_features(feat_out_, chunk_indx * this_chunk_size, offset, 0, feat_dim);
		Vector<BaseFloat> xvector(xvector_dim, kUndefined);
		tot_weight += offset;

		// Pad input if the offset is less than the minimum chunk size
		if (pad_input && offset < min_chunk_size) {
			Matrix<BaseFloat> padded_features(min_chunk_size, feat_dim);

			int32 nrepeat = min_chunk_size / num_rows;
			for (int32 i = 0; i < nrepeat; i++) {
			  padded_features.Range(min_chunk_size-(i+1)*num_rows, num_rows, 0, feat_dim).CopyFromMat(
					  feat_out_.Range(0, num_rows, 0, feat_dim));
			}

			int32 left = min_chunk_size % num_rows;
			if (left > 0)
			padded_features.Range(0, left, 0, feat_dim).CopyFromMat(
					feat_out_.Range(num_rows-left, left, 0, feat_dim));

			forward_->Forward(padded_features, &nnet_out_);
		} else {
			forward_->Forward(sub_features, &nnet_out_);
		}
		xvector.CopyFromVec(nnet_out_.Row(0));
		xvector_avg.AddVec(offset, xvector);
	}
	xvector_avg.Scale(1.0/tot_weight);

	if (xvector_config_->use_post)
		XvectorPostProcess(xvector_avg, xvector_.xvector_, type);
	else
		xvector_.xvector_ = xvector_avg;

	xvector_.num_frames_ = num_rows;
	xvector_.isvalid_ = true;
	return &xvector_;
}

BaseFloat OnlineXvectorExtractor::GetScore(const VectorBase<BaseFloat> &train_xvec, int num_utts,
		const VectorBase<BaseFloat> &test_xvec, int type) {
	KALDI_ASSERT(train_xvec.Dim() == test_xvec.Dim());

	Vector<BaseFloat> train_post, test_post;

	if (!xvector_config_->use_post) {
		XvectorPostProcess(train_xvec, train_post, type);
		XvectorPostProcess(test_xvec, test_post, type);
	} else {
		train_post = train_xvec;
		test_post = test_xvec;
	}

	BaseFloat score = 0;
	if (type < 2) {
		// dot product
		score = VecVec(train_post, test_post);
	} else if(type == 2) {
		int dim = plda_.Dim();
		Vector<BaseFloat> transformed_xvec1(dim);
		Vector<BaseFloat> transformed_xvec2(dim);
		plda_.TransformIvector(plda_config_, train_post, num_utts, &transformed_xvec1);
		plda_.TransformIvector(plda_config_, test_post, 1, &transformed_xvec2);

		Vector<double> train_ivector_dbl(transformed_xvec1), test_ivector_dbl(transformed_xvec2);
		score = plda_.LogLikelihoodRatio(train_ivector_dbl, num_utts, test_ivector_dbl);
	}

	return score;
}

void OnlineXvectorExtractor::GetEnrollSpeakerXvector(const std::vector<Vector<BaseFloat> > &xvectors,
											Vector<BaseFloat> &spk_xvector, int type) {
}

int OnlineXvectorExtractor::GetXvectorDim() {
	int raw_dim = forward_->OutputDim();
	int lda_dim = lda_transform_.NumRows();
	int plda_dim = plda_.Dim();

	int xvec_dim = 0;

    if (xvector_config_->use_post) {
	    if (plda_dim > 0)
		    xvec_dim = plda_dim;
	    else if (lda_dim > 0)
		    xvec_dim = lda_dim;
    } else
		xvec_dim = raw_dim;
	return xvec_dim;
}

int OnlineXvectorExtractor::GetAudioFrequency() {
    return feature_opts_->samp_freq;    
}

void OnlineXvectorExtractor::Reset() {
	feature_pipeline_->Reset();
	xvector_.clear();
	if (xvector_config_->use_speex) {
		delete speex_decoder_;
		speex_decoder_ = new OnlineSpeexDecoder(speex_opts_);
	}
}

void OnlineXvectorExtractor::Destory() {
	if (forward_ != NULL) {
		delete feature_opts_; feature_opts_ = NULL;
		delete feature_pipeline_; feature_pipeline_ = NULL;
		delete xvector_config_;	xvector_config_ = NULL;
		delete forward_; forward_ = NULL;
	}
}

}	// namespace kaldi




