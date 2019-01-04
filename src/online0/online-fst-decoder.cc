// online0/Online-fst-decoder.cc
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

#include "online0/online-util.h"
#include "online0/online-fst-decoder.h"

namespace kaldi {

OnlineFstDecoder::OnlineFstDecoder(OnlineFstDecoderCfg *cfg) :
		decoder_cfg_(cfg), decoder_opts_(cfg->decoder_opts_), forward_opts_(cfg->forward_opts_),
		feature_opts_(cfg->feature_opts_), decoding_opts_(cfg->decoding_opts_),
		trans_model_(cfg->trans_model_), decode_fst_(cfg->decode_fst_), word_syms_(cfg->word_syms_), 
		block_(NULL), decodable_(NULL), decoder_(NULL), decoding_(NULL), decoder_thread_(NULL),
		feature_pipeline_(NULL), forward_(NULL),
		words_writer_(NULL), alignment_writer_(NULL), state_(FEAT_START),
		len_(0), sample_offset_(0), frame_offset_(0), frame_ready_(0),
		in_skip_(0), out_skip_(0), chunk_length_(0), cur_result_idx_(0) {
}

void OnlineFstDecoder::Destory() {
	Abort();

	if (decodable_ != NULL) {
		delete decodable_;	decodable_ = NULL;
	}

	if (words_writer_ != NULL) {
		delete words_writer_;	words_writer_ = NULL;
	}

	if (alignment_writer_ != NULL) {
		delete alignment_writer_; alignment_writer_ = NULL;
	}

	if (decoder_ != NULL) {
		delete decoder_;	decoder_ = NULL;
		delete decoding_;	decoding_ = NULL;
		delete decoder_thread_;		decoder_thread_ = NULL;
		delete feature_pipeline_;	feature_pipeline_ = NULL;
		delete forward_;			forward_ = NULL;
	}
}

void OnlineFstDecoder::InitDecoder() {
#if HAVE_CUDA==1
    if (forward_opts_->use_gpu == "yes")
        CuDevice::Instantiate().Initialize();
#endif

	// decodable feature pipe to decoder
	if (decoding_opts_->model_rspecifier != "")
		decodable_ = new OnlineDecodableMatrixMapped(*trans_model_, decoding_opts_->acoustic_scale);
	else
		decodable_ = new OnlineDecodableMatrixCtc(decoding_opts_->acoustic_scale);

	if (decoding_opts_->words_wspecifier != "")
		words_writer_ = new Int32VectorWriter(decoding_opts_->words_wspecifier);
	if (decoding_opts_->alignment_wspecifier != "")
		alignment_writer_ = new Int32VectorWriter(decoding_opts_->alignment_wspecifier);

	// decoder
	decoder_ = new OnlineNnetFasterDecoder(*decode_fst_, *decoder_opts_);
	decoding_ = new OnlineNnetDecodingClass(*decoding_opts_,
		    								decoder_, decodable_, &repository_,
											&result_);
	decoder_thread_ = new MultiThreader<OnlineNnetDecodingClass>(1, *decoding_);

	// feature pipeline
	feature_pipeline_ = new OnlineNnetFeaturePipeline(*feature_opts_);

	// forward
	forward_ = new OnlineNnetForward(*forward_opts_);

	// decoding buffer
	in_skip_ = decoding_opts_->skip_inner ? 1:decoding_opts_->skip_frames;
	out_skip_ = decoding_opts_->skip_inner ? decoding_opts_->skip_frames : 1;

	int feat_dim = feature_pipeline_->Dim();
	feat_in_.Resize(out_skip_*forward_opts_->batch_size, feat_dim);
	// wav buffer
	wav_buffer_.Resize(VECTOR_INC_STEP, kSetZero); // 16k, 10s

	if (decoding_opts_->chunk_length_secs > 0) {
		chunk_length_ = int32(feature_opts_->samp_freq * decoding_opts_->chunk_length_secs);
		if (chunk_length_ == 0) chunk_length_ = 1;
	} else {
		chunk_length_ = std::numeric_limits<int32>::max();
	}
}

void OnlineFstDecoder::Reset() {
	feature_pipeline_->Reset();
	forward_->ResetHistory();
	result_.clear();
	len_ = 0;
	sample_offset_ = 0;
	frame_offset_ = 0;
	frame_ready_ = 0;
	cur_result_idx_ = 0;
	state_ = FEAT_START;
	wav_buffer_.Resize(VECTOR_INC_STEP, kUndefined); // 16k, 10s
}

void OnlineFstDecoder::FeedData(void *data, int nbytes, FeatState state) {
	// extend buffer
	if (wav_buffer_.Dim() < len_+nbytes/sizeof(float)) {
        int size = std::max((int)(wav_buffer_.Dim()+VECTOR_INC_STEP), int(len_+nbytes/sizeof(float)));
		Vector<BaseFloat> tmp(size, kUndefined);
		memcpy((char*)tmp.Data(), (char*)wav_buffer_.Data(), len_*sizeof(float));
		wav_buffer_.Swap(&tmp);
	}

	BaseFloat *wav_data = wav_buffer_.Data();
	if (nbytes > 0) {
		memcpy((char*)(wav_data+len_), (char*)data, nbytes);
		len_ += nbytes/sizeof(float);
	}

	int32 samp_remaining = len_ - sample_offset_;
	int32 batch_size = forward_opts_->batch_size * decoding_opts_->skip_frames;

	if (sample_offset_ <= len_) {
		SubVector<BaseFloat> wave_part(wav_buffer_, sample_offset_, samp_remaining);
		feature_pipeline_->AcceptWaveform(feature_opts_->samp_freq, wave_part);
		sample_offset_ += samp_remaining;

		if (state == FEAT_END)
			feature_pipeline_->InputFinished();

		while (true) {
			frame_ready_ = feature_pipeline_->NumFramesReady();
			if (!feature_pipeline_->IsLastFrame(frame_ready_-1) && frame_ready_ < frame_offset_+batch_size)
				break;
            else if (feature_pipeline_->IsLastFrame(frame_ready_-1) && frame_ready_ == frame_offset_)
                break;
			else if (feature_pipeline_->IsLastFrame(frame_ready_-1) && frame_ready_ < frame_offset_+batch_size) {
				frame_ready_ -= frame_offset_;
				feat_in_.SetZero();
			}
			else
				frame_ready_ = batch_size;

			for (int i = 0; i < frame_ready_; i += in_skip_) {
				// feature_pipeline_->GetFrame(frame_offset_+i, &feat_in_.Row(i/in_skip_));
                SubVector<BaseFloat> row(feat_in_, i/in_skip_);
                feature_pipeline_->GetFrame(frame_offset_+i, &row);
			}

			frame_offset_ += frame_ready_;
			// feed forward to neural network
			forward_->Forward(feat_in_, &feat_out_);

			// copy posterior
			if (decoding_opts_->copy_posterior) {
				feat_out_ready_.Resize(frame_ready_, feat_out_.NumCols(), kUndefined);
				for (int i = 0; i < frame_ready_; i++)
					feat_out_ready_.Row(i).CopyFromVec(feat_out_.Row(i/decoding_opts_->skip_frames));
			} else {
				int out_frames = (frame_ready_+out_skip_-1)/out_skip_;
				feat_out_ready_.Resize(out_frames, feat_out_.NumCols(), kUndefined);
				feat_out_ready_.CopyFromMat(feat_out_.RowRange(0, out_frames));
			}

			block_ = new OnlineDecodableBlock(feat_out_ready_, state);
			repository_.Accept(block_);

			// wake up decoder thread
			result_.num_frames += frame_ready_;
		}
	}
}

Result* OnlineFstDecoder::GetResult(FeatState state) {
	std::vector<int32> word_ids;
	int size = result_.word_ids_.size();
	for (int i = cur_result_idx_; i < size; i++)
		word_ids.push_back(result_.word_ids_[i]);

    bool newutt = (state == FEAT_END);

    if (newutt == FEAT_END) {
    	while (!result_.isend)
    		sleep(0.02);
    }

	PrintPartialResult(word_ids, word_syms_, newutt);
    std::cout.flush();

	cur_result_idx_ = size;
	return &result_;
}

void OnlineFstDecoder::Abort() {
	repository_.Done();
	sleep(0.1);
}

}	// namespace kaldi




