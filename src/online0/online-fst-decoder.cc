// online0/online-fst-decoder.cc
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
		decoder_cfg_(cfg), fast_decoder_opts_(cfg->fast_decoder_opts_),
		lat_decoder_opts_(cfg->lat_decoder_opts_),
		forward_opts_(cfg->forward_opts_),
		feature_opts_(cfg->feature_opts_),
		decoding_opts_(cfg->decoding_opts_),
		vad_opts_(cfg->vad_opts_),
		trans_model_(cfg->trans_model_), decode_fst_(cfg->decode_fst_), word_syms_(cfg->word_syms_), 
		block_(NULL), decodable_(NULL),
		fast_decoder_(NULL), fast_decoding_(NULL), fast_decoder_thread_(NULL),
		lat_decoder_(NULL), lat_decoding_(NULL), lat_decoder_thread_(NULL), vad_(NULL),
		feature_pipeline_(NULL), forward_(NULL), ipc_socket_(NULL),
		words_writer_(NULL), alignment_writer_(NULL), state_(FEAT_START), utt_state_(UTT_END),
		socket_sample_(NULL), sc_sample_buffer_(NULL), sc_buffer_size_(0),
		len_(0), sample_offset_(0), frame_offset_(0), frame_ready_(0),
		in_skip_(0), out_skip_(0), skip_frames_(1), chunk_length_(0), cur_result_idx_(0), 
		finish_utt_(false) {
}

void OnlineFstDecoder::Destory() {
	Abort();

    if (ipc_socket_ != NULL) {
        delete ipc_socket_; ipc_socket_ = NULL;
    }

    if (fast_decoder_thread_ != NULL) {
        delete fast_decoder_thread_; fast_decoder_thread_ = NULL;
		delete fast_decoding_;	fast_decoding_ = NULL;
		delete fast_decoder_;	fast_decoder_ = NULL;
		delete feature_pipeline_;	feature_pipeline_ = NULL;
    }

	if (lat_decoder_thread_ != NULL) {
		delete lat_decoder_thread_; lat_decoder_thread_ = NULL;
		delete lat_decoding_;	lat_decoding_ = NULL;
		delete lat_decoder_;	lat_decoder_ = NULL;
	}

    if (forward_ != NULL) {
    	delete forward_;	forward_ = NULL;
    }

    if (sc_sample_buffer_ != NULL) {
    	delete sc_sample_buffer_;	sc_sample_buffer_ = NULL;
    }

	if (decodable_ != NULL) {
		delete decodable_;	decodable_ = NULL;
	}

	if (words_writer_ != NULL) {
		delete words_writer_;	words_writer_ = NULL;
	}

	if (alignment_writer_ != NULL) {
		delete alignment_writer_; alignment_writer_ = NULL;
	}

	if (vad_ != NULL) {
		delete vad_; vad_ = NULL;
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

	// feature pipeline
	feature_pipeline_ = new OnlineNnetFeaturePipeline(*feature_opts_);

	// decoding buffer
	skip_frames_ = decoding_opts_->skip_frames;
	in_skip_ = decoding_opts_->skip_inner ? 1 : skip_frames_;
	out_skip_ = decoding_opts_->skip_inner ? skip_frames_ : 1;

	int feat_dim = feature_pipeline_->Dim();
	int in_rows = decoding_opts_->batch_size;
	feat_in_.Resize(in_rows, feat_dim, kUndefined, kStrideEqualNumCols);
	// wav buffer
	wav_buffer_.Resize(VECTOR_INC_STEP, kSetZero); // 16k, 10s
	// fsmn
	utt_state_flags_.resize(forward_opts_->num_stream, 0);
	valid_input_frames_.resize(forward_opts_->num_stream, 0);

	// forward
	if (!decoding_opts_->use_ipc && decoding_opts_->forward_cfg != "") {
		forward_ = new OnlineNnetForward(*forward_opts_);
		KALDI_LOG << "Use forward per decoder" ;
	} else if (decoding_opts_->use_ipc && decoding_opts_->socket_path != "") {
		ipc_socket_= new UnixDomainSocket(decoding_opts_->socket_path, SOCK_STREAM);
		sc_buffer_size_ = sizeof(SocketSample) + in_rows*feat_dim*sizeof(BaseFloat);
		sc_sample_buffer_ = new char[sc_buffer_size_];
		socket_sample_ = (SocketSample*)sc_sample_buffer_;
        socket_sample_->clear();
		socket_sample_->pid = getpid();
		socket_sample_->dim = feat_dim;
		KALDI_LOG << "Use ipc socket forward, socket file path: "<< decoding_opts_->socket_path ;
	} else
		KALDI_ERR << "No forward conf or ipc socket forward file path";

	// decoder
	if (decoding_opts_->use_lat) {
		lat_decoder_ = new OnlineLatticeFasterDecoder(*decode_fst_, *lat_decoder_opts_);
		lat_decoding_ = new OnlineNnetLatticeDecodingClass(*decoding_opts_, lat_decoder_,
				decodable_, &repository_, ipc_socket_, &result_);
		lat_decoder_thread_ = new MultiThreader<OnlineNnetLatticeDecodingClass>(1, *lat_decoding_);
	} else {
		fast_decoder_ = new OnlineFasterDecoder(*decode_fst_, *fast_decoder_opts_);
		fast_decoding_ = new OnlineNnetDecodingClass(*decoding_opts_, fast_decoder_,
				decodable_, &repository_, ipc_socket_, &result_);
		fast_decoder_thread_ = new MultiThreader<OnlineNnetDecodingClass>(1, *fast_decoding_);
	}

	if (decoding_opts_->chunk_length_secs > 0) {
		chunk_length_ = int32(feature_opts_->samp_freq * decoding_opts_->chunk_length_secs);
		if (chunk_length_ == 0) chunk_length_ = 1;
	} else {
		chunk_length_ = std::numeric_limits<int32>::max();
	}

	if (decoding_opts_->use_vad && vad_opts_ != NULL) {
		vad_ = new OnlineVad(*vad_opts_);
		utt_state_ = UTT_END;
	}
}

void OnlineFstDecoder::Reset() {
	feature_pipeline_->Reset();
    if(!decoding_opts_->use_ipc)
	    forward_->ResetHistory();
	result_.clear();
	len_ = 0;
	sample_offset_ = 0;
	frame_offset_ = 0;
	frame_ready_ = 0;
	cur_result_idx_ = 0;
	state_ = FEAT_START;
	utt_state_ = UTT_END;
	finish_utt_ = false;
    if (vad_ != NULL) vad_->Reset();
	wav_buffer_.Resize(VECTOR_INC_STEP, kUndefined); // 16k, 10s
}

void OnlineFstDecoder::ResetUtt() {
    if(!decoding_opts_->use_ipc)
	    forward_->ResetHistory();
}

int OnlineFstDecoder::FeedData(void *data, int nbytes, FeatState state) {
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
	int32 batch_size = decoding_opts_->batch_size;
    FeatState pos_state;
    UttState cur_state;

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
			else if (feature_pipeline_->IsLastFrame(frame_ready_-1) && frame_ready_ <= frame_offset_+batch_size) {
				frame_ready_ -= frame_offset_;
				feat_in_.SetZero();
                pos_state = FEAT_END;
			} else {
				frame_ready_ = batch_size;
                pos_state = frame_offset_ == 0 ? FEAT_START : FEAT_APPEND;
            }

			for (int i = 0; i < frame_ready_; i += in_skip_) {
				// feature_pipeline_->GetFrame(frame_offset_+i, &feat_in_.Row(i/in_skip_));
                SubVector<BaseFloat> row(feat_in_, i/in_skip_);
                feature_pipeline_->GetFrame(frame_offset_+i, &row);
			}

			frame_offset_ += frame_ready_;
			result_.num_frames += frame_ready_;

			if (!decoding_opts_->use_ipc) {
				// feed forward to neural network
				if (forward_opts_->network_type == "lstm") {
					forward_->Forward(feat_in_, &feat_out_, &blank_post_);
					// copy posterior
					if (decoding_opts_->copy_posterior) {
						feat_out_ready_.Resize(frame_ready_, feat_out_.NumCols(), kUndefined);
						for (int i = 0; i < frame_ready_; i++)
							feat_out_ready_.Row(i).CopyFromVec(feat_out_.Row(i/skip_frames_));
					} else {
						int out_frames = (frame_ready_+out_skip_-1)/out_skip_;
						feat_out_ready_.Resize(out_frames, feat_out_.NumCols(), kUndefined);
						feat_out_ready_.CopyFromMat(feat_out_.RowRange(0, out_frames));
					}

					block_ = NULL;
					if (vad_ != NULL) {
						cur_state = vad_->FeedData(blank_post_);
						if (cur_state != UTT_END || utt_state_ != UTT_END || (cur_state != UTT_END && pos_state == FEAT_END)) {
                            FeatState state = (pos_state==FEAT_END||cur_state==UTT_END) ? FEAT_END : FEAT_APPEND;
							block_ = new OnlineDecodableBlock(feat_out_ready_, state);
							// wake up decoder thread
							if (block_ != NULL) repository_.Accept(block_);
                            utt_state_ = cur_state;
                            if (cur_state == UTT_START)
                                KALDI_LOG << "START: << " << result_.num_frames << " f, "<< 
                                result_.num_frames/(100*60) << ":" << result_.num_frames%(100*60)/100 << "," << result_.num_frames%100*10;
                                
							if (cur_state == UTT_END) {
								finish_utt_ = true;
								ResetUtt();
								// break output an utterance result
								break;
							}
						}
                        if (cur_state == UTT_END && pos_state == FEAT_END) {
                           result_.isuttend = true; 
                           finish_utt_ = true;
                        }
					} else {
						block_ = new OnlineDecodableBlock(feat_out_ready_, pos_state);
						// wake up decoder thread
						if (block_ != NULL) repository_.Accept(block_);
					}
				} else if (forward_opts_->network_type == "unifsmn") {
					int n = 1, nframe = 0, N = pos_state == FEAT_END ? 2 : 1;
					while (n <= N) {
						utt_state_flags_[0] = (pos_state != FEAT_END) ? pos_state : 1;
						utt_state_flags_[0] = (n == 2) ? FEAT_END : utt_state_flags_[0];
						valid_input_frames_[0] = (frame_ready_+out_skip_-1)/out_skip_;
						valid_input_frames_[0] = (n == 2) ? 0 : valid_input_frames_[0];
						forward_->SetStreamStatus(utt_state_flags_, &valid_input_frames_);
						forward_->Forward(feat_in_, &feat_out_, &blank_post_);
						nframe = decoding_opts_->copy_posterior ?
								valid_input_frames_[0]*skip_frames_ : valid_input_frames_[0];
						feat_out_ready_.Resize(nframe, feat_out_.NumCols(), kUndefined);
						if (decoding_opts_->copy_posterior) {
							for (int i = 0; i < nframe; i++)
								feat_out_ready_.Row(i).CopyFromVec(feat_out_.Row(i/skip_frames_));
						} else {
							feat_out_ready_.CopyFromMat(feat_out_.RowRange(0, nframe));
						}

						block_ = NULL;
						if (vad_ != NULL) {
							cur_state = vad_->FeedData(blank_post_);
							if (cur_state != UTT_END || utt_state_ != UTT_END || (cur_state != UTT_END && pos_state == FEAT_END)) {
                                FeatState state = (pos_state==FEAT_END||cur_state==UTT_END) ? FEAT_END : FEAT_APPEND;
								block_ = new OnlineDecodableBlock(feat_out_ready_, state);
								// wake up decoder thread
								if (block_ != NULL) repository_.Accept(block_);
                                utt_state_ = cur_state;
								if (cur_state == UTT_END) {
									finish_utt_ = true;
									ResetUtt();
									// break output an utterance result
									break;
								}
							}
						} else {
							block_ = new OnlineDecodableBlock(feat_out_ready_, utt_state_flags_[0]);
							// wake up decoder thread
							if (block_ != NULL) repository_.Accept(block_);
						}
                        n++;
					}
				}
			} else { // ipc forward
				socket_sample_->num_sample = frame_ready_;
				memcpy((char*)socket_sample_->sample, (char*)feat_in_.RowData(0),
						frame_ready_*feat_in_.NumCols()*sizeof(BaseFloat));
                socket_sample_->is_end = pos_state == FEAT_END ? true : false;

				// wake up decoder thread
				int ret = ipc_socket_->Send(sc_sample_buffer_, sc_buffer_size_, 0);
                if (ret != sc_buffer_size_) {
                    KALDI_ERR << "Send socket socket_sample: " << ret << " less than " << sc_buffer_size_
                                << " ipc forward server may offline.";
                    return -1;
                }
			}

		}
	}

	state_ = pos_state;
    return nbytes;
}

Result* OnlineFstDecoder::GetResult() {
	bool newutt = (state_ == FEAT_END || (vad_ != NULL && finish_utt_));
    if (newutt) {
    	while (!result_.isuttend)
    		sleep(0.02);
		// reset
		result_.isuttend = false;
		finish_utt_ = false;
    }

	std::vector<int32> word_ids;
	int size = result_.word_ids_.size();
	for (int i = cur_result_idx_; i < size; i++)
		word_ids.push_back(result_.word_ids_[i]);

    if (word_ids.size() > 0)
	    PrintPartialResult(word_ids, word_syms_, newutt);
    std::cout.flush();

    if (newutt)
       KALDI_LOG << "END: << " << result_.num_frames << " f, "<< 
       result_.num_frames/(100*60) << ":" << result_.num_frames%(100*60)/100 << "," << result_.num_frames%100*10;

	cur_result_idx_ = size;
	if (state_ == FEAT_END) 
		result_.isend = true;
	return &result_;
}

void OnlineFstDecoder::Abort() {
	repository_.Done();
	sleep(0.1);
}

}	// namespace kaldi

