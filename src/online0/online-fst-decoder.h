// online0/Online-fst-decoder.h
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

#ifndef ONLINE0_ONLINE_FST_DECODER_H_
#define ONLINE0_ONLINE_FST_DECODER_H_

#include "fstext/fstext-lib.h"
#include "decoder/decodable-matrix.h"
#include "util/kaldi-semaphore.h"
#include "util/kaldi-mutex.h"
#include "util/kaldi-thread.h"

#include "online0/online-fst-decoder-cfg.h"
#include "online0/online-faster-decoder.h"
#include "online0/online-nnet-feature-pipeline.h"
#include "online0/online-nnet-forward.h"
#include "online0/online-nnet-decoding.h"
#include "online0/online-nnet-lattice-decoding.h"
#include "online0/online-am-vad.h"

namespace kaldi {

class OnlineFstDecoder {

public:
	OnlineFstDecoder(OnlineFstDecoderCfg *cfg);
	virtual ~OnlineFstDecoder() { Destory(); }

	// initialize decoder
	void InitDecoder();

	// feed wave data to decoder
	int FeedData(void *data, int nbytes, FeatState state);

	// feed full utterance wave data to decoder
	// using for offline decoder
	int FeedData(void *data, int nbytes);

	// get online decoder result
	Result* GetResult();

	// Reset decoder
	void Reset();

	// abort decoder
	void Abort();

private:
	void ResetUtt();
	void Destory();
	const static int VECTOR_INC_STEP = 16000*10;

	// read only decoder resources
	OnlineFstDecoderCfg *decoder_cfg_;

	OnlineFasterDecoderOptions *fast_decoder_opts_;
	OnlineLatticeFasterDecoderOptions *lat_decoder_opts_;
	OnlineNnetForwardOptions *forward_opts_;
	OnlineNnetFeaturePipelineOptions *feature_opts_;
	OnlineNnetDecodingOptions *decoding_opts_;
	OnlineAmVadOptions *am_vad_opts_;

	TransitionModel *trans_model_;
	fst::Fst<fst::StdArc> *decode_fst_;
	fst::SymbolTable *word_syms_;

	// likelihood
	OnlineDecodableBlock *block_;
	OnlineDecodableInterface *decodable_;

	// decoder
	Repository repository_;
	OnlineFasterDecoder *fast_decoder_;
	OnlineNnetDecodingClass *fast_decoding_;
	MultiThreader<OnlineNnetDecodingClass> *fast_decoder_thread_;
	OnlineLatticeFasterDecoder *lat_decoder_;
	OnlineNnetLatticeDecodingClass *lat_decoding_;
	MultiThreader<OnlineNnetLatticeDecodingClass> *lat_decoder_thread_;
	OnlineAmVad *am_vad_;

	// feature pipeline
	OnlineNnetFeaturePipeline *feature_pipeline_;
	// forward
	OnlineNnetForward *forward_;
	// ipc forward socket
	UnixDomainSocket *ipc_socket_;

	// decode result
	Int32VectorWriter *words_writer_;
	Int32VectorWriter *alignment_writer_;

	Result result_;
	FeatState state_;
	UttState utt_state_;

	// ipc socket input sample
	SocketSample *socket_sample_;
	char *sc_sample_buffer_;
	int sc_buffer_size_;
	// decoding buffer
	Matrix<BaseFloat> feat_in_, feat_out_, feat_out_ready_, blank_post_;
	// wav buffer
	Vector<BaseFloat> wav_buffer_;
	std::vector<int> utt_state_flags_;
	std::vector<int> valid_input_frames_;
	// online feed
	int len_, sample_offset_, frame_offset_, frame_ready_;
	int in_skip_, out_skip_, skip_frames_, chunk_length_, cur_result_idx_;
	bool finish_utt_;
};

}	 // namespace kaldi

#endif /* ONLINE0_ONLINE_FST_DECODER_H_ */
