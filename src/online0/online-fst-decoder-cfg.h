// online0/Online-fst-decoder-cfg.h
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

#ifndef ONLINE0_ONLINE_FST_DECODER_CFG_H_
#define ONLINE0_ONLINE_FST_DECODER_CFG_H_

#include "fstext/fstext-lib.h"
#include "decoder/decodable-matrix.h"
#include "online-faster-decoder.h"
#include "util/kaldi-semaphore.h"
#include "util/kaldi-mutex.h"
#include "util/kaldi-thread.h"

#include "online0/online-nnet-feature-pipeline.h"
#include "online0/online-nnet-forward.h"
#include "online0/online-nnet-decoding.h"
#include "online0/online-nnet-lattice-decoding.h"
#include "online0/online-am-vad.h"

namespace kaldi {

class OnlineFstDecoderCfg {

public:
	OnlineFasterDecoderOptions *fast_decoder_opts_;
	OnlineLatticeFasterDecoderOptions *lat_decoder_opts_;
	OnlineNnetForwardOptions *forward_opts_;
	OnlineNnetFeaturePipelineOptions *feature_opts_;
	OnlineNnetDecodingOptions *decoding_opts_;
	OnlineAmVadOptions *am_vad_opts_;

	TransitionModel *trans_model_;
	fst::Fst<fst::StdArc> *decode_fst_;
	fst::SymbolTable *word_syms_;

	OnlineFstDecoderCfg(std::string cfg);
	virtual ~OnlineFstDecoderCfg() { Destory(); }

private:
	void Destory();
	// initialize read only resource
	void Initialize();
};

}	 // namespace kaldi

#endif /* ONLINE0_ONLINE_FST_DECODER_CFG_H_ */
