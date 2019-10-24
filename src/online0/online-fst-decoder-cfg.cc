// online0/Online-fst-decoder-cfg.cc
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
#include "online0/online-fst-decoder-cfg.h"

namespace kaldi {

OnlineFstDecoderCfg::OnlineFstDecoderCfg(std::string cfg) :
		fast_decoder_opts_(NULL), lat_decoder_opts_(NULL), forward_opts_(NULL),
		feature_opts_(NULL), decoding_opts_(NULL), vad_opts_(NULL),
		trans_model_(NULL), decode_fst_(NULL), word_syms_(NULL) {

	// main config
	decoding_opts_ = new OnlineNnetDecodingOptions;
	ReadConfigFromFile(cfg, decoding_opts_);

	// decoder feature forward config
	fast_decoder_opts_ = new OnlineFasterDecoderOptions;
	lat_decoder_opts_ = new OnlineLatticeFasterDecoderOptions;
	forward_opts_ = new OnlineNnetForwardOptions;
	feature_opts_ = new OnlineNnetFeaturePipelineOptions(decoding_opts_->feature_cfg);
    vad_opts_ = new OnlineVadOptions;

	if (decoding_opts_->use_lat)
		ReadConfigFromFile(decoding_opts_->decoder_cfg, lat_decoder_opts_);
	else
		ReadConfigFromFile(decoding_opts_->decoder_cfg, fast_decoder_opts_);

	if (decoding_opts_->forward_cfg != "")
		ReadConfigFromFile(decoding_opts_->forward_cfg, forward_opts_);

	if (decoding_opts_->vad_cfg != "")
		ReadConfigFromFile(decoding_opts_->vad_cfg, vad_opts_);
    
    // load decode resources
    Initialize();
}

void OnlineFstDecoderCfg::Destory() {
	if (decoding_opts_ != NULL) {
		delete fast_decoder_opts_;	fast_decoder_opts_ = NULL;
		delete lat_decoder_opts_;	lat_decoder_opts_ = NULL;
		delete forward_opts_;	forward_opts_ = NULL;
		delete feature_opts_;	feature_opts_ = NULL;
		delete decoding_opts_;	decoding_opts_ = NULL;
		delete vad_opts_;		vad_opts_ = NULL;
	}

	if (decode_fst_ != NULL) {
        delete trans_model_; trans_model_ = NULL;
		delete decode_fst_;	decode_fst_ = NULL;
		delete word_syms_;	word_syms_ = NULL;
	}
}

void OnlineFstDecoderCfg::Initialize() {
	// trainsition model
	bool binary;
	if (decoding_opts_->model_rspecifier != "") {
		Input ki(decoding_opts_->model_rspecifier, &binary);
        trans_model_ = new TransitionModel;
		trans_model_->Read(ki.Stream(), binary);
	}

	// HCLG fst graph
	decode_fst_ = fst::ReadFstKaldiGeneric(decoding_opts_->fst_rspecifier);
	if (!(word_syms_ = fst::SymbolTable::ReadText(decoding_opts_->word_syms_filename)))
		KALDI_ERR << "Could not read symbol table from file " << decoding_opts_->word_syms_filename;
}

}	// namespace kaldi




