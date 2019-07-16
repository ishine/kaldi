// lm/kaldi-lstmlm.cc

// Copyright 2018-2019   Alibaba Inc (author: Wei Deng)

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

#include "kaldi-lstmlm.h"

#include <utility>

#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {

KaldiLstmlmWrapper::KaldiLstmlmWrapper(
    const KaldiLstmlmWrapperOpts &opts,
    const std::string &word_symbol_table_rxfilename,
	const std::string &lm_word_symbol_table_rxfilename,
    const std::string &nnlm_rxfilename) {

	// Reads symbol table.
	fst::SymbolTable *word_symbols = NULL;
	if (!(word_symbols = fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
		KALDI_ERR << "Could not read symbol table from file " << word_symbol_table_rxfilename;
	}

	// load lstm lm
	nnlm_.Read(nnlm_rxfilename);
	KALDI_ASSERT(word_symbols->NumSymbols() == nnlm_.OutputDim());

	// get rc information
	nnlm_.GetHiddenLstmLayerRCInfo(recurrent_dim_, cell_dim_);
	// split net to lstm hidden part and output part
	nnlm_.SplitLstmLm(out_linearity_, out_bias_, opts.remove_head);

	// nstream utterance parallelization
	num_stream_ = opts.num_stream;
	std::vector<int> new_utt_flags(num_stream_, 0);
	nnlm_.ResetLstmStreams(new_utt_flags);

	// init rc context buffer
	his_recurrent_.resize(recurrent_dim_.size());
	for (int i = 0; i < recurrent_dim_.size(); i++)
	his_recurrent_[i].Resize(num_stream_, recurrent_dim_[i], kUndefined);
	his_cell_.resize(cell_dim_.size());
	for (int i = 0; i < cell_dim_.size(); i++)
	his_cell_[i].Resize(num_stream_, cell_dim_[i], kUndefined);

	/// symbols
	label_to_word_.resize(word_symbols->NumSymbols());
	for (int32 i = 0; i < word_symbols->NumSymbols(); i++) {
		label_to_word_[i] = word_symbols->Find(i);
		if (label_to_word_[i] == "") {
		  KALDI_ERR << "Could not find word for integer " << i << "in the word "
			  << "symbol table, mismatched symbol table or you have discoutinuous "
			  << "integers in your symbol table?";
		}
	}

	// Reads lstm lm symbol table.
	fst::SymbolTable *lm_word_symbols = word_symbols;
	if (lm_word_symbol_table_rxfilename != "") {
	    if (!(lm_word_symbols = fst::SymbolTable::ReadText(lm_word_symbol_table_rxfilename))) {
	        KALDI_ERR << "Could not read symbol table from file " << lm_word_symbol_table_rxfilename;
	    }
	}

	unk_ = eos_ = sos_ = 0;
	in_words_.Resize(num_stream_, kUndefined);
	in_words_mat_.Resize(num_stream_, 1, kUndefined);
	words_.Resize(num_stream_, kUndefined);
	hidden_out_.Resize(num_stream_, nnlm_.OutputDim(), kUndefined);
}

void KaldiLstmlmWrapper::Forward(int words_in, LstmlmHistroy& context_in,
		  	   Vector<BaseFloat> *nnet_out, LstmlmHistroy *context_out) {
	// next produce and save current word rc information (recommend GPU)
	// restore history
	int i, num_layers = context_in.his_recurrent.size();
	for (i = 0; i < num_layers; i++) {
		his_recurrent_[i].Row(0).CopyFromVec(context_in.his_recurrent[i]);
		his_cell_[i].Row(0).CopyFromVec(context_in.his_cell[i]);
	}
	nnlm_.RestoreContext(his_recurrent_, his_cell_);

	in_words_(0) = words_in;
	in_words_mat_.CopyColFromVec(in_words_, 0);
	words_.CopyFromMat(in_words_mat_);

	// forward propagate
	nnlm_.Propagate(words_, &hidden_out_);

	if (context_out != NULL) {
		// save current words history
		nnlm_.SaveContext(his_recurrent_, his_cell_);
		for (i = 0; i < num_layers; i++) {
			context_out->his_recurrent[i] = his_recurrent_[i].Row(0);
			context_out->his_cell[i] = his_cell_[i].Row(0);
		}
	}

	if (nnet_out != NULL) {
		nnet_out->Resize(hidden_out_.NumCols(), kUndefined);
		hidden_out_.Row(0).CopyToVec(nnet_out);
	}
}

void KaldiLstmlmWrapper::ForwardMseq(const std::vector<int> &in_words,
				const std::vector<LstmlmHistroy*> &context_in,
				std::vector<Vector<BaseFloat>*> &nnet_out,
				std::vector<LstmlmHistroy*> &context_out) {
	int i, j;
	int num_layers = context_in[0]->his_recurrent.size();
	int cur_stream = in_words.size();

	if (cur_stream != num_stream_) {
		num_stream_ = cur_stream;
		KALDI_LOG << "Reset lstm lm with " << num_stream_ << " streams.";
		in_words_.Resize(num_stream_, kUndefined);
		in_words_mat_.Resize(num_stream_, 1, kUndefined);
		words_.Resize(num_stream_, kUndefined);
		hidden_out_.Resize(num_stream_, nnlm_.OutputDim(), kUndefined);
		std::vector<int> new_utt_flags(num_stream_, 0);
		nnlm_.ResetLstmStreams(new_utt_flags);
	}

	// restore history
	for (i = 0; i < num_layers; i++) {
		for (j = 0; j < num_stream_; j++) {
		his_recurrent_[i].Row(j).CopyFromVec(context_in[j]->his_recurrent[i]);
		his_cell_[i].Row(j).CopyFromVec(context_in[j]->his_cell[i]);
		}
	}
	nnlm_.RestoreContext(his_recurrent_, his_cell_);

	for (i = 0; i < num_stream_; i++)
		in_words_(i) = in_words[i];
	in_words_mat_.CopyColFromVec(in_words_, 0);
	words_.CopyFromMat(in_words_mat_);

	// forward propagate
	nnlm_.Propagate(words_, &hidden_out_);

	// save current words history
	nnlm_.SaveContext(his_recurrent_, his_cell_);
	for (i = 0; i < num_layers; i++) {
		for (j = 0; j < num_stream_; j++) {
			if (context_out[j] != NULL) {
				context_out[j]->his_recurrent[i] = his_recurrent_[i].Row(j);
				context_out[j]->his_cell[i] = his_cell_[i].Row(j);
			}
		}
	}

	for (j = 0; j < num_stream_; j++) {
		if (nnet_out[j] != NULL) {
			nnet_out[j]->Resize(hidden_out_.NumCols(), kUndefined);
			hidden_out_.Row(j).CopyToVec(nnet_out[j]);
		}
	}
}

void KaldiLstmlmWrapper::GetLogProbParallel(const std::vector<int> &curt_words,
										 const std::vector<LstmlmHistroy*> &context_in,
										 std::vector<LstmlmHistroy*> &context_out,
										 std::vector<BaseFloat> &logprob) {
	// get current words log probility (CPU done)
	LstmlmHistroy *his;
	int i, j, wid;
	logprob.resize(num_stream_);
	for (i = 0; i < num_stream_; i++) {
		wid = curt_words[i];
		in_words_(i) = wid;
		his = context_in[i];

		SubVector<BaseFloat> linear_vec(out_linearity_.Row(wid));
		Vector<BaseFloat> &hidden_out_vec = his->his_recurrent.back();
		logprob[i] = VecVec(hidden_out_vec, linear_vec) + out_bias_(wid);
	}

	// next produce and save current word rc information (recommend GPU)
	// restore history
	int num_layers = context_in[0]->his_recurrent.size();
	for (i = 0; i < num_layers; i++) {
		for (j = 0; j < num_stream_; j++) {
			his_recurrent_[i].Row(j).CopyFromVec(context_in[j]->his_recurrent[i]);
			his_cell_[i].Row(j).CopyFromVec(context_in[j]->his_cell[i]);
		}
	}

	nnlm_.RestoreContext(his_recurrent_, his_cell_);

	in_words_mat_.CopyColFromVec(in_words_, 0);
	words_.CopyFromMat(in_words_mat_);

    // forward propagate
	nnlm_.Propagate(words_, &hidden_out_);

	// save current words history
	nnlm_.SaveContext(his_recurrent_, his_cell_);
	for (i = 0; i < num_layers; i++) {
		for (j = 0; j < num_stream_; j++) {
			context_out[j]->his_recurrent[i] = his_recurrent_[i].Row(j);
			context_out[j]->his_cell[i] = his_cell_[i].Row(j);
		}
	}
}

BaseFloat KaldiLstmlmWrapper::GetLogProb(int32 curt_word,
		LstmlmHistroy *context_in, LstmlmHistroy *context_out) {
	// get current words log probility (CPU done)
	BaseFloat logprob = 0.0;
	int i;
	SubVector<BaseFloat> linear_vec(out_linearity_.Row(curt_word));
	Vector<BaseFloat> &hidden_out_vec = context_in->his_recurrent.back();

	logprob = VecVec(hidden_out_vec, linear_vec) + out_bias_(curt_word);

	if (context_out == NULL)
        return logprob;

	// next produce and save current word rc information (recommend GPU)
	// restore history
	int num_layers = context_in->his_recurrent.size();
	for (i = 0; i < num_layers; i++) {
		his_recurrent_[i].Row(0).CopyFromVec(context_in->his_recurrent[i]);
		his_cell_[i].Row(0).CopyFromVec(context_in->his_cell[i]);
	}
	nnlm_.RestoreContext(his_recurrent_, his_cell_);

	in_words_(0) = curt_word;
	in_words_mat_.CopyColFromVec(in_words_, 0);
	words_.CopyFromMat(in_words_mat_);

	// forward propagate
	nnlm_.Propagate(words_, &hidden_out_);

	// save current words history
	nnlm_.SaveContext(his_recurrent_, his_cell_);
	for (i = 0; i < num_layers; i++) {
		context_out->his_recurrent[i] = his_recurrent_[i].Row(0);
		context_out->his_cell[i] = his_cell_[i].Row(0);
	}
	return logprob;
}



}  // namespace kaldi
