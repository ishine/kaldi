// lm/kaldi-rnntlm.cc

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

#include <utility>

#include "lm/kaldi-rnntlm.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {

KaldiRNNTlmWrapper::KaldiRNNTlmWrapper(
    const KaldiRNNTlmWrapperOpts &opts,
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

	for (int i = 0; i < lm_word_symbols->NumSymbols(); i++)
	  word_to_lmwordid_[lm_word_symbols->Find(i)] = i;

	auto it = word_to_lmwordid_.find(opts.unk_symbol);
	if (it == word_to_lmwordid_.end())
	  KALDI_WARN << "Could not find symbol " << opts.unk_symbol
				  << " for out-of-vocabulary " << lm_word_symbol_table_rxfilename;
	it = word_to_lmwordid_.find(opts.sos_symbol);
	if (it == word_to_lmwordid_.end())
	  KALDI_ERR << "Could not find start of sentence symbol " << opts.sos_symbol
				  << " in " << lm_word_symbol_table_rxfilename;
	sos_ = it->second;
	it = word_to_lmwordid_.find(opts.eos_symbol);
	if (it == word_to_lmwordid_.end())
	  KALDI_ERR << "Could not find end of sentence symbol " << opts.eos_symbol
				  << " in " << lm_word_symbol_table_rxfilename;
	eos_ = it->second;

	//map label id to language model word id
	unk_ = word_to_lmwordid_[opts.unk_symbol];
	label_to_lmwordid_.resize(label_to_word_.size());
	for (int i = 0; i < label_to_word_.size(); i++)
	{
	  auto it = word_to_lmwordid_.find(label_to_word_[i]);
	  if (it != word_to_lmwordid_.end())
		  label_to_lmwordid_[i] = it->second;
	  else
		  label_to_lmwordid_[i] = unk_;
	}

	in_words_.Resize(num_stream_, kUndefined);
	in_words_mat_.Resize(num_stream_, 1, kUndefined);
	words_.Resize(num_stream_, kUndefined);
	hidden_out_.Resize(num_stream_, nnlm_.OutputDim(), kUndefined);
}

void KaldiRNNTlmWrapper::Forward(int words_in, LstmLmHistroy& context_in,
		  	   Vector<BaseFloat> &nnet_out, LstmLmHistroy& context_out) {
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

	// save current words history
	nnlm_.SaveContext(his_recurrent_, his_cell_);
	for (i = 0; i < num_layers; i++) {
		context_out.his_recurrent[i] = his_recurrent_[i].Row(0);
		context_out.his_cell[i] = his_cell_[i].Row(0);
	}

	nnet_out = hidden_out_.Row(0);
}

void KaldiRNNTlmWrapper::GetLogProbParallel(const std::vector<int> &curt_words,
										 const std::vector<LstmLmHistroy*> &context_in,
										 std::vector<LstmLmHistroy*> &context_out,
										 std::vector<BaseFloat> &logprob) {
	// get current words log probility (CPU done)
	LstmLmHistroy *his;
	int i, j, wid, cid;
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

BaseFloat KaldiRNNTlmWrapper::GetLogProb(int32 curt_word,
		LstmLmHistroy *context_in, LstmLmHistroy *context_out) {
	// get current words log probility (CPU done)
	BaseFloat logprob = 0.0;
	int i;
	SubVector<BaseFloat> linear_vec(out_linearity_.Row(curt_word));
	Vector<BaseFloat> &hidden_out_vec = context_in->his_recurrent.back();

	BaseFloat prob = VecVec(hidden_out_vec, linear_vec) + out_bias_(curt_word);

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
