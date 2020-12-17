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

	if (opts.use_classlm) {
		// class boundary
		if (opts.class_boundary == "")
			KALDI_ERR<< "The lm class boundary file '" << opts.class_boundary << "' is empty.";
		Input in;
		Vector<BaseFloat> classinfo;
		in.OpenTextMode(opts.class_boundary);
		classinfo.Read(in.Stream(), false);
		in.Close();
		class_boundary_.resize(classinfo.Dim());
		for (int i = 0; i < classinfo.Dim(); i++)
		  class_boundary_[i] = classinfo(i);

		// log(zt) class constant
		if (opts.class_constant == "")
			KALDI_ERR<< "The lm class constant file '" << opts.class_constant << "' is empty.";
		Vector<BaseFloat> constantinfo;
		in.OpenTextMode(opts.class_constant);
		constantinfo.Read(in.Stream(), false);
		in.Close();
		class_constant_.resize(constantinfo.Dim());
		for (int i = 0; i < constantinfo.Dim(); i++)
		class_constant_[i] = constantinfo(i);

		// word id to class id
		word2class_.resize(class_boundary_.back());
		int j = 0;
		for (int i = 0; i < class_boundary_.back(); i++) {
		  if (i >= class_boundary_[j] && i < class_boundary_[j+1])
			  word2class_[i] = j;
		  else
			  word2class_[i] = ++j;
		}
	}

	// load lstm lm
	nnlm_.Read(nnlm_rxfilename);

	// get rc information
	nnlm_.GetHiddenLstmLayerRCInfo(recurrent_dim_, cell_dim_);

	if (opts.use_classlm) {
		// split net to lstm hidden part and output part
		nnlm_.SplitLstmLm(out_linearity_, out_bias_,
				  class_linearity_, class_bias_, class_boundary_.size()-1);
	} else {
		// split net to lstm hidden part and output part
		nnlm_.SplitLstmLm(out_linearity_, out_bias_, opts.remove_head);
	}

	// nstream utterance parallelization
	num_stream_ = opts.num_stream;
	ResetStreams(num_stream_);

	// Reads symbol table.
	fst::SymbolTable *word_symbols = NULL;
	if (!(word_symbols = fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
		KALDI_ERR << "Could not read symbol table from file " << word_symbol_table_rxfilename;
	}
	symid_to_word_.resize(word_symbols->NumSymbols());
	for (int32 i = 0; i < word_symbols->NumSymbols(); i++) {
		symid_to_word_[i] = word_symbols->Find(i);
		if (symid_to_word_[i] == "") {
		  KALDI_ERR << "Could not find word for integer " << i << "in the word "
			  << "symbol table, mismatched symbol table or you have discoutinuous "
			  << "integers in your symbol table?";
		}
	}

	if (opts.use_classlm) {
		// Reads lstm lm symbol table.
		fst::SymbolTable *lm_word_symbols = NULL;
		if (!(lm_word_symbols = fst::SymbolTable::ReadText(lm_word_symbol_table_rxfilename))) {
			KALDI_ERR << "Could not read symbol table from file " << lm_word_symbol_table_rxfilename;
		}

		for (int i = 0; i < lm_word_symbols->NumSymbols(); i++)
			word_to_lmwordid_[lm_word_symbols->Find(i)] = i;

		auto it = word_to_lmwordid_.find(opts.unk_symbol);
		if (it == word_to_lmwordid_.end())
		  KALDI_WARN << "Could not find symbol " << opts.unk_symbol
					  << " for out-of-vocabulary " << lm_word_symbol_table_rxfilename;

		it = word_to_lmwordid_.find(opts.sos_symbol);
		if (it == word_to_lmwordid_.end()) {
			KALDI_ERR << "Could not find start of sentence symbol " << opts.sos_symbol
					  << " in " << lm_word_symbol_table_rxfilename;
		}
		sos_ = it->second;
		it = word_to_lmwordid_.find(opts.eos_symbol);
		if (it == word_to_lmwordid_.end()) {
			KALDI_ERR << "Could not find end of sentence symbol " << opts.eos_symbol
						  << " in " << lm_word_symbol_table_rxfilename;
		}
		eos_ = it->second;

		//map label id to language model word id
		unk_ = word_to_lmwordid_[opts.unk_symbol];
		label_to_lmwordid_.resize(symid_to_word_.size());
		for (int i = 0; i < symid_to_word_.size(); i++) {
			auto it = word_to_lmwordid_.find(symid_to_word_[i]);
			if (it != word_to_lmwordid_.end())
			  label_to_lmwordid_[i] = it->second;
			else
			  label_to_lmwordid_[i] = unk_;
		}
	}
}


void KaldiLstmlmWrapper::ResetStreams(int cur_stream) {
	if (cur_stream == num_stream_)
		return;

	num_stream_ = cur_stream;
	KALDI_LOG << "Reset lstm lm with " << num_stream_ << " streams.";

    // init rc context buffer
    his_recurrent_.resize(recurrent_dim_.size());
    for (int i = 0; i < recurrent_dim_.size(); i++)
        his_recurrent_[i].Resize(num_stream_, recurrent_dim_[i], kUndefined);
    his_cell_.resize(cell_dim_.size());
    for (int i = 0; i < cell_dim_.size(); i++)
        his_cell_[i].Resize(num_stream_, cell_dim_[i], kUndefined);

	in_words_.Resize(num_stream_, kUndefined);
	in_words_mat_.Resize(num_stream_, 1, kUndefined);
	words_.Resize(num_stream_, kUndefined);
	hidden_out_.Resize(num_stream_, nnlm_.OutputDim(), kUndefined);
	std::vector<int> new_utt_flags(num_stream_, 0);
	nnlm_.ResetLstmStreams(new_utt_flags);
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

	ResetStreams(cur_stream);

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

void KaldiLstmlmWrapper::ForwardMseqClass(const std::vector<int> &in_words,
				                        const std::vector<LstmlmHistroy*> &context_in,
				                        std::vector<LstmlmHistroy*> &context_out,
				                        std::vector<std::vector<int> > &out_words,
				                        std::vector<std::vector<BaseFloat> > &out_words_logprob) {
	int num_layers = context_in[0]->his_recurrent.size();
	int cur_stream = in_words.size();

	ResetStreams(cur_stream);

	// restore history
	for (int i = 0; i < num_layers; i++) {
		for (int j = 0; j < num_stream_; j++) {
			his_recurrent_[i].Row(j).CopyFromVec(context_in[j]->his_recurrent[i]);
			his_cell_[i].Row(j).CopyFromVec(context_in[j]->his_cell[i]);
		}
	}
	nnlm_.RestoreContext(his_recurrent_, his_cell_);

	for (int i = 0; i < num_stream_; i++)
		in_words_(i) = in_words[i];
	in_words_mat_.CopyColFromVec(in_words_, 0);
	words_.CopyFromMat(in_words_mat_);

	// forward propagate
	nnlm_.Propagate(words_, &hidden_out_);

	// save current words history
	nnlm_.SaveContext(his_recurrent_, his_cell_);
	for (int i = 0; i < num_layers; i++) {
		for (int j = 0; j < num_stream_; j++) {
			if (context_out[j] != NULL) {
				context_out[j]->his_recurrent[i] = his_recurrent_[i].Row(j);
				context_out[j]->his_cell[i] = his_cell_[i].Row(j);
			}
		}
	}

	// get word logprob
	int wid, cid, size;
    BaseFloat prob, classprob, total_prob;
	for (int i = 0; i < num_stream_; i++) {
		size = out_words[i].size();
		if (size <= 0)
			continue;

		Vector<BaseFloat> &hidden_out_vec = context_out[i]->his_recurrent.back();
		for (int j = 0; j < size; j++) {
			wid = out_words[i][j];
			cid = word2class_[wid];
			SubVector<BaseFloat> linear_vec(out_linearity_.Row(wid));
			SubVector<BaseFloat> class_linear_vec(class_linearity_.Row(cid));
			prob = VecVec(hidden_out_vec, linear_vec) + out_bias_(wid);
			classprob = VecVec(hidden_out_vec, class_linear_vec) + class_bias_(cid);
			total_prob = prob+classprob-class_constant_[cid]-class_constant_.back();
            out_words_logprob[i].push_back(prob+classprob);
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
