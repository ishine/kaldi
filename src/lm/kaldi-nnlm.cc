// lm/kaldi-nnlm.cc

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

#include <utility>

#include "lm/kaldi-nnlm.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

namespace kaldi {

KaldiNNlmWrapper::KaldiNNlmWrapper(
    const KaldiNNlmWrapperOpts &opts,
	const std::string &unk_prob_rspecifier,
    const std::string &word_symbol_table_rxfilename,
	const std::string &lm_word_symbol_table_rxfilename,
    const std::string &nnlm_rxfilename) {

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
	  KALDI_ERR<< "The lm class boundary file '" << opts.class_boundary << "' is empty.";
  Vector<BaseFloat> constantinfo;
  in.OpenTextMode(opts.class_constant);
  constantinfo.Read(in.Stream(), false);
  in.Close();
  class_constant_.resize(constantinfo.Dim());
  for (int i = 0; i < constantinfo.Dim(); i++)
	  class_constant_[i] = constantinfo(i);

  // map wordid to class
  word2class_.resize(class_boundary_.back());
  int j = 0;
  for (int i = 0; i < class_boundary_.back(); i++) {
	  if (i>=class_boundary_[j] && i<class_boundary_[j+1])
		  word2class_[i] = j;
	  else
		  word2class_[i] = ++j;
  }

  // load lstm lm 
  nnlm_.Read(nnlm_rxfilename);

  // get rc information
  nnlm_.GetHiddenLstmLayerRCInfo(recurrent_dim_, cell_dim_);

  // split net to lstm hidden part and output part
  nnlm_.SplitLstmLm(out_linearity_, out_bias_,
		  class_linearity_, class_bias_, class_boundary_.size()-1);

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

  // Reads symbol table.
  fst::SymbolTable *word_symbols = NULL;
  if (!(word_symbols =
        fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
    KALDI_ERR << "Could not read symbol table from file "
        << word_symbol_table_rxfilename;
  }
  label_to_word_.resize(word_symbols->NumSymbols());
  for (int32 i = 0; i < label_to_word_.size() - 1; ++i) {
    label_to_word_[i] = word_symbols->Find(i);
    if (label_to_word_[i] == "") {
      KALDI_ERR << "Could not find word for integer " << i << "in the word "
          << "symbol table, mismatched symbol table or you have discoutinuous "
          << "integers in your symbol table?";
    }
  }

  fst::SymbolTable *lm_word_symbols = NULL;
  if (!(lm_word_symbols =
       fst::SymbolTable::ReadText(lm_word_symbol_table_rxfilename))) {
	KALDI_ERR << "Could not read symbol table from file "
          << lm_word_symbol_table_rxfilename;
  }

  for (int i = 0; i < word_symbols->NumSymbols(); ++i)
	  word_to_lmwordid_[lm_word_symbols->Find(i)] = i;

  auto it = word_to_lmwordid_.find(opts.unk_symbol);
  if (it == word_to_lmwordid_.end())
	  KALDI_WARN << "Could not find symbol " << opts.unk_symbol
	  	  	  	  << " for out-of-vocabulary " << lm_word_symbol_table_rxfilename;
  it = word_to_lmwordid_.find(opts.eos_symbol);
  if (it == word_to_lmwordid_.end())
  	  KALDI_ERR << "Could not find end of sentence symbol " << opts.eos_symbol
  	  	  	  	  << " in " << lm_word_symbol_table_rxfilename;
  eos_ = it->second;

  //map label id to language model word id
  unk_ = word_to_lmwordid_[opts.unk_symbol];
  for (int i = 0; i < label_to_word_.size(); ++i)
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

void KaldiNNlmWrapper::GetLogProbParallel(const std::vector<int> &curt_words,
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
		cid = word2class_[wid];
		his = context_in[i];

		SubVector<BaseFloat> linear_vec(out_linearity_.Row(wid));
		SubVector<BaseFloat> class_linear_vec(class_linearity_.Row(cid));
		Vector<BaseFloat> &hidden_out_vec = his->his_recurrent.back();

		BaseFloat prob = VecVec(hidden_out_vec, linear_vec) + out_bias_(wid);
		BaseFloat classprob = VecVec(hidden_out_vec, class_linear_vec) + class_bias_(cid);
		logprob[i] = prob+classprob-class_constant_[cid]-class_constant_.back();
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

BaseFloat KaldiNNlmWrapper::GetLogProb(int32 curt_word,
		LstmLmHistroy *context_in, LstmLmHistroy *context_out) {

	// get current words log probility (CPU done)
	BaseFloat logprob = 0.0;
	int i, cid = word2class_[curt_word];
	SubVector<BaseFloat> linear_vec(out_linearity_.Row(curt_word));
	SubVector<BaseFloat> class_linear_vec(class_linearity_.Row(cid));
	Vector<BaseFloat> &hidden_out_vec = context_in->his_recurrent.back();

	BaseFloat prob = VecVec(hidden_out_vec, linear_vec) + out_bias_(curt_word);
	BaseFloat classprob = VecVec(hidden_out_vec, class_linear_vec) + class_bias_(cid);
	logprob = prob + classprob - class_constant_[cid] - class_constant_.back();

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



NNlmDeterministicFst::NNlmDeterministicFst(int32 max_ngram_order,
                                             KaldiNNlmWrapper *nnlm) {
  KALDI_ASSERT(nnlm != NULL);
  max_ngram_order_ = max_ngram_order;
  nnlm_ = nnlm;

  // Uses empty history for <s>.
  std::vector<Label> bos;
  LstmLmHistroy* bos_context = new LstmLmHistroy(nnlm_->GetRDim(), nnlm_->GetCDim(), kSetZero);
  state_to_wseq_.push_back(bos);
  state_to_context_.push_back(bos_context);
  wseq_to_state_[bos] = 0;
  start_state_ = 0;
}

NNlmDeterministicFst::~NNlmDeterministicFst() {
	for (int i = 0; i < state_to_context_.size(); i++) {
		delete state_to_context_[i];
		state_to_context_[i] = NULL;
	}
}

fst::StdArc::Weight NNlmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  BaseFloat logprob = nnlm_->GetLogProb(nnlm_->GetEos(),
                                         state_to_context_[s], NULL);
  return Weight(-logprob);
}

bool NNlmDeterministicFst::GetArc(StateId s, Label ilabel, fst::StdArc *oarc) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  // map to nnlm wordlist
  int32 curt_word = nnlm_->GetWordId(ilabel);
  std::vector<Label> wseq = state_to_wseq_[s];

  LstmLmHistroy *new_context = new LstmLmHistroy(nnlm_->GetRDim(), nnlm_->GetCDim(), kUndefined);

  BaseFloat logprob = nnlm_->GetLogProb(curt_word, state_to_context_[s], new_context);

  wseq.push_back(ilabel);
  if (max_ngram_order_ > 0) {
    while (wseq.size() >= max_ngram_order_) {
      // History state has at most <max_ngram_order_> - 1 words in the state.
      wseq.erase(wseq.begin(), wseq.begin() + 1);
    }
  }

  std::pair<const std::vector<Label>, StateId> wseq_state_pair(
      wseq, static_cast<Label>(state_to_wseq_.size()));

  // Attemps to insert the current <lseq_state_pair>. If the pair already exists
  // then it returns false.
  typedef MapType::iterator IterType;
  std::pair<IterType, bool> result = wseq_to_state_.insert(wseq_state_pair);

  // If the pair was just inserted, then also add it to <state_to_wseq_> and
  // <state_to_context_>.
  if (result.second == true) {
    state_to_wseq_.push_back(wseq);
    state_to_context_.push_back(new_context);
  }

  // Creates the arc.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Weight(-logprob);

  return true;
}

}  // namespace kaldi
