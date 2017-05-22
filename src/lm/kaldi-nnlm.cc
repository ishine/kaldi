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
    const std::string &word_symbol_table_rxfilename,
	const std::string &lm_word_symbol_table_rxfilename,
    const std::string &nnlm_rxfilename,
	const std::string &classboundary_file) {

  this->unk_sym = opts.unk_symbol;

  nnlm_.Read(nnlm_rxfilename);
  nnlm_.SetClassBoundary(classboundary_file);

  // Reads symbol table.
  fst::SymbolTable *word_symbols = NULL;
  if (!(word_symbols =
        fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
    KALDI_ERR << "Could not read symbol table from file "
        << word_symbol_table_rxfilename;
  }
  label_to_word_.resize(word_symbols->NumSymbols() + 1);
  for (int32 i = 0; i < label_to_word_.size() - 1; ++i) {
    label_to_word_[i] = word_symbols->Find(i);
    if (label_to_word_[i] == "") {
      KALDI_ERR << "Could not find word for integer " << i << "in the word "
          << "symbol table, mismatched symbol table or you have discoutinuous "
          << "integers in your symbol table?";
    }
  }
  label_to_word_[label_to_word_.size() - 1] = opts.eos_symbol;
  eos_ = label_to_word_.size() - 1;

  fst::SymbolTable *lm_word_symbols = NULL;
  if (!(lm_word_symbols =
       fst::SymbolTable::ReadText(lm_word_symbol_table_rxfilename))) {
	KALDI_ERR << "Could not read symbol table from file "
          << lm_word_symbol_table_rxfilename;
  }

  for (int32 i = 0; i < word_symbols->NumSymbols(); ++i)
	  word_to_lmwordid_[lm_word_symbols->Find(i)] = i;

  //map label id to language model word id
  int32 unkid = word_to_lmwordid_[opts.unk_symbol];
  for (int32 i = 0; i < label_to_word_.size(); ++i)
  {
	  auto it = word_to_lmwordid_.find(label_to_word_[i]);
	  if (it != word_to_lmwordid_.end())
		  label_to_lmwordid_[i] = it->second;
	  else
		  label_to_lmwordid_[i] = unkid;
  }

}

BaseFloat KaldiNNlmWrapper::GetLogProb(
    int32 word, const std::vector<int32> &wseq,
    const std::vector<CuMatrixBase<BaseFloat> > &context_in,
    std::vector<CuMatrix<BaseFloat> > *context_out) {

  return nnlm_.ComputeConditionalLogprob(word, wseq,
                                          context_in, *context_out);
}

NNlmDeterministicFst::NNlmDeterministicFst(int32 max_ngram_order,
                                             KaldiNNlmWrapper *nnlm) {
  KALDI_ASSERT(nnlm != NULL);
  max_ngram_order_ = max_ngram_order;
  nnlm_ = nnlm;

  // Uses empty history for <s>.
  std::vector<Label> bos;
  //std::vector<float> bos_context(nnlm->GetHiddenLayerSize(), 1.0);
  state_to_wseq_.push_back(bos);
  state_to_context_.push_back(bos_context);
  wseq_to_state_[bos] = 0;
  start_state_ = 0;
}

fst::StdArc::Weight NNlmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  BaseFloat logprob = nnlm_->GetLogProb(nnlm_->GetEos(), wseq,
                                         state_to_context_[s], NULL);
  return Weight(-logprob);
}

bool NNlmDeterministicFst::GetArc(StateId s, Label ilabel, fst::StdArc *oarc) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  // map to nnlm wordlist
  ilabel = nnlm_->GetWordId(ilabel);

  std::vector<Label> wseq = state_to_wseq_[s];
  std::vector<CuMatrix<BaseFloat> > new_context(nnlm_->GetLMNumHiddenLayer());
  BaseFloat logprob = nnlm_->GetLogProb(ilabel, wseq,
                                         state_to_context_[s], &new_context);

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

void KaldiNNlmWrapper::SetUnkPenalty(const std::string &filename)
{
	  SequentialBaseFloatReader unk_reader(filename);
	  for (; !unk_reader.Done(); unk_reader.Next()) {
	    std::string key = unk_reader.Key();
	    float prob = unk_reader.Value();
	    unk_reader.FreeCurrent();
	    unk_penalty[key] = log(prob);
	  }
}

}  // namespace kaldi
