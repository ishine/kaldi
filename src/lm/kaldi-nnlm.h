// lm/kaldi-nnlm.h

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

#ifndef KALDI_LM_KALDI_NNLM_H_
#define KALDI_LM_KALDI_NNLM_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "util/common-utils.h"

namespace kaldi {

struct KaldiNNlmWrapperOpts {
  std::string unk_symbol;
  std::string eos_symbol;

  KaldiNNlmWrapperOpts() : unk_symbol("<unk>"), eos_symbol("</s>") {}

  void Register(OptionsItf *opts) {
    opts->Register("unk-symbol", &unk_symbol, "Symbol for out-of-vocabulary "
                   "words in neural network language model.");
    opts->Register("eos-symbol", &eos_symbol, "End of sentence symbol in "
                   "neural network language model.");
  }
};

class KaldiNNlmWrapper {
 public:
  KaldiNNlmWrapper(const KaldiNNlmWrapperOpts &opts,
                    const std::string &unk_prob_rspecifier,
                    const std::string &word_symbol_table_rxfilename,
					const std::string &lm_word_symbol_table_rxfilename,
                    const std::string &nnlm_rxfilename);

  int32 GetLMNumHiddenLayer() const { return nnlm_.GetLMNumHiddenLayer(); }

  int32 GetEos() const { return eos_; }

  BaseFloat GetLogProb(int32 word, const std::vector<int32> &wseq,
		  	  	  	  const std::vector<CuMatrixBase<BaseFloat> > &context_in,
					  std::vector<CuMatrix<BaseFloat> > *context_out);

  inline int32 GetWordId(int32 wid) { return word_to_lmwordid_[wid];}

 private:
  kaldi::nnet0::Nnet nnlm_;
  std::vector<std::string> label_to_word_;
  std::unordered_map<std::string, int32> word_to_lmwordid_;
  std::unordered_map<int32, int32> label_to_lmwordid_;
  std::unordered_map<std::string, float> unk_penalty;
  int32 eos_;
  std::string unk_sym;

  KALDI_DISALLOW_COPY_AND_ASSIGN(KaldiNNlmWrapper);
};

class NNlmDeterministicFst
    : public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  // Does not take ownership.
  NNlmDeterministicFst(int32 max_ngram_order, KaldiNNlmWrapper *nnlm);

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  typedef unordered_map<std::vector<Label>,
                        StateId, VectorHasher<Label> > MapType;
  StateId start_state_;
  MapType wseq_to_state_;
  std::vector<std::vector<Label> > state_to_wseq_;

  KaldiNNlmWrapper *nnlm_;
  int32 max_ngram_order_;
  std::vector<std::vector<CuMatrix<BaseFloat> > > state_to_context_;
};

}  // namespace kaldi

#endif  // KALDI_LM_KALDI_NNLM_H_
