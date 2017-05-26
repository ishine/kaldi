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
#include "nnet0/nnet-nnet.h"

namespace kaldi {

struct LstmLmHistroy {
    LstmLmHistroy(std::vector<int> &rdim, std::vector<int> &cdim,
                    MatrixResizeType resize_type = kSetZero) {
        his_recurrent.resize(rdim.size());
        for (int i = 0; i < rdim.size(); i++)
            his_recurrent[i].Resize(rdim[i], resize_type);
        his_cell.resize(cdim.size());
        for (int i = 0; i < cdim.size(); i++)
            his_cell[i].Resize(cdim[i], resize_type);
    }

	std::vector<Vector<BaseFloat> > his_recurrent; //  each hidden lstm layer recurrent history
	std::vector<Vector<BaseFloat> > his_cell; //  each hidden lstm layer cell history
};

struct KaldiNNlmWrapperOpts {
  std::string unk_symbol;
  std::string sos_symbol;
  std::string eos_symbol;
  std::string class_boundary;
  std::string class_constant;
  int32 num_stream;

  KaldiNNlmWrapperOpts() : unk_symbol("<unk>"), sos_symbol("<s>"), eos_symbol("</s>"),
		  class_boundary(""), class_constant(""), num_stream(1) {}

  void Register(OptionsItf *opts) {
    opts->Register("unk-symbol", &unk_symbol, "Symbol for out-of-vocabulary "
                   "words in neural network language model.");
    opts->Register("sos-symbol", &sos_symbol, "Start of sentence symbol in "
                   "neural network language model.");
    opts->Register("eos-symbol", &eos_symbol, "End of sentence symbol in "
                   "neural network language model.");
    opts->Register("class-boundary", &class_boundary, "The fist index of each class(and final class class) in class based language model");
    opts->Register("class-constant", &class_constant, "The constant zt<sum(exp(yi))> of each class(and final class class) in class based language model");
    opts->Register("num-stream", &num_stream, "Number of utterance process in parallel in "
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

  int32 GetEos() const { return eos_; }
  int32 GetSos() const { return sos_; }
  int32 GetUnk() const { return unk_; }

  std::vector<int> &GetRDim() { return recurrent_dim_; }
  std::vector<int> &GetCDim() { return cell_dim_; }

  void GetLogProbParallel(const std::vector<int> &curt_words,
  										 const std::vector<LstmLmHistroy*> &context_in,
  										 std::vector<LstmLmHistroy*> &context_out,
  										 std::vector<BaseFloat> &logprob);

  BaseFloat GetLogProb(int32 curt_words, LstmLmHistroy* context_in, LstmLmHistroy* context_out);

  inline int32 GetWordId(int32 wid) { return label_to_lmwordid_[wid];}
  inline int32 GetWordId(std::string word) { return word_to_lmwordid_[word];}

 private:
  nnet0::Nnet nnlm_;
  std::vector<int> word2class_;
  std::vector<int> class_boundary_;
  std::vector<BaseFloat> class_constant_;
  std::vector<std::string> label_to_word_;
  std::vector<int32> label_to_lmwordid_;
  std::unordered_map<std::string, int32> word_to_lmwordid_;
  std::vector<int> recurrent_dim_;
  std::vector<int> cell_dim_;
  int32 unk_;
  int32 sos_;
  int32 eos_;

  int num_stream_;
  Matrix<BaseFloat> out_linearity_;
  Vector<BaseFloat> out_bias_;
  Matrix<BaseFloat> class_linearity_;
  Vector<BaseFloat> class_bias_;

  Vector<BaseFloat> in_words_;
  Matrix<BaseFloat> in_words_mat_;
  CuMatrix<BaseFloat> words_;
  CuMatrix<BaseFloat> hidden_out_;

  std::vector<Matrix<BaseFloat> > his_recurrent_; // current hidden lstm layers recurrent history
  std::vector<Matrix<BaseFloat> > his_cell_;	// current hidden lstm layers cell history
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
  virtual ~NNlmDeterministicFst();

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
  std::vector<LstmLmHistroy* > state_to_context_;
};

}  // namespace kaldi

#endif  // KALDI_LM_KALDI_NNLM_H_
