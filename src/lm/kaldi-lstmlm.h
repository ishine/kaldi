// lm/kaldi-lstmlm.h

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

#ifndef KALDI_LM_KALDI_NNLM_H_
#define KALDI_LM_KALDI_NNLM_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "util/common-utils.h"
#include "nnet0/nnet-nnet.h"

namespace kaldi {

struct LstmlmHistroy {
	LstmlmHistroy(std::vector<int> &rdim, std::vector<int> &cdim,
                    MatrixResizeType resize_type = kSetZero) {
        his_recurrent.resize(rdim.size());
        for (int i = 0; i < rdim.size(); i++)
            his_recurrent[i].Resize(rdim[i], resize_type);
        his_cell.resize(cdim.size());
        for (int i = 0; i < cdim.size(); i++)
            his_cell[i].Resize(cdim[i], resize_type);
    }
    LstmlmHistroy() {}

    void SetZero() {
        for (int i = 0; i < his_recurrent.size(); i++)
            his_recurrent[i].SetZero();
        for (int i = 0; i < his_cell.size(); i++)
            his_cell[i].SetZero();
    }

	std::vector<Vector<BaseFloat> > his_recurrent; //  each hidden lstm layer recurrent history
	std::vector<Vector<BaseFloat> > his_cell; //  each hidden lstm layer cell history
};

struct KaldiLstmlmWrapperOpts {
  std::string unk_symbol;
  std::string sos_symbol;
  std::string eos_symbol;
  std::string class_boundary;
  std::string class_constant;
  int num_stream;
  bool remove_head;
  bool use_classlm;

  KaldiLstmlmWrapperOpts() : unk_symbol("<unk>"), sos_symbol("<s>"), eos_symbol("</s>"),
		  num_stream(1), remove_head(false), use_classlm(false) {}

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
    opts->Register("use_classlm", &use_classlm, "Whether is class neural lm.");

  }
};

class KaldiLstmlmWrapper {
 public:
  KaldiLstmlmWrapper(const KaldiLstmlmWrapperOpts &opts,
                    const std::string &word_symbol_table_rxfilename,
					const std::string &lm_word_symbol_table_rxfilename,
                    const std::string &nnlm_rxfilename);

  int GetEos() const { return eos_; }
  int GetSos() const { return sos_; }
  int GetUnk() const { return unk_; }

  std::vector<int> &GetRDim() { return recurrent_dim_; }
  std::vector<int> &GetCDim() { return cell_dim_; }
  int GetVocabSize() { return nnlm_.OutputDim();}

  void GetLogProbParallel(const std::vector<int> &curt_words,
  										 const std::vector<LstmlmHistroy*> &context_in,
  										 std::vector<LstmlmHistroy*> &context_out,
  										 std::vector<BaseFloat> &logprob);

  BaseFloat GetLogProb(int curt_words, LstmlmHistroy* context_in,
		  	  	  	  	  	  	  	  	  LstmlmHistroy* context_out);

  void Forward(int words_in, LstmlmHistroy& context_in,
		  	   Vector<BaseFloat> *nnet_out, LstmlmHistroy *context_out);

  void ForwardMseq(const std::vector<int> &in_words,
		  	  	  	  	  	  	  	  	  const std::vector<LstmlmHistroy*> &context_in,
										  std::vector<Vector<BaseFloat>*> &nnet_out,
										  std::vector<LstmlmHistroy*> &context_out);

  void ForwardMseqClass(const std::vector<int> &in_words,
		  	  	  	  	  const std::vector<LstmlmHistroy*> &context_in,
						  std::vector<LstmlmHistroy*> &context_out,
						  std::vector<std::vector<int> > &out_words,
						  std::vector<std::vector<BaseFloat> > &out_words_logprob);

  inline int GetWordId(int wid) { return label_to_lmwordid_[wid];}
  inline int GetWordId(std::string word) { return word_to_lmwordid_[word];}

  void ResetStreams(int cur_stream);

 private:
  nnet0::Nnet nnlm_;
  std::vector<int> word2class_;
  std::vector<int> class_boundary_;
  std::vector<BaseFloat> class_constant_;
  std::vector<std::string> symid_to_word_;
  std::vector<int> label_to_lmwordid_;
  std::unordered_map<std::string, int> word_to_lmwordid_;
  std::vector<int> recurrent_dim_;
  std::vector<int> cell_dim_;
  int unk_;
  int sos_;
  int eos_;

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
  KALDI_DISALLOW_COPY_AND_ASSIGN(KaldiLstmlmWrapper);
};


}  // namespace kaldi

#endif  // KALDI_LM_KALDI_LSTMLM_H_
