// punc/punctuator.h

// Copyright 2017 NPU (Author: Kaituo Xu)

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

#ifndef KALDI_PUNC_PUNCTUATOR_H_
#define KALDI_PUNC_PUNCTUATOR_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "nnet/nnet-nnet.h"
#include "scws/scws.h"
#include "util/common-utils.h"
#include "util/kaldi-io.h"

namespace kaldi {
namespace punc {

class Punctuator {
 public:
  Punctuator();
  ~Punctuator();

 public:
  /// Load models, word list, punctuation list and init scws
  void Load(const std::string &models_filename_list,
            const std::string &vocab_filename,
            const std::string &punc_vocab_filename);

  /// Add punctuation
  void AddPunc(const std::string &in, const int32 &model_id, std::string *out);

 private:
  std::map<std::string, size_t> BuildWordToId(
      const std::string &vocab_filename);

  std::map<size_t, std::string> BuildIdToWord(
      const std::string &vocab_filename);

  std::string WordSegment(const std::string &in);

  std::vector<size_t> WordsToIds(const std::string &words,
                                 const std::map<std::string, size_t> &vocab);

  void IdsToMatrix(const std::vector<size_t> &ids, Matrix<BaseFloat> *m);

  void ProbsToIds(const Matrix<BaseFloat> &m, std::vector<size_t> &ids);

  void AddPuncToTxt(const std::string &txt_line,
                    const std::vector<size_t> &punc_ids,
                    std::string *txt_line_with_punc);

 private:
  /// Vector which contains all the models
  std::vector<nnet1::Nnet> models_;

  /// Word segmentation using scws
  scws_t scws_;

  /// Word list, map word to id
  std::map<std::string, size_t> word_to_id_;

  /// Punctuation list, map punctuation id to punctuation
  std::map<size_t, std::string> id_to_punc_;
};

}  // namespace punc
}  // namespace kaldi

#endif  // KALDI_PUNC_PUNCTUATOR_H_
