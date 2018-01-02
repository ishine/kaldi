// punc/punctuator.cc

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

#include "punc/punctuator.h"

namespace kaldi {
namespace punc {

Punctuator::Punctuator() {}

Punctuator::~Punctuator() { scws_free(scws_); }

/**
 * Load models, word list and punctuation list
 */
void Punctuator::Load(const std::string &models_filename_list,
                      const std::string &vocab_filename,
                      const std::string &punc_vocab_filename) {
  KALDI_ASSERT(models_filename_list.size() > 0);
  KALDI_ASSERT(vocab_filename.size() > 0);
  KALDI_ASSERT(vocab_filename.substr(vocab_filename.size() - 4) == ".txt" &&
               "the suffix of vocab filename must be .txt");
  KALDI_ASSERT(punc_vocab_filename.size() > 0);

  // Split model list
  std::vector<std::string> models_filename;
  SplitStringToVector(models_filename_list, ",", false, &models_filename);

  // 1. Load Model
  for (int32 i = 0; i < models_filename.size(); ++i) {
    KALDI_ASSERT(models_filename[i].size() > 0);
    nnet1::Nnet nnet;
    nnet.Read(models_filename[i]);
    // disable dropout and BN
    nnet.SetDropoutRate(0.0);
    nnet.SetBatchNormMode("test");
    models_.push_back(nnet);
  }
  // 2. Load word list
  word_to_id_ = BuildWordToId(vocab_filename);
  // 3. Load punctuation list
  id_to_punc_ = BuildIdToWord(punc_vocab_filename);
  // 4. set word segmentation object scws_
  if (!(scws_ = scws_new())) {
    KALDI_ERR << "ERROR: cann't init the scws!";
  }
  scws_set_charset(scws_, "utf8");
  scws_set_dict(scws_, vocab_filename.c_str(), SCWS_XDICT_TXT);
}

/**
 *
 */
void Punctuator::AddPunc(const std::string &in, const int32 &model_id,
                         std::string *out) {
  KALDI_ASSERT(model_id < models_.size());

  CuMatrix<BaseFloat> feats, nnet_out;
  Matrix<BaseFloat> nnet_out_host;
  // 0. word segmenation
  std::string in_seg = WordSegment(in);
  // 1. words to ids
  std::vector<size_t> ids;
  ids = WordsToIds(in_seg + " <END>");
  // 2. ids to Matrix
  Matrix<BaseFloat> mat;
  IdsToMatrix(ids, &mat);
  // push it to gpu,
  feats = mat;
  // 3. nnet forward pass
  models_[model_id].Feedforward(feats, &nnet_out);
  // download from GPU,
  nnet_out_host = Matrix<BaseFloat>(nnet_out);
  // 4. prob to punc id
  std::vector<size_t> predict_punc_ids;
  ProbsToIds(nnet_out_host, predict_punc_ids);
  // 5. punc id to punc txt
  AddPuncToTxt(in_seg, predict_punc_ids, out);
}

std::map<std::string, size_t> Punctuator::BuildWordToId(
    const std::string &vocab_filename) {
  std::ifstream vocab_file(vocab_filename);
  std::map<std::string, size_t> word_to_id;
  std::string word;
  size_t i = 2;
  // This must be same with training code
  word_to_id["<UNK>"] = 0;
  word_to_id["<END>"] = 1;
  while (vocab_file >> word) {
    word_to_id[word] = i;
    i++;
  }
  return word_to_id;
}

std::map<size_t, std::string> Punctuator::BuildIdToWord(
    const std::string &vocab_filename) {
  std::ifstream vocab_file(vocab_filename);
  std::map<size_t, std::string> id_to_word;
  std::string word;
  size_t i = 1;
  // This must be same with training code
  id_to_word[0] = std::string(" ");
  while (getline(vocab_file, word)) {
    id_to_word[i] = word;
    i++;
  }
  return id_to_word;
}

std::string Punctuator::WordSegment(const std::string &in) {
  std::string result;
  scws_res_t res, cur;

  scws_send_text(scws_, in.c_str(), in.size());
  while ((res = cur = scws_get_result(scws_))) {
    while (cur != NULL) {
      result += in.substr(cur->off, cur->len) + " ";
      cur = cur->next;
    }
    scws_free_result(res);
  }
  return result;
}

std::vector<size_t> Punctuator::WordsToIds(const std::string &words) {
  std::vector<size_t> ids;
  std::istringstream stream(words);
  std::string word;
  size_t id;

  if (word_to_id_.find("<UNK>") == word_to_id_.end()) {
    KALDI_ERR << "Check BuildWordToId(), word_to_id_ should include <UNK>";
  }
  while (stream >> word) {
    auto map_it = word_to_id_.find(word);
    if (map_it == word_to_id_.end()) {
      map_it = word_to_id_.find("<UNK>");
    }
    id = map_it->second;
    ids.push_back(id);
  }
  return ids;
}

void Punctuator::IdsToMatrix(const std::vector<size_t> &ids,
                             Matrix<BaseFloat> *m) {
  std::string s("[ ");
  int i;
  for (i = 0; i < ids.size() - 1; ++i) {
    s += std::to_string(ids[i]) + " \n";
  }
  s += std::to_string(ids[i]) + " ]";
  // std::cout << s << endl;

  std::istringstream is(s + "\n");
  m->Read(is, false);  // false for ascii
}

void Punctuator::ProbsToIds(const Matrix<BaseFloat> &m,
                            std::vector<size_t> &ids) {
  // m: T x C
  // ids: T
  for (int r = 0; r < m.NumRows(); ++r) {
    BaseFloat max = 0.0;
    int32 max_id = -1;
    for (int c = 0; c < m.NumCols(); ++c) {
      if (m(r, c) > max) {
        max = m(r, c);
        max_id = c;
      }
    }
    ids.push_back(max_id);
  }
}

void Punctuator::AddPuncToTxt(const std::string &txt_line,
                              const std::vector<size_t> &punc_ids,
                              std::string *txt_line_with_punc) {
  std::istringstream stream(txt_line);
  std::string word, punc;
  size_t i = 0;
  while (stream >> word) {
    punc = id_to_punc_.find(punc_ids[i])->second;
    if (punc == " ") {
      *txt_line_with_punc += word + " ";
    } else {
      *txt_line_with_punc += punc + " " + word + " ";
    }
    ++i;
  }
  punc = id_to_punc_.find(punc_ids[i])->second;
  *txt_line_with_punc += punc;
}

}  // namespace punc
}  // namespace kaldi
