// nnetbin/nnet-forward.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

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

#include <limits>

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;

map<string, size_t> BuildVocab(ifstream &vocab_file) {
    map<string, size_t> word_to_id;
    string word;
    int i = 0;

    while (vocab_file >> word) {
        word_to_id[word] = i;
        i++;
    }

    return word_to_id;
}

map<size_t, string> BuildReverseVocab(ifstream &vocab_file) {
    map<size_t, string> id_to_word;
    string word;
    size_t i = 0;

    while (getline(vocab_file, word)) {
        id_to_word[i] = word;
        i++;
    }

    return id_to_word;
}

vector<size_t> Transform(const string &words, const map<string, size_t> &vocab) {
    vector<size_t> ids;
    istringstream stream(words);
    string word;
    size_t id;

    while (stream >> word) {
        auto map_it = vocab.find(word);
        if (map_it == vocab.end()) {
            map_it = vocab.find("<unk>");
            if (map_it == vocab.end()) {
                std::cout << "Your Vocab Should include <unk>" << std::endl;
                exit(0);
            }
        }
        id = map_it->second;
        ids.push_back(id);
    }
    return ids;
}

void PrintVec(const vector<size_t> & ids) {
    for (int i = 0; i < ids.size(); ++i) {
        cout << ids[i] << " ";
    }
    cout << endl;
}

template<typename Real>
void IdsToMatrix(const std::vector<size_t>& ids, Matrix<Real>* m) {
  std::string s("[ ");
  int i;
  for (i = 0; i <ids.size()-1; ++i) {
      s += to_string(ids[i]) + " \n";
  }
  s += to_string(ids[i]) + " ]";
  std::cout << s << endl;

  std::istringstream is(s + "\n");
  m->Read(is, false);  // false for ascii
}

template<typename Real>
void ProbToId(const Matrix<Real> &m, vector<size_t> &ids) { 
    // m: T x C
    // ids: T
    for (int r = 0; r < m.NumRows(); ++r) {
        Real max = 0.0;
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

void AddPuncToTxt(const std::string &txt_line, 
                  const std::vector<size_t> &punc_ids, 
                  const std::map<size_t, std::string> &id_to_punc,
                  std::string &txt_line_with_punc) {
  istringstream stream(txt_line);
  std::string word, punc;
  size_t i = 0;
  while (stream >> word) {
    punc = id_to_punc.find(punc_ids[i])->second;
    if (punc == " ") {
      txt_line_with_punc += word + " ";
    } else {
      txt_line_with_punc += punc + " " + word + " ";
    }
    ++i;
  }
  punc = id_to_punc.find(punc_ids[i])->second;
  txt_line_with_punc += punc;
}


int main(int argc, char *argv[]) {
  try {
    const char *usage =
      "Perform forward pass through Neural Network.\n"
      "Usage: nnet-forward [options] <nnet1-in> <feature-rspecifier> <feature-wspecifier>\n"
      "e.g.: nnet-forward final.nnet ark:input.ark ark:output.ark\n";

    ParseOptions po(usage);

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax,
        "Removes the last component with Softmax, if found. The pre-softmax "
        "activations are the output of the network. Decoding them leads to "
        "the same lattices as if we had used 'log-posteriors'.");

    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform NN output by log()");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        vocab_file_path = po.GetArg(2),
        punc_vocab_file_path = po.GetArg(3),
        txt_file_path = po.GetArg(4);

    // Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    // Step 1. Load Model
    Nnet nnet;
    nnet.Read(model_filename);
    // disable dropout,
    nnet.SetDropoutRate(0.0);
    nnet.SetBatchNormMode("test");

    // Step 2. Load Vocab
    std::ifstream vocab_file(vocab_file_path);
    std::ifstream punc_vocab_file(punc_vocab_file_path);
    std::map<std::string, size_t> word_to_id = BuildVocab(vocab_file);
    std::map<size_t, std::string> id_to_punc = BuildReverseVocab(punc_vocab_file);
    // for (auto b = id_to_punc.begin(); b != id_to_punc.end(); b++) {
    //     cout << b->first << " " << b->second << endl;
    // }
    // for (auto b = word_to_id.begin(); b != word_to_id.end(); b++) {
    //     cout << b->first << " " << b->second << endl;
    // }

    kaldi::int64 tot_t = 0;

    CuMatrix<BaseFloat> feats, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    Timer time;
    int32 num_done = 0;

    std::ifstream txt_file(txt_file_path);
    std::string txt_line;
    // main loop,
    while (getline(txt_file, txt_line)) {
      // 2. words to ids
      std::vector<size_t> ids;
      ids = Transform(txt_line + " <END>", word_to_id);
      std::cout << txt_line << std::endl;
      PrintVec(ids);
      // 3. ids to Matrix
      Matrix<BaseFloat> mat;
      IdsToMatrix(ids, &mat);
      KALDI_LOG << mat;
      // push it to gpu,
      feats = mat;
      // 4. fwd-pass, nnet,
      nnet.Feedforward(feats, &nnet_out);
      // download from GPU,
      nnet_out_host = Matrix<BaseFloat>(nnet_out);
      KALDI_LOG << nnet_out_host;
      // 5. prob to punc id
      std::vector<size_t> predict_punc_ids;
      ProbToId(nnet_out_host, predict_punc_ids);
      PrintVec(predict_punc_ids);

      std::string txt_line_with_punc;
      AddPuncToTxt(txt_line, predict_punc_ids, id_to_punc, txt_line_with_punc);
      std::cout << txt_line_with_punc << std::endl;

      num_done++;
      tot_t += mat.NumRows();
    }

#if HAVE_CUDA == 1
    if (GetVerboseLevel() >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
