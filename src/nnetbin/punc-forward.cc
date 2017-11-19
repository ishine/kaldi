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
#include "nnet/nnet-loss.h"
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
void ReadCuMatrixFromString(const std::string& s, CuMatrix<Real>* m) {
  std::istringstream is(s + "\n");
  m->Read(is, false);  // false for ascii
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

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        vocab_file_path = po.GetArg(2),
        txt_file_path = po.GetArg(3);
        // feature_rspecifier = po.GetArg(2),
        // feature_wspecifier = po.GetArg(3);

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
    ifstream vocab_file(vocab_file_path);
    auto word_to_id = BuildVocab(vocab_file);
    for (auto b = word_to_id.begin(); b != word_to_id.end(); b++) {
        cout << b->first << " " << b->second << endl;
    }

    kaldi::int64 tot_t = 0;

    CuMatrix<BaseFloat> feats, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    Timer time;
    int32 num_done = 0;

    ifstream txt_file(txt_file_path);
    string line;
    // main loop,
    while (getline(txt_file, line)) {
      // 2. words to ids
      vector<size_t> ids;
      ids = Transform(line, word_to_id);
      cout << line;
      PrintVec(ids);

/*
      // read
      // Matrix<BaseFloat> mat = feature_reader.Value();

      // push it to gpu,
      feats = mat;

      // fwd-pass, nnet,
      nnet.Feedforward(feats, &nnet_out);

      // download from GPU,
      nnet_out_host = Matrix<BaseFloat>(nnet_out);

      // write,
      // feature_writer.Write(feature_reader.Key(), nnet_out_host);
      */

      num_done++;
      // tot_t += mat.NumRows();
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
