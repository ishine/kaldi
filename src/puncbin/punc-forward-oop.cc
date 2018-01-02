// puncbin/punc-forward-oop.cc

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

#include "base/kaldi-common.h"
#include "punc/punctuator.h"


int main(int argc, char *argv[]) {
  try {
    const char *usage =
      "Perform forward pass through Neural Network.\n"
      "Usage: punc-forward-oop [options] <nnet1-list> <vocab> <punc-vocab> <txt-file>\n"
      "e.g.: punc-forward-oop a.nnet,b.nnet,c.nnet vocab.txt punc_vocab nopunc.txt\n";

    using namespace std;
    using namespace kaldi;
    using namespace kaldi::punc;
    typedef kaldi::int32 int32;

    ParseOptions po(usage);

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string models_filename_list = po.GetArg(1),
                vocab_filename = po.GetArg(2),
                punc_vocab_filename = po.GetArg(3),
                txt_file_path = po.GetArg(4);

    // Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    // 1. Load Model
    Punctuator punctuator;
    punctuator.Load(models_filename_list, vocab_filename, punc_vocab_filename);

    std::ifstream txt_file(txt_file_path);
    std::string txt_line;
    int32 num_done = 0;
    // For each line in txt file
    while (getline(txt_file, txt_line)) {
      std::string txt_line_with_punc;
      int32 model_id = 0;

      // 2. Add Punctuation
      punctuator.AddPunc(txt_line, model_id, &txt_line_with_punc);

      std::cout << txt_line_with_punc << std::endl;
      num_done++;
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
