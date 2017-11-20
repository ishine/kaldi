// nnetbin/nnet-forward.cc

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
#include "puncbin/punc-utils.h"
using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;


int main(int argc, char *argv[]) {
  try {
    const char *usage =
      "Perform forward pass through Neural Network.\n"
      "Usage: punc-forward [options] <nnet1-in> <vocab> <punc-vocab> <txt-file>\n"
      "e.g.: punc-forward final.nnet vocab punc_vocab nopunc.txt\n";

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
