// nnet3/attention-test.cc

// Copyright      2017  Hossein Hadian
//                2017  Johns Hopkins University (author: Daniel Povey)

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

#include "util/common-utils.h"
#include "nnet3/attention.h"

namespace kaldi {
namespace nnet3 {

void TestPositionEmbedding() {

  int32 sentence_len = 5, d_model = 5;
     
  CuMatrix<BaseFloat> position_embedding(sentence_len, d_model, kUndefined);
  position_embedding.PosEmbStandard(10000.0);

  Matrix<BaseFloat> position_embedding_gpu(position_embedding);
  Matrix<BaseFloat> position_embedding_cpu(sentence_len, d_model, kUndefined);



  for(int t=0;t<position_embedding_cpu.NumRows();t++){
	SubVector<BaseFloat> embedding(position_embedding_cpu, t);
    for(int i=0;i < d_model;i++){
      if(i%2){
        //odd
        embedding.Data()[i] = std::cos(t/std::pow(10000.0,2.0*i/d_model));
      }
      else{
        //even
        embedding.Data()[i] = std::sin(t/std::pow(10000.0,2.0*i/d_model));
      }
    }
  }
  printf("=============GPU Matrix===============\n");
  for (int i = 0; i < position_embedding.NumRows(); i++)
  {
	for (int j = 0; j < position_embedding.NumCols(); j++) {
	  printf("%f  ", position_embedding_gpu(i, j));
	}
	printf("\n");
  }

  printf("=============CPU Matrix===============\n");
  for (int i = 0; i < sentence_len; i++)
  {
	for (int j = 0; j < d_model; j++) {
	  printf("%f  ", position_embedding_cpu(i, j));
	}
	printf("\n");
  }
}

void UnitTestPE() {
  TestPositionEmbedding();
}

} // namespace nnet3
} // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  using namespace kaldi::nnet3::attention;
  for (int32 loop = 0; loop < 1; loop++) {
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SetDebugStrideMode(true);
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no"); // -1 means no GPU
    else
      CuDevice::Instantiate().SelectGpuId("optional"); // -2 .. automatic selection
#endif
	printf("======= Loop %d======\n", loop);
    for (int32 i = 0; i < 1; i++) {
	  printf("####### Test %d in loop %d\n", i, loop);
      UnitTestPE();
    }
  }
}
