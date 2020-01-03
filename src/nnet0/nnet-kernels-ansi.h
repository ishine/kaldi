// nnet0/nnet-kernels-ansi.h

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_NNET_NNET_KERNELS_ANSI_H_
#define KALDI_NNET_NNET_KERNELS_ANSI_H_
#include "cudamatrix/cu-matrixdim.h" 

#undef ATOMIC_CONST
#define ATOMIC_CONST 32

#if HAVE_CUDA == 1
extern "C" {
  // "C" version of the BaseFloat typedef-- this saves us having to write
  // multiple versions of these kernels.
#if (KALDI_DOUBLEPRECISION != 0)
  typedef double  BaseFloat;
#else
  typedef float   BaseFloat;
#endif

struct Transition {
    float weight;
    int label;
    int state;
};

struct IntPair {
    int first;
    int second;
};

	void cuda_compute_alpha(dim3 Gr, dim3 Bl,
					   BaseFloat *alpha,
					   const BaseFloat *logits,
					   const int batch_size,
					   int T,
					   const int alpha_size,
					   int logits_size,
					   int *input_lengths,
					   BaseFloat *loglikelihood,
					   const BaseFloat *start_weight,
					   const BaseFloat *end_weight,
					   const IntPair *transition_index_alpha,
					   const Transition *transition_alpha,
					   bool batch_first = true);

	void cuda_compute_beta_and_grad(dim3 Gr, dim3 Bl,
					   BaseFloat *beta,
					   const BaseFloat * const alpha,
					   const BaseFloat * const logits,
					   const BaseFloat * const alpha_lld,
					   BaseFloat *grad_storage,
					   BaseFloat *grad_net,
					   const int batch_size,
					   const int T,
					   const int beta_size,
					   const int logits_size,
					   const int * const input_lengths,
					   BaseFloat * loglikelihood,
					   const BaseFloat *start_weight,
					   const BaseFloat *end_weight,
					   const IntPair *transition_index_beta,
					   const Transition *transition_beta,
					   bool batch_first = true);

} // extern "C"

#endif  // HAVE_CUDA


#endif  // KALDI_NNET_NNET_KERNELS_ANSI_H_
