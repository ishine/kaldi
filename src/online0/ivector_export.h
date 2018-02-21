// online0/ivector_export.h

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

#ifndef KALDI_ONLINE0_IVECTOR_EXPORT_H_
#define KALDI_ONLINE0_IVECTOR_EXPORT_H_

#include "online0/online-ivector-extractor.h"

#ifdef __cplusplus
extern "C" {
#endif

void *CreateIvectorExtractor(const char *cfg_path);
int	FeedData(void *lp_extractor, void *data, int nbytes, int state);
int GetCurrentIvector(void *lp_extractor, float *result, int type = 1);
float GetScore(void *lp_extractor, float *ivec1, float *ivec2, int size);
int GetEnrollSpeakerIvector(void *lp_extractor, float *spk_ivector, float *ivectors,
		int ivec_dim, int num_ivec, int type = 1);
void Reset(void *lp_extractor);
void DestroyIvectorExtractor(void *lp_extractor);

#ifdef __cplusplus
}
#endif



#endif /* KALDI_ONLINE0_IVECTOR_EXPORT_H_ */
