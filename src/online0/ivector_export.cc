// online0/ivector_export.cc

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


#include "online0/ivector_export.h"
#include "online0/online-ivector-extractor.h"

using namespace kaldi;

void *CreateIvectorExtractor(const char *cfg_path) {
	#ifdef __DEBUG
	    printf("create decoder, config path: %s.\n", cfg_path);
	#endif
	std::string cfg(cfg_path);
	OnlineIvectorExtractor *extractor = new OnlineIvectorExtractor(cfg);
	extractor->InitExtractor();
	return (void *)extractor;
}

int	IvectorExtractorFeedData(void *lp_extractor, void *data, int nbytes, int state) {
	OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
    if (state==FEAT_START)
    	extractor->Reset();

    int num_samples = nbytes / sizeof(short);
    float *audio = new float[num_samples];
    for (int i = 0; i < num_samples; i++)
        audio[i] = ((short *)data)[i];
    extractor->FeedData(audio, num_samples*sizeof(float), (kaldi::FeatState)state);
    delete [] audio;
    return nbytes;
}

int GetIvectorDim(void *lp_extractor) {
    OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
    return extractor->GetIvectorDim();
}

int GetCurrentIvector(void *lp_extractor, float *result, int type) {
	OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
	Ivector *ivector = extractor->GetCurrentIvector(type);
	int dim = ivector->ivector_.Dim();
	memcpy(result, ivector->ivector_.Data(), dim*sizeof(float));
	return dim;
}

float GetIvectorScore(void *lp_extractor, float *ivec1, float *ivec2, int size, int type) {
	OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
	SubVector<BaseFloat> vec1(ivec1, size);
	SubVector<BaseFloat> vec2(ivec2, size);
	return extractor->GetScore(vec1, vec2, type);
}

int GetEnrollSpeakerIvector(void *lp_extractor, float *spk_ivec, float *ivecs,
		int ivec_dim, int num_ivec, int type) {
	OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
	std::vector<Vector<BaseFloat> > ivectors(num_ivec);
	Vector<BaseFloat> spk_ivector;

	for (int i = 0; i < num_ivec; i++) {
		SubVector<BaseFloat> ivec(ivecs+i*ivec_dim, ivec_dim);
		ivectors[i] = ivec;
	}
	extractor->GetEnrollSpeakerIvector(ivectors, spk_ivector, type);
	int dim = spk_ivector.Dim();
	memcpy(spk_ivec, spk_ivector.Data(), dim*sizeof(float));
	return dim;
}

void ResetIvectorExtractor(void *lp_extractor) {
	OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
	extractor->Reset();
}

void DestroyIvectorExtractor(void *lp_extractor) {
#ifdef __DEBUG
    printf("destroy ivector extractor instance.\n");
#endif
    if (lp_extractor!=nullptr) {
        delete (OnlineIvectorExtractor *)lp_extractor;
        lp_extractor = nullptr;
    }

}


