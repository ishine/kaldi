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
#include "online0/speex-ogg-dec.h"

using namespace kaldi;

void *CreateIvectorExtractor(const char *cfg_path) {
	#ifdef __DEBUG
	    printf("create decoder, config path: %s.\n", cfg_path);
	#endif
	std::string cfg(cfg_path);
	OnlineIvectorExtractor *extractor = nullptr;

	extractor = new OnlineIvectorExtractor(cfg);
	if (nullptr == extractor) {
		fprintf(stderr, "create decoder failed, config path: %s\n", cfg_path);
		printf("%s\n",strerror(errno));
		return nullptr;
	}

	extractor->InitExtractor();
	return (void *)extractor;
}

int	IvectorExtractorFeedData(void *lp_extractor, const void *data, int nbytes,
    int cut_time, int type, int state) {
	OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
    if (state==FEAT_START)
    	extractor->Reset();

    int num_samples = 0, len = cut_time;
    float *audio = nullptr;
    char *pcm_audio = nullptr;
    const void *raw_audio = data;

    if (nbytes <= 0)
        return nbytes;

    if (0 == type) { // convert ogg speex to pcm
#ifdef HAVE_SPEEX
    	nbytes = SpeexOggDecoder((char*)data, nbytes, pcm_audio);
    	raw_audio = pcm_audio;
#endif
    }

    len = len < 0 ? 0: len;
    len = len*extractor->GetAudioFrequency()*2;
    // process
    if (0 == type || 1 == type) { // pcm data
        nbytes = len < nbytes && len > 0 ? len : nbytes;
    	num_samples = nbytes / sizeof(short);
    	audio = new float[num_samples];
    	for (int i = 0; i < num_samples; i++)
    		audio[i] = ((short *)raw_audio)[i];
    } else { // default raw feature(fbank, plp, mfcc...)
    	num_samples = nbytes / sizeof(float);
    	audio = new float[num_samples];
        if (num_samples > 0)
    	    memcpy(audio, data, num_samples*sizeof(float));
    }

    extractor->FeedData(audio, num_samples*sizeof(float), (kaldi::FeatState)state);

    delete [] audio;
    if (nullptr != pcm_audio)
    	delete pcm_audio;

    return nbytes;
}

int GetIvectorDim(void *lp_extractor) {
    OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
    return extractor->GetIvectorDim();
}

int GetCurrentIvector(void *lp_extractor, float *result, int type) {
	OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
    int dim = 0;
	Ivector *ivector = extractor->GetCurrentIvector(type);

    if (nullptr != ivector) {
	    dim = ivector->ivector_.Dim();
        if (dim > 0)
            memcpy(result, ivector->ivector_.Data(), dim*sizeof(float));
    }
	return dim;
}

float GetIvectorScore(void *lp_extractor, float *enroll, float *eval,
		int size, int enroll_num, int type) {
	OnlineIvectorExtractor *extractor = (OnlineIvectorExtractor *)lp_extractor;
	SubVector<BaseFloat> vec1(enroll, size);
	SubVector<BaseFloat> vec2(eval, size);
	return extractor->GetScore(vec1, enroll_num, vec2, type);
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


