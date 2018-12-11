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

#ifdef __cplusplus
extern "C" {
#endif

	void *CreateIvectorExtractor(const char *cfg_path);

	/// lp_extractor: ivector extractor handle
	/// data: speex ogg, pcm or feature data in byte
	/// nbyte: data size in byte
	//  cut_time: effective when state=2 and type=0,1. cut_time > 0 extract the begining of cutoff time audio,
	//            otherwise extract the whole audio.
	/// type:  ogg(0), pcm(1), raw feature(2, e.g.fbank, plp, mfcc)
	///	state: start(0), append(1), end(2)
	int IvectorExtractorFeedData(void *lp_extractor, const void *data, int nbytes,
									float cut_time = 6, int type = 1, int state = 2);

	/// return ivector size
	int	GetIvectorDim(void *lp_extractor);

	/// ivector: ivector for current utterance
	/// type: plda(2)
	int GetCurrentIvector(void *lp_extractor, float *ivector, int type = 2);

	/// compute two ivector similarity score
	/// enroll: enroll or register ivector
	/// eval: test or evaluate ivector
	/// size: ivector size
	/// type: plda(2)
	float GetXvectorScore(void *lp_extractor, float *enroll, float *eval, int size,
			int enroll_num = 1, int type = 2);

	int GetEnrollSpeakerIvector(void *lp_extractor, float *spk_ivector, float *ivectors,
		    int ivec_dim, int num_ivec, int type = 2);

	/// in general, reset ivector extractor state when start a new utterance.
	void ResetIvectorExtractor(void *lp_extractor);

	/// destroy ivctor extractor handle
	void DestroyIvectorExtractor(void *lp_extractor);

#ifdef __cplusplus
}
#endif

#endif /* KALDI_ONLINE0_IVECTOR_EXPORT_H_ */
