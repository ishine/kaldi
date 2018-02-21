// online0/online-ivector-extractor.h
// Copyright 2017-2018   Shanghai Jiao Tong University (author: Wei Deng)

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

#ifndef ONLINE0_ONLINE_IVECTOR_EXTRACTOR_H_
#define ONLINE0_ONLINE_IVECTOR_EXTRACTOR_H_

#include "online0/online-ivector-feature.h"
#include "online0/online-nnet-feature-pipeline.h"

namespace kaldi {

typedef struct Ivector_ {
    bool    isvalid_;
	Vector<float> ivector_;
	double	ubm_loglike_perframe_;
	double	objf_impr_perframe_;
	double	num_frames_;
	std::string utt;
	void clear() {
        isvalid_ = false;
        ubm_loglike_perframe_ = 0;
        objf_impr_perframe_ = 0;
        num_frames_ = 0;
		utt = "";
	}
}Ivector;

typedef enum {
	FEAT_START,
	FEAT_APPEND,
	FEAT_END,
}FeatState;

class OnlineIvectorExtractor {

public:
	OnlineIvectorExtractor(std::string cfg);
	virtual ~OnlineIvectorExtractor() { Destory(); }

	// initialize decoder
	void InitExtractor();

	// feed wave data to extractor
	int FeedData(void *data, int nbytes, FeatState state);

	// get current frame ivector,
	// type: 0, raw ivector; 1, lda transformed ivector
	Ivector* GetCurrentIvector(int type = 1);

	// compute ivector score
	BaseFloat GetScore(const VectorBase<BaseFloat> &ivec1, const VectorBase<BaseFloat> &ivec2);

	// compute enroll ivector for a speaker
	void GetEnrollSpeakerIvector(const std::vector<Vector<BaseFloat> > &ivectors,
			Vector<BaseFloat> &spk_ivector, int type = 1);

	// Reset Extractor
	void Reset();

	int GetIvectorDim() { return ivector_feature_->Dim(); }

private:
	void Destory();

	/// feature pipeline config
	OnlineNnetFeaturePipelineOptions *feature_opts_;
	OnlineStreamIvectorExtractionConfig *ivector_config_;
	OnlineStreamIvectorExtractionInfo *ivector_info_;

	// feature pipeline
	OnlineNnetFeaturePipeline *base_feature_pipeline_;
	OnlineStreamIvectorFeature *ivector_feature_;
	Ivector ivector;
};

}	 // namespace kaldi

#endif /* ONLINE0_ONLINE_IVECTOR_EXTRACTOR_H_ */
