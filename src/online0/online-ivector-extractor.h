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
#include "ivector/voice-activity-detection.h"
#include "ivector/plda.h"

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
        ivector_.Resize(0);
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

struct OnlineIvectorExtractorConfig {

	/// feature pipeline config
	OnlineStreamIvectorExtractionConfig ivector_config;
	std::string vad_cfg;
	std::string plda_cfg;

    bool use_post;

	std::string mean_filename;
	std::string lda_filename;
	std::string plda_filename;

	OnlineIvectorExtractorConfig():use_post(false){ }

	void Register(OptionsItf *opts) {
		ivector_config.Register(opts);
		opts->Register("vad-config", &vad_cfg, "Configuration file for voice active detection");
		opts->Register("plda-config", &plda_cfg, "Configuration file for PLDA(Probabilistic Linear Discriminant Analysis) model");

		opts->Register("use-post", &use_post, "If true, ivector will be post processed after network output, "
				"e.g. lda, plda, length normalize.");
		opts->Register("mean-vec", &mean_filename, "the global mean of xvectors filename");
		opts->Register("lda-transform", &lda_filename, "Filename of xvector lda transform matrix, e.g. transform.mat");
		opts->Register("plda", &plda_filename, "PLDA model for computes log-likelihood ratios for trials.");
	}
};

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
	// type: 0, raw xvector; 1, lda transformed xvector; 2, plda transformed xvector
	BaseFloat GetScore(const VectorBase<BaseFloat> &train_ivec, int num_utts,
			const VectorBase<BaseFloat> &test_ivec, int type = 2);

	// compute enroll ivector for a speaker
	// type: 0, raw ivector; 1, lda transformed ivector
	void GetEnrollSpeakerIvector(const std::vector<Vector<BaseFloat> > &ivectors,
			Vector<BaseFloat> &spk_ivector, int type = 1);

	// Reset Extractor
	void Reset();

	int GetIvectorDim();
    int GetAudioFrequency();

private:
	void Destory();
	int VadProcess(const Matrix<BaseFloat> &vad_feat, const Matrix<BaseFloat> &in, Matrix<BaseFloat> &out);
	void IvectorPostProcess(const VectorBase<BaseFloat> &in, Vector<BaseFloat> &out, int tpye = 3);
	void IvectorLengthNormalize(Vector<BaseFloat> &xvector);

	/// feature pipeline config
	OnlineIvectorExtractorConfig *extractor_config_;
	OnlineStreamIvectorExtractionInfo *ivector_info_;
	OnlineNnetFeaturePipelineOptions *feature_opts_;
	VadEnergyOptions vad_opts_;
	PldaConfig plda_config_;

	// feature pipeline
	OnlineNnetFeaturePipeline *base_feature_pipeline_;
	OnlineStreamIvectorFeature *ivector_feature_;
	// lda transform
	Matrix<BaseFloat> lda_transform_;
	// out-of-domain PLDA model
	Plda plda_;
	// train xvector mean
	Vector<BaseFloat> mean_vec_;

	Ivector ivector_;
};

}	 // namespace kaldi

#endif /* ONLINE0_ONLINE_IVECTOR_EXTRACTOR_H_ */
