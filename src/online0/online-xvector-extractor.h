// online0/online-xvector-extractor.h
// Copyright 2018	Alibaba Inc (author: Wei Deng)

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

#ifndef ONLINE0_ONLINE_XVECTOR_EXTRACTOR_H_
#define ONLINE0_ONLINE_XVECTOR_EXTRACTOR_H_

#include "online0/online-nnet-feature-pipeline.h"
#include "online0/online-nnet3-forward.h"
#include "ivector/voice-activity-detection.h"
#include "ivector/plda.h"
#include "online2/online-speex-wrapper.h"

namespace kaldi {

typedef struct Xvector_ {
    bool    isvalid_;
	Vector<float> xvector_;
	double	num_frames_;
	std::string utt;
	void clear() {
        isvalid_ = false;
        xvector_.Resize(0);
        num_frames_ = 0;
		utt = "";
	}
}Xvector;

struct OnlineXvectorExtractorConfig {

	/// feature pipeline config
	OnlineNnetFeaturePipelineConfig feature_cfg;
	/// neural network forward config
	std::string forward_cfg;
	std::string vad_cfg;
	std::string plda_cfg;
	std::string speex_cfg;

    int32 chunk_size;
    int32 min_chunk_size;
    bool pad_input;
    bool use_post;
    bool use_speex;

	std::string mean_filename;
	std::string lda_filename;
	std::string plda_filename;

	OnlineXvectorExtractorConfig():chunk_size(-1), min_chunk_size(100), pad_input(true),
			use_post(false), use_speex(false){ }

	void Register(OptionsItf *opts) {
		feature_cfg.Register(opts);
		opts->Register("forward-config", &forward_cfg, "Configuration file for neural network forward");
		opts->Register("vad-config", &vad_cfg, "Configuration file for voice active detection");
		opts->Register("plda-config", &plda_cfg, "Configuration file for PLDA(Probabilistic Linear Discriminant Analysis) model");
		opts->Register("speex-config", &speex_cfg, "Speex audio decode configure file.");

		opts->Register("chunk-size", &chunk_size,
          "If set, extracts xectors from specified chunk-size, and averages.  "
          "If not set, extracts an xvector from all available features.");
		opts->Register("min-chunk-size", &min_chunk_size,
          "Minimum chunk-size allowed when extracting xvectors.");
		opts->Register("pad-input", &pad_input, "If true, duplicate the first and "
          "last frames of the input features as required to equal min-chunk-size.");
		opts->Register("use-post", &use_post, "If true, xvector will be post processed after network output, "
				"e.g. lda, plda, length normalize.");
		opts->Register("use-speex", &use_speex, "If true, audio will be pre processed by speex decoder.");

		opts->Register("mean-vec", &mean_filename, "the global mean of xvectors filename");
		opts->Register("lda-transform", &lda_filename, "Filename of xvector lda transform matrix, e.g. transform.mat");
		opts->Register("plda", &plda_filename, "PLDA model for computes log-likelihood ratios for trials.");
	}
};


typedef enum {
	FEAT_START,
	FEAT_APPEND,
	FEAT_END,
}FeatState;

class OnlineXvectorExtractor {

public:
	OnlineXvectorExtractor(std::string cfg);
	virtual ~OnlineXvectorExtractor() { Destory(); }

	// initialize decoder
	void InitExtractor();

	// feed wave data to extractor
	int FeedData(void *data, int nbytes, FeatState state);

	// get current frame xvector,
	// type: 0, raw xvector; 1, lda transformed xvector; 2, plda transformed xvector
	Xvector* GetCurrentXvector(int type = 2);

	// compute xvector score
	// type: 0, raw xvector; 1, lda transformed xvector; 2, plda transformed xvector
	BaseFloat GetScore(const VectorBase<BaseFloat> &train_xvec, int num_utts,
			const VectorBase<BaseFloat> &test_xvec, int type = 2);

	// compute enroll xvector for a speaker
	// type: 0, raw xvector; 1, lda transformed xvector
	void GetEnrollSpeakerXvector(const std::vector<Vector<BaseFloat> > &xvectors,
			Vector<BaseFloat> &spk_xvector, int type = 1);

	// Reset Extractor
	void Reset();

	int GetXvectorDim();

private:
	void Destory();
	int VadProcess(const Matrix<BaseFloat> &vad_feat, const Matrix<BaseFloat> &in, Matrix<BaseFloat> &out);
	void XvectorPostProcess(const VectorBase<BaseFloat> &in, Vector<BaseFloat> &out, int tpye = 3);
	void XvectorLengthNormalize(Vector<BaseFloat> &xvector);

	/// feature pipeline config
	OnlineXvectorExtractorConfig *xvector_config_;
	OnlineNnetFeaturePipelineOptions *feature_opts_;
	OnlineNnet3ForwardOptions *forward_opts_;
	VadEnergyOptions vad_opts_;
	PldaConfig plda_config_;
	SpeexOptions speex_opts_;

	// feature pipeline
	OnlineNnetFeaturePipeline *feature_pipeline_;
	// forward
	OnlineNnet3Forward *forward_;
	// lda transform
	Matrix<BaseFloat> lda_transform_;
	// out-of-domain PLDA model
	Plda plda_;
	// speex stream decoder;
	OnlineSpeexDecoder *speex_decoder_;

	// train xvector mean
	Vector<BaseFloat> mean_vec_;

	Matrix<BaseFloat> feat_in_, feat_in_vad_, feat_out_, nnet_out_;

	Xvector xvector_;
};

}	 // namespace kaldi

#endif /* ONLINE0_ONLINE_XVECTOR_EXTRACTOR_H_ */
