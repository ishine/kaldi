// online0/online-nnet-decoding.h

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

#ifndef ONLINE0_ONLINE_NNET_DECODING_H_
#define ONLINE0_ONLINE_NNET_DECODING_H_

#include "fstext/fstext-lib.h"
#include "decoder/decodable-matrix.h"
#include "util/kaldi-semaphore.h"
#include "util/kaldi-mutex.h"
#include "util/kaldi-thread.h"

#include "online0/online-nnet-faster-decoder.h"
#include "online0/online-nnet-feature-pipeline.h"
#include "online0/online-nnet-forward.h"

namespace kaldi {

struct OnlineNnetDecodingOptions {
	/// feature pipeline config
	OnlineNnetFeaturePipelineConfig feature_cfg;

	/// decoder search config
	std::string decoder_cfg;

	/// neural network forward config
	std::string forward_cfg;

	/// decoding options
	BaseFloat acoustic_scale;
	bool allow_partial;
	BaseFloat chunk_length_secs;
	int batch_size;
	int32 skip_frames;
	bool  copy_posterior;
    bool  skip_inner;
	std::string silence_phones_str;

	std::string word_syms_filename;
	std::string fst_rspecifier;
	std::string model_rspecifier;
	std::string words_wspecifier;
	std::string alignment_wspecifier;
	std::string model_type;  // hybrid, ctc

	OnlineNnetDecodingOptions():
							acoustic_scale(0.1), allow_partial(true), chunk_length_secs(0.05), batch_size(18),
							skip_frames(1), copy_posterior(true), skip_inner(false), silence_phones_str(""),
                            word_syms_filename(""), fst_rspecifier(""), model_rspecifier(""),
                            words_wspecifier(""), alignment_wspecifier(""), model_type("hybrid")
    { }

	void Register(OptionsItf *po)
	{
		feature_cfg.Register(po);

		po->Register("decoder-config", &decoder_cfg, "Configuration file for decoder search");
		po->Register("forward-config", &forward_cfg, "Configuration file for neural network forward");

		po->Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
		po->Register("allow-partial", &allow_partial, "Produce output even when final state was not reached");

	    po->Register("chunk-length", &chunk_length_secs,
	                "Length of chunk size in seconds, that we process.  Set to <= 0 "
	                "to use all input in one chunk.");
	    po->Register("batch-size", &batch_size, "batch frames feed to decoder in one time.");
	    po->Register("skip-frames", &skip_frames, "skip frames for next input");
	    po->Register("copy-posterior", &copy_posterior, "Copy posterior for skip frames output");
	    po->Register("skip-inner", &skip_inner, "Skip frame in neural network inner or input");
		po->Register("silence-phones", &silence_phones_str,
                     "Colon-separated list of integer ids of silence phones, e.g. 1:2:3");

		po->Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
	    po->Register("fst-rspecifier", &fst_rspecifier, "fst filename");
	    po->Register("model-rspecifier", &model_rspecifier, "transition model filename");
	    po->Register("words-wspecifier", &words_wspecifier, "transcript wspecifier");
	    po->Register("alignment-wspecifier", &alignment_wspecifier, "alignment wspecifier");
	}
};

typedef enum {
	FEAT_START,
	FEAT_APPEND,
	FEAT_END,
}FeatState;

struct OnlineDecodableBlock {

	int utt_flag;
	Matrix<BaseFloat> decodable;
	OnlineDecodableBlock(const MatrixBase<BaseFloat> &in, int flag) {
		utt_flag = flag;
		decodable.Resize(in.NumRows(), in.NumCols(), kUndefined);
		decodable.CopyFromMat(in);
	}
};

typedef struct Result_ {
	std::vector<int> word_ids_;
	std::vector<int> tids_;
	std::string utt;
	BaseFloat score_;
	int num_frames;
    int post_frames;
	bool isend;
	void clear() {
		word_ids_.clear();
		tids_.clear();
		score_ = 0.0;
		num_frames = 0;
        post_frames = 0;
		utt = "";
		isend = false;
	}
}Result;

class OnlineNnetDecodingClass : public MultiThreadable
{
public:
	OnlineNnetDecodingClass(const OnlineNnetDecodingOptions &opts,
			OnlineNnetFasterDecoder *decoder,
			OnlineDecodableInterface *decodable,
			Repository *repository,
			Result *result):
				opts_(opts),
				decoder_(decoder), decodable_(decodable), repository_(repository),
				result_(result)
	{ }

	~OnlineNnetDecodingClass() {}

	void operator () ()
	{
		typedef OnlineNnetFasterDecoder::DecodeState DecodeState;
		fst::VectorFst<LatticeArc> out_fst;
		std::vector<int> word_ids;
		std::vector<int> tids;
		LatticeWeight weight;

		OnlineDecodableBlock *block = NULL;
		DecodeState state;
        bool new_partial = false;

		decoder_->ResetDecoder(true);
		decoder_->InitDecoding();
		decodable_->Reset();

		while (1) {

			// get decodable
			while ((block = (OnlineDecodableBlock*)(repository_->Provide())) != NULL) {
				decodable_->AcceptLoglikes(&block->decodable);
				if (block->utt_flag == FEAT_END)
					decodable_->InputIsFinished();
				delete block;
				break;
			}

			if (block == NULL) break;

			// decoding
			while (decoder_->frame() < decodable_->NumFramesReady()) {
				state = decoder_->Decode(decodable_);
				if (state != DecodeState::kEndFeats) {
					new_partial = decoder_->PartialTraceback(&out_fst);
				} else {
					decoder_->FinishTraceBack(&out_fst);
				}

                if (new_partial || state == DecodeState::kEndFeats) {
                    tids.clear();
                    word_ids.clear();
				    fst::GetLinearSymbolSequence(out_fst, &tids, &word_ids, &weight);

				    for (int i = 0; i < word_ids.size(); i++)
					    result_->word_ids_.push_back(word_ids[i]);
				    for (int i = 0; i < tids.size(); i++)
					    result_->tids_.push_back(tids[i]);
					result_->score_ += (-weight.Value1() - weight.Value2());
                }

				if (state == DecodeState::kEndFeats) {
					result_->score_ /= result_->post_frames;
					result_->isend = true;
				}
			}

			// new utterance, reset decoder
			if (decodable_->IsLastFrame(decoder_->frame()-1)) {
				decoder_->ResetDecoder(true);
				decoder_->InitDecoding();
				decodable_->Reset();
			}
		}
	}

private:

	const OnlineNnetDecodingOptions &opts_;
	OnlineNnetFasterDecoder *decoder_;
	OnlineDecodableInterface *decodable_;
	Repository *repository_;
	Result *result_;
};


}// namespace kaldi

#endif /* ONLINE0_ONLINE_NNET_DECODING_H_ */
