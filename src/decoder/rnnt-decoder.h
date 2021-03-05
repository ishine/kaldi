// decoder/rnnt-decoder.h

// Copyright 2018-2019   Alibaba Inc (author: Wei Deng)

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

#ifndef KALDI_DECODER_RNNT_DECODER_H_
#define KALDI_DECODER_RNNT_DECODER_H_

#include <list>

#include "lm/kaldi-lstmlm.h"
#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"


namespace kaldi {

struct Sequence {
	Sequence(LstmlmHistroy *h, int blank = 0) {
		pred.clear();
		k.push_back(blank);
		lmhis = h;
		logp = 0;
	}

	std::vector<Vector<BaseFloat>* > pred; 	// rnnt language model output
	std::vector<int> k;						// decoded word list
	LstmlmHistroy *lmhis;					// rnnt language model history
	BaseFloat logp;							// probability of this sequence, in log scale

	std::string tostring() {
		return "";
	}
};

struct RNNTDecoderUtil {
	static bool compare_len(const Sequence *a, const Sequence *b) {
		return a->k.size() < b->k.size();
	}

	static bool compare_len_reverse(const Sequence *a, const Sequence *b) {
		return a->k.size() > b->k.size();
	}

	static bool compare_logp(const Sequence *a, const Sequence *b) {
		return a->logp < b->logp;
	}

	static bool compare_logp_reverse(const Sequence *a, const Sequence *b) {
		return a->logp > b->logp;
	}

	static bool compare_normlogp_reverse(const Sequence *a, const Sequence *b) {
		return a->logp/a->k.size() > b->logp/b->k.size();
	}

	static bool isprefix(const std::vector<int> &a, const std::vector<int> &b) {
		int lena = a.size();
		int lenb = b.size();
		if (lena >= lenb) return false;
		for (int i = 0; i <= lena; i++)
			if (a[i] != b[i]) return false;
		return true;
	}

};

struct RNNTDecoderOptions {
  int beam;
  int blank;
  bool use_prefix;
  bool norm_length;
  int max_mem;
  float blank_posterior_scale;
  int topk;

  RNNTDecoderOptions(): beam(5), blank(0),
		  	  	  	  	use_prefix(false), norm_length(true), max_mem(20000),
                        blank_posterior_scale(-1.0), topk(30)
                        { }
  void Register(OptionsItf *opts) {
	opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
	opts->Register("blank", &blank, "RNNT bank id.");
	opts->Register("use-prefix", &use_prefix, "Process prefix probability.");
	opts->Register("norm-length", &norm_length, "Sort beam use log probability with normalized prefix length.");
	opts->Register("max-mem", &max_mem, "maximum memory in decoding.");
    opts->Register("blank-posterior-scale", &blank_posterior_scale, "For RNNT decoding, scale blank label posterior by a constant value(e.g. 0.11), other label posteriors are directly used in decoding.");
	opts->Register("topk", &topk, "For each time step beam search, keep top K am output probability words.");
  }
};

class RNNTDecoder {
	typedef Vector<BaseFloat> Pred;
	public:
		RNNTDecoder(KaldiLstmlmWrapper &rnntlm, RNNTDecoderOptions &config);
		void GreedySearch(const Matrix<BaseFloat> &loglikes);
		void BeamSearchNaive(const Matrix<BaseFloat> &loglikes);
		void BeamSearch(const Matrix<BaseFloat> &loglikes);
		bool GetBestPath(std::vector<int> &words, BaseFloat &logp);

	protected:
		void InitDecoding();
		void FreeList(std::list<Sequence* > *list);
		void FreeSeq(Sequence *seq);
		void FreePred(Pred *pred);
		void FreeHis(LstmlmHistroy *his);
		Pred* MallocPred();
		LstmlmHistroy* MallocHis();
		void CopyPredList(std::vector<Pred*> &predlist);
		void CopyHis(LstmlmHistroy* his);
		void DeepCopySeq(Sequence *seq);
        void CleanBuffer();


		RNNTDecoderOptions &config_;
		KaldiLstmlmWrapper &rnntlm_;
		std::list<Sequence* > *A_;
		std::list<Sequence* > *B_;
		std::unordered_map<Vector<BaseFloat> *, int> pred_buffer_;
		std::unordered_map<LstmlmHistroy *, int> his_buffer_;
		std::list<Vector<BaseFloat> *> pred_list_;
		std::list<LstmlmHistroy *>	his_list_;
		std::vector<int> rd_;
		std::vector<int> cd_;

	KALDI_DISALLOW_COPY_AND_ASSIGN(RNNTDecoder);
};


} // end namespace kaldi.


#endif
