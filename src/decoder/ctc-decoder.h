// decoder/ctc-decoder.h

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

#ifndef KALDI_DECODER_CTC_DECODER_H_
#define KALDI_DECODER_CTC_DECODER_H_

#include <list>
#include "base/kaldi-common.h"
#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"
#include "lm/kaldi-lstmlm.h"
#include "lm/const-arpa-lm.h"

#if HAVE_KENLM == 1
#include "lm/model.hh"
		typedef lm::ngram::Model KenModel;
		typedef lm::ngram::State KenState;
		typedef lm::ngram::Vocabulary KenVocab;
#endif


namespace kaldi {

struct PrefixSeq {
	PrefixSeq(LstmlmHistroy *h, int blank = 0) {
		prefix.push_back(blank);
		lmhis = h;
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
	}

	PrefixSeq(LstmlmHistroy *h, const std::vector<int> &words) {
		lmhis = h;
		prefix = words;
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
	}

	PrefixSeq(const std::vector<int> &words) {
		prefix = words;
		lmhis = NULL;
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
	}

#if HAVE_KENLM == 1
	PrefixSeq(LstmlmHistroy *h, const std::vector<int> &words, KenState &state) {
		lmhis = h;
		prefix = words;
		ken_state = state;
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
	}

	KenState	ken_state;
#endif

	// decoded word list
	std::vector<int> prefix;

	// rnnt language model history
	LstmlmHistroy *lmhis;

	// log probabilities for the prefix given that
	// it ends in a blank and dose not end in a blank at this time step.
	BaseFloat logp_blank;
	BaseFloat logp_nblank;

	std::string tostring() {
		return "";
	}
};

struct CTCDecoderUtil {
	static bool compare_PrefixSeq_reverse(const PrefixSeq *a, const PrefixSeq *b) {
		return LogAdd(a->logp_blank,a->logp_nblank) > LogAdd(b->logp_blank,b->logp_nblank);
	}

	static bool compare_PrefixSeq_penalty_reverse(const PrefixSeq *a, const PrefixSeq *b) {
		float score_a, score_b;
		int len_a, len_b;
		bool use_penalty = true;
		len_a = use_penalty ? a->prefix.size()-1 : 1;
		len_b = use_penalty ? b->prefix.size()-1 : 1;
		score_a = LogAdd(a->logp_blank, a->logp_nblank)/(len_a*0.1);
		score_b = LogAdd(b->logp_blank, b->logp_nblank)/(len_b*0.1);
		return score_a > score_b;
	}
};

struct CTCDecoderOptions {
  int beam;
  int blank;
  int   am_topk;
  float lm_scale;
  float blank_threshold;
  int max_mem;
  float rnnlm_scale;
  int sos;
  int eos;
  bool use_kenlm;
  std::string pinyin2words_id_rxfilename;
  std::string word2wordid_rxfilename;

  CTCDecoderOptions(): beam(5), blank(0), am_topk(30),
		  	  	  	   lm_scale(0.0), blank_threshold(0.0), max_mem(50000),
					   rnnlm_scale(1.0), sos(0), eos(0), use_kenlm(false),
                       pinyin2words_id_rxfilename(""), word2wordid_rxfilename("")
                        { }
  void Register(OptionsItf *opts) {
	opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
	opts->Register("blank", &blank, "CTC bank id.");
	opts->Register("am-topk", &am_topk, "For each time step beam search, keep top K am output probability words.");
	opts->Register("lm-scale", &lm_scale, "Process extend language model log probability.");
	opts->Register("blank-threshold", &blank_threshold, "Procee am blank output probability, exceed threshold will be blank directly.");
	opts->Register("max-mem", &max_mem, "maximum memory in decoding.");
	opts->Register("rnnlm-scale", &rnnlm_scale, "rnnlm ritio (ngramlm 1-ritio) when using rnnlm and ngramlm fusion score.");
    opts->Register("sos", &sos, "Integer corresponds to <s>. You must set this to your actual SOS integer.");
    opts->Register("eos", &eos, "Integer corresponds to </s>. You must set this to your actual EOS integer.");
    opts->Register("use-kenlm", &use_kenlm, "Weather to use ken arpa language wrapper.");
	opts->Register("pinyin2words-table", &pinyin2words_id_rxfilename, "Map from pinyin to words table.");
	opts->Register("word2wordid-table", &word2wordid_rxfilename, "Map from word to word id table.");
  }
};

class CTCDecoder {
	typedef Vector<BaseFloat> Pred;
	public:
		CTCDecoder(CTCDecoderOptions &config, KaldiLstmlmWrapper &lstmlm, ConstArpaLm *const_arpa);

#if HAVE_KENLM == 1
		CTCDecoder(CTCDecoderOptions &config, KaldiLstmlmWrapper &lstmlm, KenModel *kenlm_arpa_);
#endif

		void GreedySearch(const Matrix<BaseFloat> &loglikes);

        // loglikes: The output probabilities (e.g. log post-softmax) for each time step.
        // Should be an array of shape (time x output dim).
        // beam_size (int): Size of the beam to use during inference.
        // blank (int): Index of the CTC blank label.
        // Returns the output label sequence and the corresponding negative
        // log-likelihood estimated by the decoder.
		void BeamSearchNaive(const Matrix<BaseFloat> &loglikes);

		void BeamSearch(const Matrix<BaseFloat> &loglikes);

		bool GetBestPath(std::vector<int> &words, BaseFloat &logp);

	protected:
        typedef unordered_map<std::vector<int>,
        		PrefixSeq*, VectorHasher<int> > BeamType;
        typedef unordered_map<std::vector<int>,
        		LstmlmHistroy*, VectorHasher<int> > HisType;
        typedef unordered_map<std::vector<int>,
        		Vector<BaseFloat>*, VectorHasher<int> > LogProbType;

        void Initialize();
		void InitDecoding();
		void FreeBeam(BeamType *beam);
		void FreeSeq(PrefixSeq *seq);
		void FreePred(Pred *pred);
		void FreeHis(LstmlmHistroy *his);
		Pred* MallocPred();
		LstmlmHistroy* MallocHis();
		void CopyHis(LstmlmHistroy* his);
        void CleanBuffer();


		CTCDecoderOptions &config_;
		KaldiLstmlmWrapper &lstmlm_;
		ConstArpaLm *const_arpa_;
		BeamType beam_;
		BeamType next_beam_;
		HisType next_his_;
		LogProbType next_logprob_;
		std::list<PrefixSeq*> pre_seq_list_;

		std::unordered_map<Vector<BaseFloat> *, int> pred_buffer_;
		std::unordered_map<LstmlmHistroy *, int> his_buffer_;
		std::list<Vector<BaseFloat> *> pred_list_;
		std::list<LstmlmHistroy *>	his_list_;
		std::vector<int> rd_;
		std::vector<int> cd_;
		std::vector<std::vector<int> > pinyin2words_;
		std::vector<std::string> wordid_to_word_;
		bool use_pinyin_;

#if HAVE_KENLM == 1
		KenModel *kenlm_arpa_;
		const KenVocab *kenlm_vocab_;
#endif

	KALDI_DISALLOW_COPY_AND_ASSIGN(CTCDecoder);
};


} // end namespace kaldi.


#endif
