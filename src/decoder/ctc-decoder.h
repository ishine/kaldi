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
#include "util/trie-tree.h"

#if HAVE_KENLM == 1
#include "lm/model.hh"
		typedef lm::ngram::Model KenModel;
		typedef lm::ngram::State KenState;
		typedef lm::ngram::Vocabulary KenVocab;
#endif

#define PREFIX_MAX_LEN 30

namespace kaldi {

struct PrefixSeq {
	PrefixSeq(LstmlmHistroy *h, int blank = 0) {
		prefix.push_back(blank);
		lmhis = h;
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
        logp_lm = 0;
        prefix_len = 1;
        logp = 0;
        scene_node = NULL;
	}

	PrefixSeq(LstmlmHistroy *h, const std::vector<int> &words) {
		lmhis = h;
		prefix = words;
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
        logp_lm = 0;
        prefix_len = prefix.size();
        logp = 0;
        scene_node = NULL;
	}

	PrefixSeq(const std::vector<int> &words) {
		prefix = words;
		lmhis = NULL;
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
        logp_lm = 0;
        prefix_len = prefix.size();
        logp = 0;
        scene_node = NULL;
	}

	PrefixSeq() {
		Reset();
	}

	void Reset() {
		//prefix.resize(PREFIX_MAX_LEN, 0);
        prefix.clear();
		prefix_len = 0;
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
		logp_lm = 0;
		logp = 0;
		scene_node = NULL;
	}

	void PrefixAppend(int word) {
        /*
		if (prefix_len >= prefix.size()) {
			prefix.resize(prefix.size()+PREFIX_MAX_LEN, 0);
		}
		prefix[prefix_len] = word;
		prefix_len++;
        */
        prefix.push_back(word);
	}

    int PrefixBack() {
        /*
        int id = prefix_len > 0 ? prefix_len-1 : 0;
        return prefix[id];
        */
        return prefix.back();
    }

	bool operator < (const PrefixSeq& preseq) const {
		return logp > preseq.logp;
	}

#if HAVE_KENLM == 1
	PrefixSeq(LstmlmHistroy *h, const std::vector<int> &words, KenState &state, std::vector<KenState> &sub_state) {
		lmhis = h;
		prefix = words;
		ken_state = state;
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
        logp_lm = 0;
		logp = 0;
		sub_ken_state = sub_state;
		scene_node = NULL;
	}

	KenState	ken_state;
	std::vector<KenState> sub_ken_state;
#endif

	// decoded word list
	std::vector<int> prefix;

	// rnnt language model history
	LstmlmHistroy *lmhis;
	LstmlmHistroy *next_lmhis;
	Vector<BaseFloat> *lmlogp;
	TrieNode *scene_node;

	// log probabilities for the prefix given that
	// it ends in a blank and dose not end in a blank at this time step.
	BaseFloat logp_blank;
	BaseFloat logp_nblank;
    BaseFloat logp_lm;
    BaseFloat logp;
    int prefix_len;

	std::string tostring() {
		return "";
	}
};

struct CTCDecoderUtil {
	static bool compare_PrefixSeq_reverse(const PrefixSeq *a, const PrefixSeq *b) {
		return a->logp > b->logp;
	}

    static float len_penalty(int len, float alpha) {
        return pow((5+len), alpha)/pow((5+1), alpha);
    }

	static bool compare_PrefixSeq_penalty_reverse(const PrefixSeq *a, const PrefixSeq *b) {
		float score_a, score_b;
		int len_a, len_b;
		bool use_penalty = true;
		len_a = use_penalty ? a->prefix.size()-1 : 1;
		len_b = use_penalty ? b->prefix.size()-1 : 1;
		score_a = a->logp/len_penalty(len_a, 0.65);
		score_b = b->logp/len_penalty(len_b, 0.65);
		return score_a > score_b;
	}
};

struct CTCDecoderOptions {
  int beam;
  int scene_beam;
  int blank;
  int   am_topk;
  float lm_scale;
  float blank_threshold;
  float blank_penalty;
  int max_mem;
  float rnnlm_scale;
  int sos;
  int eos;
  bool use_kenlm;
  int vocab_size;
  int scene_topk;
  std::string use_mode;
  std::string keywords;
  std::string pinyin2words_id_rxfilename;
  std::string word2wordid_rxfilename;
  std::string scene_syms_filename;

  CTCDecoderOptions(): beam(5), scene_beam(5), blank(0), am_topk(0),
		  	  	  	   lm_scale(0.0), blank_threshold(0.0), blank_penalty(0.1), max_mem(50000),
					   rnnlm_scale(1.0), sos(0), eos(0), use_kenlm(false), vocab_size(7531), scene_topk(0),
                       use_mode("normal"), keywords(""),
					   pinyin2words_id_rxfilename(""), word2wordid_rxfilename(""), scene_syms_filename("")
                        { }
  void Register(OptionsItf *opts) {
	opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
	opts->Register("scene-beam", &scene_beam, "Decoding scene beam.  Larger->slower, more accurate.");
	opts->Register("blank", &blank, "CTC bank id.");
	opts->Register("am-topk", &am_topk, "For each time step beam search, keep top K am output probability words.");
	opts->Register("lm-scale", &lm_scale, "Process extend language model log probability.");
	opts->Register("blank-threshold", &blank_threshold, "Procee am blank output probability, exceed threshold will be blank directly.");
	opts->Register("blank-penalty", &blank_penalty, "Blank posterior scale penalty.");
	opts->Register("max-mem", &max_mem, "maximum memory in decoding.");
	opts->Register("rnnlm-scale", &rnnlm_scale, "rnnlm ritio (ngramlm 1-ritio) when using rnnlm and ngramlm fusion score.");
    opts->Register("sos", &sos, "Integer corresponds to <s>. You must set this to your actual SOS integer.");
    opts->Register("eos", &eos, "Integer corresponds to </s>. You must set this to your actual EOS integer.");
    opts->Register("use-kenlm", &use_kenlm, "Weather to use ken arpa language wrapper.");
	opts->Register("vocab-size", &vocab_size, "Acoustic model output size.");
    opts->Register("use-mode", &use_mode, "Select beam search algorithm mode(normal|easy).");
    opts->Register("keywords", &keywords, "Cat the keywords before the utterance (keyword1+keyword2).");
	opts->Register("word2wordid-table", &word2wordid_rxfilename, "Map from word to word id table.");
	opts->Register("scene_syms_filename", &scene_syms_filename, "Symbol table for scene asr filename");
	opts->Register("scene_topk", &scene_topk, "For each time step beam search, keep top K am output probability words in scene asr.");
  }
};

class CTCDecoder {
	typedef Vector<BaseFloat> Pred;
	public:
		CTCDecoder(CTCDecoderOptions &config, KaldiLstmlmWrapper *lstmlm, ConstArpaLm *const_arpa,
				std::vector<ConstArpaLm *> &sub_const_arpa);

#if HAVE_KENLM == 1
		CTCDecoder(CTCDecoderOptions &config, KaldiLstmlmWrapper *lstmlm, KenModel *kenlm_arpa,
				std::vector<KenModel *> &sub_kenlm_apra);
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

		void BeamSearchTopk(const Matrix<BaseFloat> &loglikes);

		int ProcessKeywordsTopk(const Matrix<BaseFloat> &loglikes);

		void BeamSearchEasyTopk(const Matrix<BaseFloat> &loglikes);

		bool GetBestPath(std::vector<int> &words, BaseFloat &logp, BaseFloat &logp_lm);

		void BeamSearchEasySceneTopk(const Matrix<BaseFloat> &loglikes);

	protected:
        typedef unordered_map<std::vector<int>,
        		PrefixSeq*, VectorHasher<int> > BeamType;
        typedef unordered_map<std::vector<int>,
        		LstmlmHistroy*, VectorHasher<int> > HisType;
        typedef unordered_map<std::vector<int>,
        		Vector<BaseFloat>*, VectorHasher<int> > LogProbType;

        void Initialize();
		void InitDecoding();
		void InitEasyDecoding(int topk);
		void FreeBeam(BeamType *beam);
		void FreeSeq(PrefixSeq *seq);
		void FreePred(Pred *pred);
		void FreeHis(LstmlmHistroy *his);
		Pred* MallocPred();
		LstmlmHistroy* MallocHis();
		void CopyHis(LstmlmHistroy* his);
        void CleanBuffer();

        void BeamMerge(std::vector<PrefixSeq*> &merge_beam,
        		std::vector<PrefixSeq*> *scene_beam, bool skip_blank = false);


		CTCDecoderOptions &config_;
		KaldiLstmlmWrapper *lstmlm_;
		ConstArpaLm *const_arpa_;
		std::vector<ConstArpaLm *> sub_const_arpa_;
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
		std::unordered_map<std::string, int> word_to_wordid_;
		bool use_pinyin_;

		// easy beam search
		int cur_beam_size_;
		int cur_scene_beam_size_;
		int next_beam_size_;
		int next_scene_beam_size_;
		std::vector<PrefixSeq> beam_easy_;
		std::vector<PrefixSeq> next_beam_easy_;
		std::vector<PrefixSeq> next_scene_beam_;
		std::vector<LstmlmHistroy> rnnlm_his_;
		std::vector<Vector<BaseFloat> > rnnlm_logp_;
		std::vector<std::vector<int> > keywords_;
		std::vector<int> keyword_;
		Trie scene_trie_;
		bool in_scene_;

#if HAVE_KENLM == 1
		const KenVocab *kenlm_vocab_;
		std::vector<const KenVocab *> sub_kenlm_vocab_;
		KenModel *kenlm_arpa_;
		std::vector<KenModel *> sub_kenlm_apra_;
#endif

	KALDI_DISALLOW_COPY_AND_ASSIGN(CTCDecoder);
};


} // end namespace kaldi.


#endif
