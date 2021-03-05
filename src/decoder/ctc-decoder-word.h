// decoder/ctc-decoder-word.h

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

#ifndef KALDI_DECODER_CTC_DECODER_WORD_H_
#define KALDI_DECODER_CTC_DECODER_WORD_H_

#include <list>
#include "base/kaldi-common.h"
#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/trie-tree.h"
#include "lm/kaldi-lstmlm.h"
#include "lm/const-arpa-lm.h"

#if HAVE_KENLM == 1
#include "lm/model.hh"
		typedef lm::ngram::Model KenModel;
		typedef lm::ngram::State KenState;
		typedef lm::ngram::Vocabulary KenVocab;
#endif

#define PREFIX_BPE_MAX_LEN 30
#define PREFIX_WORD_MAX_LEN 20

namespace kaldi {

struct PrefixSeqWord {

	PrefixSeqWord() {
		Reset();
	}

	void Reset() {
        prefix_bpe.clear();
        prefix_word.clear();
        prefix_bpe.reserve(PREFIX_BPE_MAX_LEN);
        prefix_word.reserve(PREFIX_WORD_MAX_LEN);
		logp_blank = kLogZeroFloat;
		logp_nblank = kLogZeroFloat;
        ngram_logp = kLogZeroFloat;
        rnnlm_logp = kLogZeroFloat;
        ranklm_logp = 0;
		logp_lm = 0;
		logp = 0;
		trie_node = NULL;
	}

	void PrefixBpeAppend(int bpe) {
        prefix_bpe.push_back(bpe);
	}

    int PrefixBpeBack() {
        return prefix_bpe.back();
    }

	void PrefixWordAppend(int word) {
        prefix_word.push_back(word);
	}

    int PrefixWordBack() {
        return prefix_word.back();
    }

	bool operator < (const PrefixSeqWord& preseq) const {
		return logp > preseq.logp;
	}

#if HAVE_KENLM == 1
	KenState	ken_state;
	std::vector<KenState> sub_ken_state;
#endif

	// decoded word list
	std::vector<int> prefix_bpe;
	std::vector<int> prefix_word;

	// rnn language model history
	LstmlmHistroy *lmhis;
	LstmlmHistroy *next_lmhis;
    int idx1;
    int idx2;

	// prefix tree for word dict
	TrieNode *trie_node;

	// log probabilities for the prefix given that
	// it ends in a blank and dose not end in a blank at this time step.
	BaseFloat logp_blank;
	BaseFloat logp_nblank;
    BaseFloat ngram_logp;
    BaseFloat rnnlm_logp;
    BaseFloat ranklm_logp;
    BaseFloat logp_lm;
    BaseFloat logp;

	std::string ToStr();
};

struct CTCDecoderWordUtil {
	static bool compare_PrefixSeqWord_reverse(const PrefixSeqWord *a, const PrefixSeqWord *b) {
		return a->logp > b->logp;
	}

	static bool compare_PrefixSeqRank_reverse(const PrefixSeqWord *a, const PrefixSeqWord *b) {
		return a->ranklm_logp+LogAdd(a->logp_blank, a->logp_nblank) > b->ranklm_logp+LogAdd(b->logp_blank, b->logp_nblank);
	}

	static bool compare_PrefixSeqBpe_reverse(const PrefixSeqWord *a, const PrefixSeqWord *b) {
        KALDI_ASSERT(a->trie_node != NULL && b->trie_node != NULL);

        float logp_lm_a, logp_lm_b, bpe_len, nbpe;
        bpe_len = a->prefix_bpe.size()-1;
        nbpe = a->trie_node->layer_+1;
        logp_lm_a = a->logp_lm > 0 ? a->logp_lm/(bpe_len-nbpe)*bpe_len : a->logp_lm;

        bpe_len = b->prefix_bpe.size()-1;
        nbpe = b->trie_node->layer_+1;
        logp_lm_b = b->logp_lm > 0 ? b->logp_lm/(bpe_len-nbpe)*bpe_len : b->logp_lm;
		return logp_lm_a+LogAdd(a->logp_blank, a->logp_nblank) > logp_lm_b+LogAdd(b->logp_blank, b->logp_nblank);
	}
};

struct CTCDecoderWordOptions {
	int beam;
	int py_beam;
	int word_beam;
	int blank;
	int   am_topk;
	float lm_scale;
	float blank_threshold;
	float blank_penalty;
	float rnnlm_scale;
	int sos;
	int eos;
	bool use_kenlm;
	int vocab_size;
    std::string bpe2bpeid_rxfilename;
	std::string word2wordid_rxfilename;
	std::string word2bpeid_rxfilename;
    std::string pinyin2bpeid_rxfilename;

	CTCDecoderWordOptions(): beam(10), py_beam(10), word_beam(10), blank(0), am_topk(0),
					   lm_scale(0.0), blank_threshold(0.0), blank_penalty(0.1),
					   rnnlm_scale(1.0), sos(0), eos(0), use_kenlm(true), vocab_size(7531),
					   bpe2bpeid_rxfilename(""), word2wordid_rxfilename(""), word2bpeid_rxfilename(""),
					   pinyin2bpeid_rxfilename("") {}

	void Register(OptionsItf *opts) {
		opts->Register("beam", &beam, "Decoding beam. sLarger->slower, more accurate.");
		opts->Register("word-beam", &word_beam, "Decoding word beam. Larger->slower, more accurate.");
		opts->Register("blank", &blank, "CTC bank id.");
		opts->Register("am-topk", &am_topk, "For each time step beam search, keep top K am output probability words.");
		opts->Register("lm-scale", &lm_scale, "Process extend language model log probability.");
		opts->Register("blank-threshold", &blank_threshold, "Procee am blank output probability, exceed threshold will be blank directly.");
		opts->Register("blank-penalty", &blank_penalty, "Blank posterior scale penalty.");
		opts->Register("rnnlm-scale", &rnnlm_scale, "rnnlm ritio (ngramlm 1-ritio) when using rnnlm and ngramlm fusion score.");
		opts->Register("sos", &sos, "Integer corresponds to <s>. You must set this to your actual SOS integer.");
		opts->Register("eos", &eos, "Integer corresponds to </s>. You must set this to your actual EOS integer.");
		opts->Register("use-kenlm", &use_kenlm, "Weather to use ken arpa language wrapper.");
		opts->Register("vocab-size", &vocab_size, "Acoustic model output size.");
		opts->Register("bpe2bpeid-table", &bpe2bpeid_rxfilename, "Map from bpe to bpe id table.");
		opts->Register("word2wordid-table", &word2wordid_rxfilename, "Map from word to word id table.");
		opts->Register("word2bpeid-table", &word2bpeid_rxfilename, "Map from word to bpe ids table");
		opts->Register("pinyin2bpeid-table", &pinyin2bpeid_rxfilename, "Map from pinyin to bpe ids table");
	}
};

class CTCDecoderWord {
	public:

#if HAVE_KENLM == 1
		CTCDecoderWord(CTCDecoderWordOptions &config, KaldiLstmlmWrapper *lstmlm, KenModel *kenlm_arpa,
				std::vector<KenModel *> &sub_kenlm_apra, KenModel *rank_kenlm_arpa = NULL);
#endif

		void GreedySearch(const Matrix<BaseFloat> &loglikes);

        // loglikes: The output probabilities (e.g. log post-softmax) for each time step.
        // Should be an array of shape (time x output dim).
        // beam_size (int): Size of the beam to use during inference.
        // blank (int): Index of the CTC blank label.
        // Returns the output label sequence and the corresponding negative
        // log-likelihood estimated by the decoder.

		bool GetBestPath(std::vector<int> &words, BaseFloat &logp, BaseFloat &logp_lm);

		void BeamSearchTopk(const Matrix<BaseFloat> &loglikes);

	protected:
        void DeleteInWordBeam(std::vector<PrefixSeqWord> &beam, int size);

        std::string DebugBeam(std::vector<PrefixSeqWord> &beam, int size, int nframe);

        typedef unordered_map<std::vector<int>,
        		PrefixSeqWord*, VectorHasher<int> > BeamType;
        typedef unordered_map<std::vector<int>,
        		LstmlmHistroy*, VectorHasher<int> > HisType;
        typedef unordered_map<std::vector<int>,
        		Vector<BaseFloat>*, VectorHasher<int> > LogProbType;

        void Initialize();
		void InitDecoding(int topk);

        void BeamMerge(std::vector<PrefixSeqWord*> &bep_beam,
        		std::vector<PrefixSeqWord*> &word_beam, bool skip_blank = false);


		CTCDecoderWordOptions &config_;
		KaldiLstmlmWrapper *lstmlm_;

        BeamType beam_union_;

		std::vector<std::string> wordid_to_word_;
		std::vector<std::string> bpeid_to_bpe_;
		std::unordered_map<std::string, int> word_to_wordid_;
		std::unordered_map<std::string, int> bpe_to_bpeid_;

        // pinyin
        bool use_pinyin_;
		std::vector<std::vector<int> > pinyin_to_bpe_;

		// rnn lm
		std::vector<int> rd_;
		std::vector<int> cd_;
		std::vector<LstmlmHistroy> rnnlm_his_;
		std::vector<int> in_words_;
		std::vector<std::vector<int> > out_words_;
		std::vector<std::vector<PrefixSeqWord*> > out_preseq_;
		std::vector<std::vector<BaseFloat> > out_words_score_;
		std::vector<LstmlmHistroy*> rnnlm_his_in_;
		std::vector<LstmlmHistroy*> rnnlm_his_out_;
        std::vector<bool> rnnlm_his_table_;

		// beam search
		Trie word_trie_;
		std::vector<PrefixSeqWord> cur_beam_;
		std::vector<PrefixSeqWord> next_bpe_beam_;
		std::vector<PrefixSeqWord> next_word_beam_;

		int cur_bpe_beam_size_;
		int cur_word_beam_size_;
		int next_bpe_beam_size_;
		int next_word_beam_size_;

#if HAVE_KENLM == 1
		const KenVocab *kenlm_vocab_;
		std::vector<const KenVocab *> sub_kenlm_vocab_;
		KenModel *kenlm_arpa_;
		std::vector<KenModel *> sub_kenlm_apra_;
		const KenVocab *rank_kenlm_vocab_;
		KenModel *rank_kenlm_arpa_;
#endif

	KALDI_DISALLOW_COPY_AND_ASSIGN(CTCDecoderWord);
};


} // end namespace kaldi.


#endif
