// decoder/ctc-decoder.cc

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
#include "decoder/ctc-decoder.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace kaldi {

CTCDecoder::CTCDecoder(CTCDecoderOptions &config,
						KaldiLstmlmWrapper *lstmlm,
						ConstArpaLm *const_arpa,
						std::vector<ConstArpaLm *> &sub_const_arpa):
		config_(config), lstmlm_(lstmlm), const_arpa_(const_arpa), sub_const_arpa_(sub_const_arpa) {
	Initialize();
#if HAVE_KENLM == 1
    kenlm_arpa_ = NULL;
    kenlm_vocab_ = NULL;
#endif
}

#if HAVE_KENLM == 1
CTCDecoder::CTCDecoder(CTCDecoderOptions &config,
			            KaldiLstmlmWrapper *lstmlm,
			            KenModel *kenlm_arpa,
						std::vector<KenModel *> &sub_kenlm_apra):
		config_(config), lstmlm_(lstmlm), kenlm_arpa_(kenlm_arpa), sub_kenlm_apra_(sub_kenlm_apra) {
	Initialize();

    const_arpa_ = NULL;
    kenlm_vocab_ = NULL;
    if (kenlm_arpa_ != NULL) {
        kenlm_vocab_ = &(kenlm_arpa_->GetVocabulary());
	    // sub language lmodel vocab
        int nsub = sub_kenlm_apra.size();
        sub_kenlm_vocab_.resize(nsub);
        for (int i = 0; i < nsub; i++)
            sub_kenlm_vocab_[i] = &(sub_kenlm_apra[i]->GetVocabulary());
    }

	/// symbols
	fst::SymbolTable *word_symbols = NULL;
	if (!(word_symbols = fst::SymbolTable::ReadText(config_.word2wordid_rxfilename))) {
		KALDI_ERR << "Could not read symbol table from file " << config_.word2wordid_rxfilename;
	}
	wordid_to_word_.resize(word_symbols->NumSymbols());
	for (int32 i = 0; i < word_symbols->NumSymbols(); i++) {
		wordid_to_word_[i] = word_symbols->Find(i);
		if (wordid_to_word_[i] == "") {
		  KALDI_ERR << "Could not find word for integer " << i << "in the word "
			  << "symbol table, mismatched symbol table or you have discoutinuous "
			  << "integers in your symbol table?";
		}
	}
	for (int32 i = 0; i < word_symbols->NumSymbols(); i++) {
		if (word_symbols->Find(i) == "") {
		  KALDI_ERR << "Could not find word for integer " << i << "in the word "
			  << "symbol table, mismatched symbol table or you have discoutinuous "
			  << "integers in your symbol table?";
		}
		word_to_wordid_[word_symbols->Find(i)] = i;
	}

    if (config.scene_syms_filename != "") {
	    sceneword_.resize(word_symbols->NumSymbols(), 0.0);
	    std::ifstream is(config.scene_syms_filename);
	    std::string ss;
	    while (!is.eof()) {
		    is >> ss;
		    sceneword_[word_to_wordid_[ss]] = 1.0;
	    }
    }
}
#endif

void CTCDecoder::Initialize() {
	if (lstmlm_ != NULL) {
		rd_ = lstmlm_->GetRDim();
		cd_ = lstmlm_->GetCDim();
	}

	use_pinyin_ = false;
	if (config_.pinyin2words_id_rxfilename != "") {
		SequentialInt32VectorReader wordid_reader(config_.pinyin2words_id_rxfilename);
		for (; !wordid_reader.Done(); wordid_reader.Next()) {
			pinyin2words_.push_back(wordid_reader.Value());
		}
		use_pinyin_ = true;
	}

	if (config_.use_mode == "easy") {
		beam_easy_.resize(config_.beam, PrefixSeq());
		cur_beam_size_ = 0;

		// rnn lm
	    if (config_.rnnlm_scale > 0) {
	    	LstmlmHistroy his(rd_, cd_, kSetZero);
	    	rnnlm_his_.resize(2*config_.beam, his);
	    	rnnlm_logp_.resize(config_.beam);
	    	for (int i = 0; i < config_.beam; i++) {
	    		beam_easy_[i].lmhis = &rnnlm_his_[i];
	    		beam_easy_[i].next_lmhis = &rnnlm_his_[config_.beam+i];
	    		beam_easy_[i].lmlogp = &rnnlm_logp_[i];
	    	}
	    }
	}

	if (config_.keywords != "") {
		std::vector<std::string> kws;
		kaldi::SplitStringToVector(config_.keywords, "+", false, &kws);
		int size = kws.size();
		keywords_.resize(size);
		for (int i = 0; i < size; i++) {
			if (!kaldi::SplitStringToIntegers(kws[i], ":", false, &keywords_[i]))
				KALDI_ERR << "Invalid keyword string " << kws[i];
		}
	}
}

void CTCDecoder::FreeBeam(BeamType *beam) {
	for (auto &seq : *beam) {
		FreeSeq(seq.second);
		delete seq.second;
	}

	int idx = 0;
	if (pred_list_.size() > config_.max_mem) {
		for (auto it = pred_list_.begin(); it != pred_list_.end(); it++) {
			if (idx >= config_.max_mem) delete (*it);
			idx++;
		}
		pred_list_.resize(config_.max_mem);
	}

	idx = 0;
	if (his_list_.size() > config_.max_mem) {
		for (auto it = his_list_.begin(); it != his_list_.end(); it++) {
			if (idx >= config_.max_mem) delete (*it);
			idx++;
		}
		his_list_.resize(config_.max_mem);
	}
}

void CTCDecoder::FreeSeq(PrefixSeq *seq) {
	if (seq == NULL) return;
    FreeHis(seq->lmhis);
}

void CTCDecoder::FreePred(Pred *pred) {
	auto it = pred_buffer_.find(pred);
	if (it != pred_buffer_.end()) {
		if (it->second-1 == 0) {
			pred_buffer_.erase(it);
			pred_list_.push_back(pred);
		} else {
			pred_buffer_[pred] = it->second-1;
		}
	}
}

void CTCDecoder::FreeHis(LstmlmHistroy *his) {
	auto it = his_buffer_.find(his);
	if (it != his_buffer_.end()) {
		if (it->second-1 == 0) {
			his_buffer_.erase(it);
			his_list_.push_back(his);
		} else {
			his_buffer_[his] = it->second-1;
		}
	}
}

Vector<BaseFloat>* CTCDecoder::MallocPred() {
	Pred *pred = NULL;
	if (pred_list_.size() > 0) {
		pred = pred_list_.front();
		pred_list_.pop_front();
	} else {
		pred = new Vector<BaseFloat>;
	}

	if (pred != NULL)
		pred_buffer_[pred] = 1;

	return pred;
}

LstmlmHistroy* CTCDecoder::MallocHis() {
	LstmlmHistroy *his = NULL;
	if (his_list_.size() > 0) {
		his = his_list_.front();
		his_list_.pop_front();
	} else {
		his = new LstmlmHistroy(rd_, cd_, kUndefined);
	}

	if (his != NULL)
		his_buffer_[his] = 1;

	return his;
}

void CTCDecoder::CopyHis(LstmlmHistroy* his) {
	auto it = his_buffer_.find(his);
	if (it != his_buffer_.end())
		his_buffer_[his]++;
}

void CTCDecoder::CleanBuffer() {
    for (auto it = pred_buffer_.begin(); it != pred_buffer_.end(); it++)
        delete it->first;
    for (auto it = his_buffer_.begin(); it != his_buffer_.end(); it++)
        delete it->first;
    pred_buffer_.clear();
    his_buffer_.clear();
}

bool CTCDecoder::GetBestPath(std::vector<int> &words, BaseFloat &logp, BaseFloat &logp_lm) {
    PrefixSeq *seq = NULL;
    if (config_.use_mode == "easy")
        seq = &beam_easy_[0];
    else
	    seq = pre_seq_list_.front();

	if (seq == NULL) return false;

	logp = seq->logp;
    logp_lm = seq->logp_lm;


    int size = keyword_.size() + seq->prefix.size();
    words.resize(size);
    words[0] = seq->prefix[0];
    int k = 1;
    for (int i = 0; i < keyword_.size(); i++)
    	words[k++] = keyword_[i];
	for (int i = 1; i < seq->prefix.size(); i++)
		words[k++] = seq->prefix[i];

    /*
    if (config_.use_mode == "easy")
        words.resize(seq->prefix_len);
    */

	return true;
}

void CTCDecoder::InitDecoding() {
	FreeBeam(&beam_);
	beam_.clear();

	FreeBeam(&next_beam_);
	next_beam_.clear();

	pre_seq_list_.clear();

    CleanBuffer();

    // first input <s>
    LstmlmHistroy *sos_h = new LstmlmHistroy(rd_, cd_, kSetZero);

    // Elements in the beam are (prefix, (p_blank, p_no_blank))
    // Initialize the beam with the empty sequence, a probability of
    // 1 for ending in blank and zero for ending in non-blank (in log space).
	PrefixSeq *seq = new PrefixSeq(sos_h, config_.blank);
    seq->logp_blank = 0.0;
	CopyHis(sos_h);
#if HAVE_KENLM == 1
    if (kenlm_arpa_ != NULL)
	    seq->ken_state = kenlm_arpa_->BeginSentenceState();

    int nsub = sub_kenlm_apra_.size();
    seq->sub_ken_state.resize(nsub);
    for (int i = 0; i < nsub; i++)
        seq->sub_ken_state[i] = sub_kenlm_apra_[i]->BeginSentenceState();
#endif

	beam_[seq->prefix] = seq;
}

void CTCDecoder::InitEasyDecoding(int topk) {
    // Elements in the beam are (prefix, (p_blank, p_no_blank))
    // Initialize the beam with the empty sequence, a probability of
    // 1 for ending in blank and zero for ending in non-blank (in log space).
	PrefixSeq *seq = &beam_easy_[0];
    seq->Reset();
	seq->PrefixAppend(config_.blank);
	seq->logp_blank = 0.0;
	cur_beam_size_ = 1;
	keyword_.clear();

    if (config_.rnnlm_scale != 0) {
    	// first input <s>
    	seq->lmhis->SetZero();
    }

	// next beam buffer
	next_beam_easy_.resize(config_.beam*topk);
	next_beam_size_ = 0;
#if HAVE_KENLM == 1
    if (kenlm_arpa_ != NULL)
	    seq->ken_state = kenlm_arpa_->BeginSentenceState();

    int nsub = sub_kenlm_apra_.size();
    seq->sub_ken_state.resize(nsub);
    for (int i = 0; i < nsub; i++)
        seq->sub_ken_state[i] = sub_kenlm_apra_[i]->BeginSentenceState();
#endif
}

void CTCDecoder::GreedySearch(const Matrix<BaseFloat> &loglikes) {
	int nframe = loglikes.NumRows(), k;
	PrefixSeq *pre_seq;
	BaseFloat logp;

	InitDecoding();
	// decode one utterance
	pre_seq = beam_.begin()->second;
	for (int n = 0; n < nframe; n++) {
		logp = loglikes.Row(n).Max(&k);
		pre_seq->logp_blank += logp;
		pre_seq->prefix.push_back(k);
	}
	std::vector<int> words;
    words.push_back(config_.blank);
	for (int i = 1; i < pre_seq->prefix.size(); i++) {
		if (pre_seq->prefix[i] != config_.blank && pre_seq->prefix[i] != pre_seq->prefix[i-1])
			words.push_back(pre_seq->prefix[i]);
	}
	pre_seq->prefix = words;
	pre_seq_list_.push_back(pre_seq);
}

void CTCDecoder::BeamSearchNaive(const Matrix<BaseFloat> &loglikes) {
	// decode one utterance
	int nframe = loglikes.NumRows();
	int vocab_size = loglikes.NumCols();
	PrefixSeq *preseq, *n_preseq;
	std::vector<int> n_prefix;
	Vector<BaseFloat> *lmlogp;
	LstmlmHistroy *his;
	float logp, n_p_b, n_p_nb;
	int end_t;

	InitDecoding();
	// decode one utterance
	for (int n = 0; n < nframe; n++) {
		if (config_.lm_scale > 0.0) {
            for (auto &seq : beam_) {
				preseq = seq.second;
				lmlogp = MallocPred();
				his = MallocHis();
				lstmlm_->Forward(preseq->prefix.back(), *preseq->lmhis, lmlogp, his);
				lmlogp->ApplyLog();
				next_his_[preseq->prefix] = his;
				next_logprob_[preseq->prefix] = lmlogp;
			}
		}

		for (int k = 0; k < vocab_size; k++) {
			logp = loglikes(n, k);

            // The variables p_b and p_nb are respectively the
            // probabilities for the prefix given that it ends in a
            // blank and does not end in a blank at this time step.
            for (auto &seq : beam_) { // Loop over beam
				preseq = seq.second;

				// If we propose a blank the prefix doesn't change.
				// Only the probability of ending in blank gets updated.
				if (k == config_.blank) {
					auto it = next_beam_.find(preseq->prefix);
					if (it == next_beam_.end()) {
						n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix);
						CopyHis(preseq->lmhis);
					} else {
                        n_preseq  = it->second;
				    }

					n_p_b = LogAdd(n_preseq->logp_blank,
							LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp));
					n_preseq->logp_blank = n_p_b;
					next_beam_[n_preseq->prefix] = n_preseq;
					continue;
				}

				// Extend the prefix by the new character s and add it to
				// the beam. Only the probability of not ending in blank
				// gets updated.
				end_t = preseq->prefix.back();
				n_prefix = preseq->prefix;
				n_prefix.push_back(k);
				auto it = next_beam_.find(n_prefix);
				if (it == next_beam_.end()) {
					n_preseq = new PrefixSeq(n_prefix);
				} else {
                    n_preseq  = it->second;
				}

				if (k != end_t) {
					n_p_nb = LogAdd(n_preseq->logp_nblank,
							LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp));
				} else {
					// We don't include the previous probability of not ending
					// in blank (p_nb) if s is repeated at the end. The CTC
					// algorithm merges characters not separated by a blank.
					n_p_nb = LogAdd(n_preseq->logp_nblank, preseq->logp_blank+logp);
				}

				// *NB* this would be a good place to include an LM score.
				if (config_.lm_scale > 0.0) {
					lmlogp = next_logprob_[preseq->prefix];
					his = next_his_[preseq->prefix];
					n_preseq->logp_nblank = n_p_nb + config_.lm_scale*(*lmlogp)(k);
					n_preseq->lmhis = his;
					CopyHis(his);
				} else {
					n_preseq->logp_nblank = n_p_nb;
				}
				next_beam_[n_preseq->prefix] = n_preseq;

				// If s is repeated at the end we also update the unchanged
				// prefix. This is the merging case.
				if (k == end_t) {
					auto it = next_beam_.find(preseq->prefix);
					if (it == next_beam_.end()) {
						n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix);
						CopyHis(preseq->lmhis);
					} else {
                        n_preseq  = it->second;
                    }

					n_p_nb = LogAdd(n_preseq->logp_nblank, preseq->logp_nblank+logp);
					n_preseq->logp_nblank = n_p_nb;
					next_beam_[n_preseq->prefix] = n_preseq;
				}
			}
		}


		// Sort and trim the beam before moving on to the next time-step.
		pre_seq_list_.clear();
		for (auto it = next_beam_.begin(); it != next_beam_.end(); it++) {
			preseq = it->second;
			pre_seq_list_.push_back(preseq);
		}
		pre_seq_list_.sort(CTCDecoderUtil::compare_PrefixSeq_reverse);

		// free memory
		FreeBeam(&beam_);
		beam_.clear();
		int idx = 0;
		for (auto it = pre_seq_list_.begin(); it != pre_seq_list_.end(); it++) {
		    preseq = *it;
			if (idx < config_.beam) {
				beam_[preseq->prefix] = preseq;
			} else {
				FreeSeq(preseq);
                delete preseq;
			}
			idx++;
		}
		next_beam_.clear();

		if (config_.lm_scale > 0.0) {
			for (auto &seq : next_his_)
				FreeHis(seq.second);
			for (auto &seq : next_logprob_)
				FreePred(seq.second);
			next_his_.clear();
			next_logprob_.clear();
		}
	}
}

void CTCDecoder::BeamSearch(const Matrix<BaseFloat> &loglikes) {
	// decode one utterance
	int nframe = loglikes.NumRows();
	int likes_size = loglikes.NumCols();
	int vocab_size = config_.vocab_size;
	PrefixSeq *preseq, *n_preseq;
	std::vector<int> n_prefix, prefix;
	std::vector<float> next_words(vocab_size);
	Vector<BaseFloat> *lmlogp;
	LstmlmHistroy *his = NULL;
    std::vector<BaseFloat> next_step(likes_size);
    std::vector<int> in_words;
    std::vector<Vector<BaseFloat>*> nnet_out;
    std::vector<LstmlmHistroy*> context_in, context_out;
	float logp, logp_b, n_p_b, n_p_nb,
			ngram_logp = 0, rnnlm_logp = 0, sub_ngram_logp = 0,
            rscale = config_.rnnlm_scale;
	int end_t, bz, index;
    bool uselm;

    InitDecoding();
	// decode one utterance
	for (int n = 0; n < nframe; n++) {
		logp_b = loglikes(n, config_.blank);

		// Lstm language model process, beam streams parallel.
        uselm = false;
        if (config_.lm_scale > 0.0) {
           if (config_.blank_threshold <= 0)
                uselm = true;
           else if (config_.blank_threshold > 0 && Exp(logp_b) <= config_.blank_threshold)
                uselm = true;
        }

		if (uselm && rscale != 0) {
			in_words.clear();
			nnet_out.clear();
			context_in.clear();
			context_out.clear();

            // always padding to beam streams
            bz = 0;
            auto it = beam_.begin();
            while (bz < config_.beam) {
				preseq = it->second;
				lmlogp = MallocPred();
				his = MallocHis();
				in_words.push_back(preseq->prefix.back());
				context_in.push_back(preseq->lmhis);
				context_out.push_back(his);
				nnet_out.push_back(lmlogp);
                bz++;
                if (bz < beam_.size()) it++;
			}

            // beam streams parallel process
            lstmlm_->ForwardMseq(in_words, context_in, nnet_out, context_out);

            // get the valid streams
            for (bz = 0, it = beam_.begin(); bz < config_.beam; bz++) {
                if (bz < beam_.size()) {
				    preseq = it->second;
            	    nnet_out[bz]->ApplyLog();
            	    next_his_[preseq->prefix] = context_out[bz];
            	    next_logprob_[preseq->prefix] = nnet_out[bz];
                    it++;
                } else {
                    FreePred(nnet_out[bz]);
                    FreeHis(context_out[bz]);
                }
            }
		}

		std::fill(next_words.begin(), next_words.end(), 0);
		// blank pruning
		if (config_.blank_threshold > 0 && Exp(logp_b) > config_.blank_threshold) {
			next_words[config_.blank] = logp_b + log(config_.blank_penalty);
		} else if (config_.am_topk > 0) {
			// Top K pruning, the nth bigest words
            memcpy(&next_step.front(), loglikes.RowData(n), next_step.size()*sizeof(BaseFloat));
            std::nth_element(next_step.begin(), next_step.begin()+config_.am_topk, next_step.end(), std::greater<BaseFloat>());
            for (int k = 0; k < likes_size; k++) {
            	logp = loglikes(n, k);
                if (k == config_.blank) logp += log(config_.blank_penalty); // -2.30259
            	// top K pruning
            	if (logp > next_step[config_.am_topk]) {
            		if (!use_pinyin_) {
            			next_words[k] = logp;
            		} else {
            			for (int i = 0; i < pinyin2words_[k].size(); i++)
            				next_words[pinyin2words_[k][i]] = logp;
            		}
            	}
            }
        }

        // For each word
		for (int k = 0; k < vocab_size; k++) {
			logp = next_words[k];
            if (logp == 0) continue;

            // The variables p_b and p_nb are respectively the
            // probabilities for the prefix given that it ends in a
            // blank and does not end in a blank at this time step.
            for (auto &seq : beam_) { // Loop over beam
				preseq = seq.second;

				// If we propose a blank the prefix doesn't change.
				// Only the probability of ending in blank gets updated.
				if (k == config_.blank) {
					auto it = next_beam_.find(preseq->prefix);
					if (it == next_beam_.end()) {
					#if HAVE_KENLM == 1
						if (config_.use_kenlm) {
							n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix,
									preseq->ken_state, preseq->sub_ken_state);
						} else
					#endif
						n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix);
						CopyHis(preseq->lmhis);
					} else {
                        n_preseq  = it->second;
				    }

					n_p_b = LogAdd(n_preseq->logp_blank,
							LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp));
					n_preseq->logp_blank = n_p_b;
                    n_preseq->logp_lm = preseq->logp_lm;
                    n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_p_b, n_preseq->logp_nblank);
					next_beam_[n_preseq->prefix] = n_preseq;
					continue;
				}


				// Extend the prefix by the new character s and add it to
				// the beam. Only the probability of not ending in blank
				// gets updated.
				end_t = preseq->prefix.back();
				n_prefix = preseq->prefix;
				n_prefix.push_back(k);
				auto it = next_beam_.find(n_prefix);
				if (it == next_beam_.end()) {
					n_preseq = new PrefixSeq(n_prefix);
				} else {
                    n_preseq  = it->second;
				}

				if (k != end_t) {
					n_p_nb = LogAdd(n_preseq->logp_nblank,
							LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp));
				} else {
					// We don't include the previous probability of not ending
					// in blank (p_nb) if s is repeated at the end. The CTC
					// algorithm merges characters not separated by a blank.
					n_p_nb = LogAdd(n_preseq->logp_nblank, preseq->logp_blank+logp);
				}

				// *NB* this would be a good place to include an LM score.
				// if (config_.lm_scale > 0.0 && it == next_beam_.end()) {
				if (config_.lm_scale > 0.0) {
                    // rnn lm score
					if (rscale != 0) {
						lmlogp = next_logprob_[preseq->prefix];
						his = next_his_[preseq->prefix];
						n_preseq->lmhis = his;
						CopyHis(his);
                        rnnlm_logp = (*lmlogp)(k);
					}
                    // ngram lm score
                    if (rscale < 1.0) {
					#if HAVE_KENLM == 1
                        if (config_.use_kenlm) {
                        	index = kenlm_vocab_->Index(wordid_to_word_[k]);
                        	ngram_logp = kenlm_arpa_->Score(preseq->ken_state, index, n_preseq->ken_state);
                            n_preseq->sub_ken_state.resize(sub_kenlm_apra_.size());
                        	for (int i = 0; i < sub_kenlm_apra_.size(); i++) {
                        		index = sub_kenlm_vocab_[i]->Index(wordid_to_word_[k]);
                        		sub_ngram_logp = sub_kenlm_apra_[i]->Score(preseq->sub_ken_state[i], index, n_preseq->sub_ken_state[i]);
                        		ngram_logp = LogAdd(ngram_logp, sub_ngram_logp);
                        	}
							// Convert to natural log.
							ngram_logp *= M_LN10;
                        } else if (const_arpa_ != NULL)
					#endif
                        {
                            prefix = preseq->prefix;
                            prefix[0] = config_.sos; // <s>
                            ngram_logp = const_arpa_->GetNgramLogprob(k, prefix);
                            for (int i = 0; i < sub_const_arpa_.size(); i++) {
                        	    sub_ngram_logp = sub_const_arpa_[i]->GetNgramLogprob(k, prefix);
                        	    ngram_logp = LogAdd(ngram_logp, sub_ngram_logp);
                            }
                        }
                    }
                    // fusion score
					// n_preseq->logp_nblank = n_p_nb + config_.lm_scale*(rscale*rnnlm_logp + (1.0-rscale)*ngram_logp);
                    float logp_lm = config_.lm_scale*Log(rscale*Exp(rnnlm_logp) + (1.0-rscale)*Exp(ngram_logp));
                    n_preseq->logp_lm = preseq->logp_lm + logp_lm;
				}
				n_preseq->logp_nblank = n_p_nb;
				n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_preseq->logp_blank, n_p_nb);
				next_beam_[n_preseq->prefix] = n_preseq;


				// If s is repeated at the end we also update the unchanged
				// prefix. This is the merging case.
				if (k == end_t) {
					auto it = next_beam_.find(preseq->prefix);
					if (it == next_beam_.end()) {
					#if HAVE_KENLM == 1
						if (config_.use_kenlm) {
							n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix,
									preseq->ken_state, preseq->sub_ken_state);
						} else
					#endif
						n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix);
						CopyHis(preseq->lmhis);
					} else {
                        n_preseq  = it->second;
                    }

					n_p_nb = LogAdd(n_preseq->logp_nblank, preseq->logp_nblank+logp);
					n_preseq->logp_nblank = n_p_nb;
                    n_preseq->logp_lm = preseq->logp_lm;
                    n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_preseq->logp_blank, n_p_nb);
					next_beam_[n_preseq->prefix] = n_preseq;
				}
			}
		}


		// Sort and trim the beam before moving on to the next time-step.
		pre_seq_list_.clear();
		for (auto it = next_beam_.begin(); it != next_beam_.end(); it++) {
			preseq = it->second;
			pre_seq_list_.push_back(preseq);
		}
		pre_seq_list_.sort(CTCDecoderUtil::compare_PrefixSeq_reverse);

		// free memory
		FreeBeam(&beam_);
		beam_.clear();
		int idx = 0;
		for (auto it = pre_seq_list_.begin(); it != pre_seq_list_.end(); it++) {
		    preseq = *it;
			if (idx < config_.beam) {
				beam_[preseq->prefix] = preseq;
			} else {
				FreeSeq(preseq);
                delete preseq;
			}
			idx++;
		}

		next_beam_.clear();

		if (config_.lm_scale > 0.0) {
			for (auto &seq : next_his_)
				FreeHis(seq.second);
			for (auto &seq : next_logprob_)
				FreePred(seq.second);
			next_his_.clear();
			next_logprob_.clear();
		}
	}

    /*
    int size = pre_seq_list_.size();
    if (size > config_.beam) size = config_.beam;
    pre_seq_list_.resize(size);
    pre_seq_list_.sort(CTCDecoderUtil::compare_PrefixSeq_penalty_reverse);
    */
}

void CTCDecoder::BeamSearchTopk(const Matrix<BaseFloat> &loglikes) {
	// decode one utterance
	int nframe = loglikes.NumRows();
	int likes_size = loglikes.NumCols();
	int vocab_size = config_.vocab_size;
	PrefixSeq *preseq, *n_preseq;
	std::vector<int> n_prefix, prefix;
	std::vector<float> next_words(vocab_size);
	Vector<BaseFloat> *lmlogp;
	LstmlmHistroy *his = NULL;
    std::vector<BaseFloat> next_step(likes_size);
    std::vector<int> in_words;
    std::vector<Vector<BaseFloat>*> nnet_out;
    std::vector<LstmlmHistroy*> context_in, context_out;
	float logp, logp_b, n_p_b, n_p_nb,
			ngram_logp = 0, rnnlm_logp = 0, sub_ngram_logp = 0,
            rscale = config_.rnnlm_scale;
	int end_t, bz, index, blankid = 0;
    bool uselm;
    
    //loglikes.ColRange(0, 1).Add(-2.30259);

    InitDecoding();
	// decode one utterance
	for (int n = 0; n < nframe; n++) {

		logp_b = loglikes(n, blankid);
		// Lstm language model process, beam streams parallel.
        uselm = false;
        if (config_.lm_scale > 0.0) {
           if (config_.blank_threshold <= 0)
                uselm = true;
           else if (config_.blank_threshold > 0 && Exp(logp_b) <= config_.blank_threshold)
                uselm = true;
        }

		if (uselm && rscale != 0) {
			in_words.clear();
			nnet_out.clear();
			context_in.clear();
			context_out.clear();

            // always padding to beam streams
            bz = 0;
            auto it = beam_.begin();
            while (bz < config_.beam) {
				preseq = it->second;
				lmlogp = MallocPred();
				his = MallocHis();
				in_words.push_back(preseq->prefix.back());
				context_in.push_back(preseq->lmhis);
				context_out.push_back(his);
				nnet_out.push_back(lmlogp);
                bz++;
                if (bz < beam_.size()) it++;
			}

            // beam streams parallel process
            lstmlm_->ForwardMseq(in_words, context_in, nnet_out, context_out);

            // get the valid streams
            for (bz = 0, it = beam_.begin(); bz < config_.beam; bz++) {
                if (bz < beam_.size()) {
				    preseq = it->second;
            	    nnet_out[bz]->ApplyLog();
            	    next_his_[preseq->prefix] = context_out[bz];
            	    next_logprob_[preseq->prefix] = nnet_out[bz];
                    it++;
                } else {
                    FreePred(nnet_out[bz]);
                    FreeHis(context_out[bz]);
                }
            }
		}

		std::fill(next_words.begin(), next_words.end(), 0);
		// blank pruning
		if (config_.blank_threshold > 0 && Exp(logp_b) > config_.blank_threshold) {
			next_words[blankid] = logp_b + log(config_.blank_penalty); // -2.30259
		} else {
			// Top K pruning, the nth bigest words
			int topk = likes_size/2, key;
            for (int k = 1; k < topk; k++) {
            	logp = loglikes(n, k);
            	key = loglikes(n, topk+k);
                if (key == 0) logp += log(config_.blank_penalty); // -2.30259
				if (key < vocab_size && key >= 0) {
        			if (!use_pinyin_) {
        				next_words[key] = logp;
        			} else {
        				for (int i = 0; i < pinyin2words_[key].size(); i++)
        					next_words[pinyin2words_[key][i]] = logp;
        			}
				} /*else {
                    KALDI_ERR << "topk key " << key << " out of range [0, " << vocab_size << ")";
                }*/
            }
        }

        // For each word
		for (int k = 0; k < vocab_size; k++) {
			logp = next_words[k];
            if (logp == 0) continue;

            // The variables p_b and p_nb are respectively the
            // probabilities for the prefix given that it ends in a
            // blank and does not end in a blank at this time step.
            for (auto &seq : beam_) { // Loop over beam
				preseq = seq.second;

				// If we propose a blank the prefix doesn't change.
				// Only the probability of ending in blank gets updated.
				if (k == config_.blank) {
					auto it = next_beam_.find(preseq->prefix);
					if (it == next_beam_.end()) {
					#if HAVE_KENLM == 1
						if (config_.use_kenlm) {
							n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix,
									preseq->ken_state, preseq->sub_ken_state);
						} else
					#endif
						n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix);
						CopyHis(preseq->lmhis);
					} else {
                        n_preseq  = it->second;
				    }

					n_p_b = LogAdd(n_preseq->logp_blank,
							LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp));
					n_preseq->logp_blank = n_p_b;
                    n_preseq->logp_lm = preseq->logp_lm;
                    n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_p_b, n_preseq->logp_nblank);
					next_beam_[n_preseq->prefix] = n_preseq;
					continue;
				}


				// Extend the prefix by the new character s and add it to
				// the beam. Only the probability of not ending in blank
				// gets updated.
				end_t = preseq->prefix.back();
				n_prefix = preseq->prefix;
				n_prefix.push_back(k);
				auto it = next_beam_.find(n_prefix);
				if (it == next_beam_.end()) {
					n_preseq = new PrefixSeq(n_prefix);
				} else {
                    n_preseq  = it->second;
				}

				if (k != end_t) {
					n_p_nb = LogAdd(n_preseq->logp_nblank,
							LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp));
				} else {
					// We don't include the previous probability of not ending
					// in blank (p_nb) if s is repeated at the end. The CTC
					// algorithm merges characters not separated by a blank.
					n_p_nb = LogAdd(n_preseq->logp_nblank, preseq->logp_blank+logp);
				}

				// *NB* this would be a good place to include an LM score.
				// if (config_.lm_scale > 0.0 && it == next_beam_.end()) {
				if (config_.lm_scale > 0.0) {
                    // rnn lm score
					if (rscale != 0) {
						lmlogp = next_logprob_[preseq->prefix];
						his = next_his_[preseq->prefix];
						n_preseq->lmhis = his;
						CopyHis(his);
                        rnnlm_logp = (*lmlogp)(k);
					}
                    // ngram lm score
                    if (rscale < 1.0) {
					#if HAVE_KENLM == 1
                        if (config_.use_kenlm) {
                        	index = kenlm_vocab_->Index(wordid_to_word_[k]);
                        	ngram_logp = kenlm_arpa_->Score(preseq->ken_state, index, n_preseq->ken_state);
                            n_preseq->sub_ken_state.resize(sub_kenlm_apra_.size());
                        	for (int i = 0; i < sub_kenlm_apra_.size(); i++) {
                        		index = sub_kenlm_vocab_[i]->Index(wordid_to_word_[k]);
                        		sub_ngram_logp = sub_kenlm_apra_[i]->Score(preseq->sub_ken_state[i], index, n_preseq->sub_ken_state[i]);
                        		ngram_logp = LogAdd(ngram_logp, sub_ngram_logp);
                        	}
							// Convert to natural log.
							ngram_logp *= M_LN10;
                        } else if (const_arpa_ != NULL)
					#endif
                        {
                            prefix = preseq->prefix;
                            prefix[0] = config_.sos; // <s>
                            ngram_logp = const_arpa_->GetNgramLogprob(k, prefix);
                            for (int i = 0; i < sub_const_arpa_.size(); i++) {
                        	    sub_ngram_logp = sub_const_arpa_[i]->GetNgramLogprob(k, prefix);
                        	    ngram_logp = LogAdd(ngram_logp, sub_ngram_logp);
                            }
                        }
                    }
                    // fusion score
					// n_preseq->logp_nblank = n_p_nb + config_.lm_scale*(rscale*rnnlm_logp + (1.0-rscale)*ngram_logp);
                    float logp_lm = config_.lm_scale*Log(rscale*Exp(rnnlm_logp) + (1.0-rscale)*Exp(ngram_logp));
                    n_preseq->logp_lm = preseq->logp_lm + logp_lm;
				}
				n_preseq->logp_nblank = n_p_nb;
				n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_preseq->logp_blank, n_p_nb);
				next_beam_[n_preseq->prefix] = n_preseq;


				// If s is repeated at the end we also update the unchanged
				// prefix. This is the merging case.
				if (k == end_t) {
					auto it = next_beam_.find(preseq->prefix);
					if (it == next_beam_.end()) {
					#if HAVE_KENLM == 1
						if (config_.use_kenlm) {
							n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix,
									preseq->ken_state, preseq->sub_ken_state);
						} else
					#endif
						n_preseq = new PrefixSeq(preseq->lmhis, preseq->prefix);
						CopyHis(preseq->lmhis);
					} else {
                        n_preseq  = it->second;
                    }

					n_p_nb = LogAdd(n_preseq->logp_nblank, preseq->logp_nblank+logp);
					n_preseq->logp_nblank = n_p_nb;
                    n_preseq->logp_lm = preseq->logp_lm;
                    n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_preseq->logp_blank, n_p_nb);
					next_beam_[n_preseq->prefix] = n_preseq;
				}
			}
		}


		// Sort and trim the beam before moving on to the next time-step.
		pre_seq_list_.clear();
		for (auto it = next_beam_.begin(); it != next_beam_.end(); it++) {
			preseq = it->second;
			pre_seq_list_.push_back(preseq);
		}
		pre_seq_list_.sort(CTCDecoderUtil::compare_PrefixSeq_reverse);

		// free memory
		FreeBeam(&beam_);
		beam_.clear();
		int idx = 0;
		for (auto it = pre_seq_list_.begin(); it != pre_seq_list_.end(); it++) {
		    preseq = *it;
			if (idx < config_.beam) {
				beam_[preseq->prefix] = preseq;
			} else {
				FreeSeq(preseq);
                delete preseq;
			}
			idx++;
		}

		next_beam_.clear();

		if (config_.lm_scale > 0.0) {
			for (auto &seq : next_his_)
				FreeHis(seq.second);
			for (auto &seq : next_logprob_)
				FreePred(seq.second);
			next_his_.clear();
			next_logprob_.clear();
		}
	}

    /*
    int size = pre_seq_list_.size();
    if (size > config_.beam) size = config_.beam;
    pre_seq_list_.resize(size);
    pre_seq_list_.sort(CTCDecoderUtil::compare_PrefixSeq_penalty_reverse);
    */
}

/*
void CTCDecoder::BeamMerge(std::vector<PrefixSeq*> &merge_beam) {
	KALDI_ASSERT(next_beam_size_%cur_beam_size_ == 0);
	BeamType beam;
    merge_beam.clear();
	PrefixSeq *preseq, *n_preseq;
	for (int i = 0; i < next_beam_size_; i++) {
		auto it = beam.find(next_beam_easy_[i]->prefix);
		if (it != beam.end()) {
			preseq = it->second;
			n_preseq = next_beam_easy_[i];
			n_preseq->logp_nblank = LogAdd(preseq->logp_nblank, n_preseq->logp_nblank);
			n_preseq->logp_blank = LogAdd(preseq->logp_blank, n_preseq->logp_blank);
			n_preseq->logp = preseq->logp_lm + LogAdd(n_preseq->logp_blank, n_preseq->logp_nblank);
		    beam[n_preseq->prefix] = n_preseq;
            delete preseq;
        }
		beam[next_beam_easy_[i]->prefix] = next_beam_easy_[i];
	}

	for (auto &seq : beam) {
		merge_beam.push_back(seq.second);
	}
}
*/

/*
void CTCDecoder::BeamSearchEasyTopk(const Matrix<BaseFloat> &loglikes) {
	// decode one utterance
	int nframe = loglikes.NumRows();
	int likes_size = loglikes.NumCols();
	int vocab_size = config_.vocab_size;
	PrefixSeq *preseq, *n_preseq;
	std::vector<int> prefix, n_prefix;
	std::vector<float> next_words(vocab_size);
	std::vector<BaseFloat> next_step(likes_size);
	float logp, logp_b, logp_lm, n_p_b, n_p_nb;
	float ngram_logp = 0, rnnlm_logp = 0, sub_ngram_logp = 0,
			rscale = config_.rnnlm_scale;
	int end_t, index, topk = config_.am_topk, key;

	InitEasyDecoding(topk);

	// decode one utterance
	for (int n = 0; n < nframe; n++) {
		logp_b = loglikes(n, config_.blank);
        cur_beam_size_ = beam_easy_.size();

		// blank pruning
		// Only the probability of ending in blank gets updated.
		if (config_.blank_threshold > 0 && Exp(logp_b) > config_.blank_threshold) {
			logp = logp_b + log(config_.blank_penalty); // -2.30259
			for (int i = 0; i < cur_beam_size_; i++) {
				preseq = beam_easy_[i];
				n_p_b = LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp);
				preseq->logp_blank = n_p_b;
			    preseq->logp = preseq->logp_lm +  LogAdd(n_p_b, preseq->logp_nblank);
			}
			continue;
		}

		std::fill(next_words.begin(), next_words.end(), -50);
		// blank pruning
		if (config_.blank_threshold > 0 && Exp(logp_b) > config_.blank_threshold) {
			next_words[config_.blank] = logp_b + log(config_.blank_penalty);
		} else if (config_.am_topk > 0) {
			// Top K pruning, the nth bigest words
            memcpy(&next_step.front(), loglikes.RowData(n), next_step.size()*sizeof(BaseFloat));
            std::nth_element(next_step.begin(), next_step.begin()+config_.am_topk, next_step.end(), std::greater<BaseFloat>());
            for (int k = 0; k < likes_size; k++) {
            	logp = loglikes(n, k);
                if (k == config_.blank) logp += log(config_.blank_penalty); // -2.30259
            	// top K pruning
            	if (logp > next_step[config_.am_topk]) {
            		if (!use_pinyin_) {
            			next_words[k] = logp;
            		} else {
            			for (int i = 0; i < pinyin2words_[k].size(); i++)
            				next_words[pinyin2words_[k][i]] = logp;
            		}
            	}
            }
        }


		/// produce next beam
		/// not extended
        next_beam_easy_.clear();
		for (int i = 0; i < cur_beam_size_; i++) {
			preseq = beam_easy_[i];
			end_t = preseq->PrefixBack();

			// blank
			logp = logp_b + log(config_.blank_penalty); // -2.30259
			n_p_b = LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp);

			// If s is repeated at the end we also update the unchanged
			// prefix. This is the merging case.
			n_p_nb = kLogZeroFloat;
			if (end_t != config_.blank) {
				logp = next_words[end_t];
				n_p_nb = preseq->logp_nblank+logp;
			}

			n_preseq = new PrefixSeq();
			*n_preseq = *preseq;
			n_preseq->logp_blank = n_p_b;
			n_preseq->logp_nblank = n_p_nb;
			n_preseq->logp = preseq->logp_lm +  LogAdd(n_p_b, n_p_nb);
            next_beam_easy_.push_back(n_preseq);
		}

		/// extended
		for (int k = 0; k < vocab_size; k++) {
			key = k;
			logp = next_words[k];
			if (key == config_.blank || logp == -50)
				continue;

			// for each beam appending word key
			for (int i = 0; i < cur_beam_size_; i++) {
				preseq = beam_easy_[i];
				end_t = preseq->PrefixBack();
			    n_preseq = new PrefixSeq();

				n_p_b = kLogZeroFloat;
				if (key != end_t) {
					n_p_nb = LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp);
				} else {
					// We don't include the previous probability of not ending
					// in blank (p_nb) if s is repeated at the end. The CTC
					// algorithm merges characters not separated by a blank.
					n_p_nb = preseq->logp_blank+logp;
				}

				// *NB* this would be a good place to include an LM score.
				if (config_.lm_scale > 0.0) {
					// ngram lm score
					if (rscale < 1.0) {
					#if HAVE_KENLM == 1
						if (config_.use_kenlm) {
							index = kenlm_vocab_->Index(wordid_to_word_[key]);
							ngram_logp = kenlm_arpa_->Score(preseq->ken_state, index, n_preseq->ken_state);
							n_preseq->sub_ken_state.resize(sub_kenlm_apra_.size());
							for (int i = 0; i < sub_kenlm_apra_.size(); i++) {
								index = sub_kenlm_vocab_[i]->Index(wordid_to_word_[key]);
								sub_ngram_logp = sub_kenlm_apra_[i]->Score(preseq->sub_ken_state[i], index, n_preseq->sub_ken_state[i]);
								ngram_logp = LogAdd(ngram_logp, sub_ngram_logp);
							}
							// Convert to natural log.
							ngram_logp *= M_LN10;
						} else
					#endif
                        {
						    prefix = preseq->prefix;
						    prefix[0] = config_.sos; // <s>
						    ngram_logp = const_arpa_->GetNgramLogprob(key, prefix);
						    for (int i = 0; i < sub_const_arpa_.size(); i++) {
							    sub_ngram_logp = sub_const_arpa_[i]->GetNgramLogprob(key, prefix);
							    ngram_logp = LogAdd(ngram_logp, sub_ngram_logp);
						    }
                        }
					}
					// fusion score
					logp_lm = config_.lm_scale*Log(rscale*Exp(rnnlm_logp) + (1.0-rscale)*Exp(ngram_logp));
				}

                n_preseq->prefix = preseq->prefix;
				n_preseq->PrefixAppend(key);
				n_preseq->logp_blank = n_p_b;
				n_preseq->logp_nblank = n_p_nb;
				n_preseq->logp_lm = preseq->logp_lm + logp_lm;
				n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_p_b, n_p_nb);
                next_beam_easy_.push_back(n_preseq);
			}
		}

        next_beam_size_ = next_beam_easy_.size(); 
		std::vector<PrefixSeq*> valid_beam;
		BeamMerge(valid_beam);
		// select best TopN beam
		std::sort(valid_beam.begin(), valid_beam.end(), CTCDecoderUtil::compare_PrefixSeq_reverse);

        for (int i = 0; i < beam_easy_.size(); i++)
            delete beam_easy_[i];
		int size = valid_beam.size();
		cur_beam_size_ = size >= config_.beam ? config_.beam : size;
        beam_easy_ = std::vector<PrefixSeq*>(valid_beam.begin(), valid_beam.begin() + cur_beam_size_);
		for (int i = cur_beam_size_; i < size; i++)
                delete valid_beam[i];
		size++;
	}
}
*/

void CTCDecoder::BeamMerge(std::vector<PrefixSeq*> &merge_beam, bool skip_blank) {
	KALDI_ASSERT(next_beam_size_%cur_beam_size_ == 0);
	BeamType beam;

    if (skip_blank) {
	    for (int i = 0; i < cur_beam_size_; i++)
            merge_beam.push_back(&beam_easy_[i]);
        return ;
    }

	for (int i = 0; i < cur_beam_size_; i++) {
		beam[next_beam_easy_[i].prefix] = &next_beam_easy_[i];
	}

	PrefixSeq *preseq, *n_preseq;
	for (int i = cur_beam_size_; i < next_beam_size_; i++) {
		n_preseq = &next_beam_easy_[i];
		auto it = beam.find(n_preseq->prefix);
		if (it != beam.end()) {
			preseq = it->second;
			n_preseq->logp_nblank = LogAdd(preseq->logp_nblank, n_preseq->logp_nblank);
			n_preseq->logp_blank = LogAdd(preseq->logp_blank, n_preseq->logp_blank);
			n_preseq->logp = n_preseq->logp_lm + LogAdd(n_preseq->logp_blank, n_preseq->logp_nblank);
			//n_preseq->logp = LogAdd(n_preseq->logp_blank, n_preseq->logp_nblank);
			beam.erase(it);
		}
		merge_beam.push_back(n_preseq);
	}

	for (auto &seq : beam) {
		merge_beam.push_back(seq.second);
	}
}

int CTCDecoder::ProcessKeywordsTopk(const Matrix<BaseFloat> &loglikes) {
	int nframe = loglikes.NumRows();
	int likes_size = loglikes.NumCols();
	int topk = likes_size/2, last_id, id,
		ksize, cut_frame, ks;
	float logp, logp_b;

	for (int i = 0; i < keywords_.size(); i++) {
		ksize = keywords_[i].size();
	    cut_frame = 0;
	    last_id = config_.blank;
	    std::vector<int> kws;
	    std::vector<int> kws_tim;
		for (int n = 0; n < nframe; n++) {
			logp_b = loglikes(n, config_.blank);
			if (Exp(logp_b) > config_.blank_threshold) {
				last_id = config_.blank;
				continue;
			}

			// max logp and id
			logp = loglikes(n, 0);
			id = loglikes(n, topk);
			for (int k = 1; k < topk; k++) {
				if (logp < loglikes(n, k)) {
					logp = loglikes(n, k);
					id = loglikes(n, topk+k);
				}
			}

			if (id != config_.blank && id != last_id) {
				kws.push_back(id);
				kws_tim.push_back(n);
			}

			ks = kws_tim.size();
			if (id != config_.blank && id == last_id)
				kws_tim[ks-1] = n;

			if (ks > ksize || (ks > 0 && kws[ks-1] != keywords_[i][ks-1]))
				break;

			last_id = id;
		}

		if (kws.size() >= ksize && kws[ksize-1] == keywords_[i][ksize-1]) {
			keyword_ = keywords_[i];
			cut_frame = kws_tim[ksize-1]+1;
			break;
		} else {
            cut_frame = 0;
        }
	}
	return cut_frame;
}

void CTCDecoder::BeamSearchEasyTopk(const Matrix<BaseFloat> &loglikes) {
	// decode one utterance
	int nframe = loglikes.NumRows();
	int likes_size = loglikes.NumCols();
	int vocab_size = config_.vocab_size;
	PrefixSeq *preseq, *n_preseq;
	std::vector<int> prefix, n_prefix;
	std::vector<float> next_words(vocab_size);
	float logp = 0, logp_b = 0, logp_lm = 0, n_p_b, n_p_nb;
	float ngram_logp = kLogZeroFloat, rnnlm_logp = kLogZeroFloat, sub_ngram_logp = kLogZeroFloat,
			rscale = config_.rnnlm_scale,
            blank_penalty = log(config_.blank_penalty);
	int end_t, index, topk = likes_size/2, key, cur_his = 0, start = 0;
    bool skip_blank = false, uselm;

    // rnnlm
    std::vector<int> in_words;
    std::vector<Vector<BaseFloat>*> nnet_out;
    std::vector<LstmlmHistroy*> context_in, context_out;

	InitEasyDecoding(topk);

	if (config_.keywords != "")
		start = ProcessKeywordsTopk(loglikes);

	// decode one utterance
	for (int n = start; n < nframe; n++) {
		logp_b = loglikes(n, config_.blank);

		// Lstm language model process, beam streams parallel.
		uselm = false;
		if (config_.lm_scale > 0.0) {
		   if (config_.blank_threshold <= 0)
				uselm = true;
		   else if (config_.blank_threshold > 0 && Exp(logp_b) <= config_.blank_threshold)
				uselm = true;
		}

		if (uselm && rscale != 0) {
			in_words.clear();
			nnet_out.clear();
			context_in.clear();
			context_out.clear();

			// always padding to beam streams
			int i = 0, bz = 0;
			while (bz < config_.beam) {
				preseq = &beam_easy_[i];
				in_words.push_back(preseq->PrefixBack());
				context_in.push_back(preseq->lmhis);
				context_out.push_back(preseq->next_lmhis);
				nnet_out.push_back(preseq->lmlogp);
				bz++;
				if (bz < cur_beam_size_) i++;
			}

			// beam streams parallel process
			lstmlm_->ForwardMseq(in_words, context_in, nnet_out, context_out);

			// get the valid streams
			for (int i = 0; i < cur_beam_size_; i++) {
				preseq = &beam_easy_[i];
				preseq->lmlogp->ApplyLog();
			}
		}


		// blank pruning
		// Only the probability of ending in blank gets updated.
		if (config_.blank_threshold > 0 && Exp(logp_b) > config_.blank_threshold) {
			logp = logp_b + blank_penalty; // -2.30259
			for (int i = 0; i < cur_beam_size_; i++) {
				preseq = &beam_easy_[i];
				n_p_b = LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp);
                // n_p_b approximate 1, meanwhile n_p_nb = 0;
				n_p_nb = preseq->logp_nblank+kLogZeroFloat;
				preseq->logp_blank = n_p_b;
			    preseq->logp_nblank = n_p_nb;
			    preseq->logp = preseq->logp_lm +  LogAdd(n_p_b, preseq->logp_nblank);
			    //preseq->logp = LogAdd(n_p_b, preseq->logp_nblank);
			}
            skip_blank = true;
			continue;
		}

		std::fill(next_words.begin(), next_words.end(), kLogZeroFloat);
		// blank pruning
		if (config_.blank_threshold > 0 && Exp(logp_b) > config_.blank_threshold) {
			next_words[config_.blank] = logp_b + blank_penalty; // -2.30259
		} else {
			int nhit = 0;
			BaseFloat logsum = 0;
			for (int k = 1; k < config_.scene_topk; k++) {
				logp = loglikes(n, k);
				key = loglikes(n, topk+k);
				if (key != config_.blank && sceneword_[key] > 0) {
					logsum += Exp(logp);
					nhit++;
				}
			}

			if (nhit > 0) {
				for (int k = 1; k < config_.scene_topk; k++) {
					logp = loglikes(n, k);
					key = loglikes(n, topk+k);
					if (key == config_.blank) logp += blank_penalty; // -2.30259
					else if (sceneword_[key] > 0) logp = Log(Exp(logp)/logsum);
					if (key == config_.blank || sceneword_[key] > 0)
						next_words[key] = logp;
				}
			} else {
				// Top K pruning, the nth bigest words
				for (int k = 1; k < topk; k++) {
					logp = loglikes(n, k);
					key = loglikes(n, topk+k);
					if (key == config_.blank) logp += blank_penalty; // -2.30259
					if (key < vocab_size && key >= 0) {
						if (!use_pinyin_) {
							next_words[key] = logp;
						} else {
							for (int i = 0; i < pinyin2words_[key].size(); i++)
								next_words[pinyin2words_[key][i]] = logp;
						}
					} /* else {
						KALDI_ERR << "topk key " << key << " out of range [0, " << vocab_size << ")";
					} */
				}
			}
        }


		/// produce next beam
		/// not extended
        next_beam_size_ = 0;
        skip_blank = false;
		for (int i = 0; i < cur_beam_size_; i++) {
			preseq = &beam_easy_[i];
			end_t = preseq->PrefixBack();
			n_preseq = &next_beam_easy_[next_beam_size_];
			next_beam_size_++;

			// blank
			logp = logp_b + blank_penalty; // -2.30259
			n_p_b = LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp);

			// If s is repeated at the end we also update the unchanged
			// prefix. This is the merging case.
			n_p_nb = kLogZeroFloat;
			if (end_t != config_.blank) {
				logp = next_words[end_t];
				n_p_nb = preseq->logp_nblank+logp;
			}

			*n_preseq = *preseq;
			n_preseq->logp_blank = n_p_b;
			n_preseq->logp_nblank = n_p_nb;
			n_preseq->logp = preseq->logp_lm +  LogAdd(n_p_b, n_p_nb);
			//n_preseq->logp = LogAdd(n_p_b, n_p_nb);
		}

		/// extended
		for (int k = 1; k < topk; k++) {
			key = loglikes(n, topk+k);
			//logp = loglikes(n, k);
            logp = next_words[key];
			if (key == config_.blank || key >= vocab_size || key < 0)
				continue;

			// for each beam appending word key
			for (int i = 0; i < cur_beam_size_; i++) {
				preseq = &beam_easy_[i];
				end_t = preseq->PrefixBack();
				n_preseq = &next_beam_easy_[next_beam_size_];
				next_beam_size_++;
				*n_preseq = *preseq;

				n_p_b = kLogZeroFloat;
				if (key != end_t) {
					n_p_nb = LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp);
				} else {
					// We don't include the previous probability of not ending
					// in blank (p_nb) if s is repeated at the end. The CTC
					// algorithm merges characters not separated by a blank.
					n_p_nb = preseq->logp_blank+logp;
				}

				// *NB* this would be a good place to include an LM score.
				if (config_.lm_scale > 0.0) {
                    // rnn lm score
					if (rscale != 0) {
                        rnnlm_logp = (*preseq->lmlogp)(key);
                        // rnnlm history
                        n_preseq->lmhis = preseq->next_lmhis;
                        // n_preseq->next_lmhis = preseq->lmhis;
					}

					// ngram lm score
					if (rscale < 1.0) {
					#if HAVE_KENLM == 1
						if (config_.use_kenlm && kenlm_arpa_ != NULL) {
							index = kenlm_vocab_->Index(wordid_to_word_[key]);
							ngram_logp = kenlm_arpa_->Score(preseq->ken_state, index, n_preseq->ken_state);
							n_preseq->sub_ken_state.resize(sub_kenlm_apra_.size());
							for (int i = 0; i < sub_kenlm_apra_.size(); i++) {
								index = sub_kenlm_vocab_[i]->Index(wordid_to_word_[key]);
								sub_ngram_logp = sub_kenlm_apra_[i]->Score(preseq->sub_ken_state[i], index, n_preseq->sub_ken_state[i]);
								ngram_logp = LogAdd(ngram_logp, sub_ngram_logp);
								//ngram_logp = std::max(ngram_logp, sub_ngram_logp);
							}
							// Convert to natural log.
							ngram_logp *= M_LN10;
						} else if (const_arpa_ != NULL)
					#endif
                        {
						    prefix = preseq->prefix;
						    prefix[0] = config_.sos; // <s>
						    ngram_logp = const_arpa_->GetNgramLogprob(key, prefix);
						    for (int i = 0; i < sub_const_arpa_.size(); i++) {
							    sub_ngram_logp = sub_const_arpa_[i]->GetNgramLogprob(key, prefix);
							    ngram_logp = LogAdd(ngram_logp, sub_ngram_logp);
						    }
                        }
					}
					// fusion score
					logp_lm = config_.lm_scale*Log(rscale*Exp(rnnlm_logp) + (1.0-rscale)*Exp(ngram_logp));
				}
                
				n_preseq->PrefixAppend(key);
				n_preseq->logp_blank = n_p_b;
				n_preseq->logp_nblank = n_p_nb;
				//n_preseq->logp_nblank = n_p_nb + logp_lm;
				n_preseq->logp_lm += logp_lm;
				n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_p_b, n_p_nb);
				//n_preseq->logp = LogAdd(n_p_b, n_p_nb);
			}
		}

		std::vector<PrefixSeq*> valid_beam;
		BeamMerge(valid_beam, skip_blank);
		// select best TopN beam
		std::sort(valid_beam.begin(), valid_beam.end(), CTCDecoderUtil::compare_PrefixSeq_reverse);
		int size = valid_beam.size();
		cur_beam_size_ = size >= config_.beam ? config_.beam : size;
		for (int i = 0; i < cur_beam_size_; i++) {
			beam_easy_[i] = *valid_beam[i];
			if (rscale != 0) {
                beam_easy_[i].lmlogp = &rnnlm_logp_[i];
                beam_easy_[i].next_lmhis = &rnnlm_his_[config_.beam*cur_his+i];
            }
        }
        cur_his = (cur_his+1)%2;
	}
}


} // end namespace kaldi.
