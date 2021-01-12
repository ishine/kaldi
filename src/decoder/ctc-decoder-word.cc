// decoder/ctc-decoder-word.cc

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
#include "decoder/ctc-decoder-word.h"

namespace kaldi {

std::string PrefixSeqWord::ToStr() {
	std::ostringstream out;
    out << "prefix = {";
    for (int i = 0; i < prefix_word.size(); i++)
        out << prefix_word[i] << ", ";
    out << "},";

    out << " score = " << logp << ", ctc_score = " << logp-logp_lm
                << ", lm_score = " << logp_lm << ", trie_node = " << trie_node
                << ", lmhis = " << lmhis << ", next_lmhis = " << next_lmhis;
   return out.str();
}

#if HAVE_KENLM == 1
CTCDecoderWord::CTCDecoderWord(CTCDecoderWordOptions &config,
			            KaldiLstmlmWrapper *lstmlm,
			            KenModel *kenlm_arpa,
						std::vector<KenModel *> &sub_kenlm_apra,
                        KenModel *rank_kenlm_arpa):
		config_(config), lstmlm_(lstmlm), 
        kenlm_arpa_(kenlm_arpa), sub_kenlm_apra_(sub_kenlm_apra), rank_kenlm_arpa_(rank_kenlm_arpa) {
	if (!word_trie_.LoadDict(config.word2bpeid_rxfilename))
		KALDI_ERR << "Could not read symbol table from file " << config_.word2bpeid_rxfilename;

	Initialize();

    kenlm_vocab_ = NULL;
    if (kenlm_arpa_ != NULL) {
        kenlm_vocab_ = &(kenlm_arpa_->GetVocabulary());
	    // sub language lmodel vocab
        int nsub = sub_kenlm_apra.size();
        sub_kenlm_vocab_.resize(nsub);
        for (int i = 0; i < nsub; i++)
            sub_kenlm_vocab_[i] = &(sub_kenlm_apra[i]->GetVocabulary());
    }
    if (rank_kenlm_arpa_ != NULL)
        rank_kenlm_vocab_ = &(rank_kenlm_arpa_->GetVocabulary());

	/// word symbols
	fst::SymbolTable *word_symbols = NULL;
	if (!(word_symbols = fst::SymbolTable::ReadText(config_.word2wordid_rxfilename)))
		KALDI_ERR << "Could not read symbol table from file " << config_.word2wordid_rxfilename;

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

	/// bpe symbols
	fst::SymbolTable *bpe_symbols = NULL;
	if (!(bpe_symbols = fst::SymbolTable::ReadText(config_.bpe2bpeid_rxfilename)))
		KALDI_ERR << "Could not read symbol table from file " << config_.bpe2bpeid_rxfilename;

	bpeid_to_bpe_.resize(bpe_symbols->NumSymbols());
	for (int32 i = 0; i < bpe_symbols->NumSymbols(); i++) {
		bpeid_to_bpe_[i] = bpe_symbols->Find(i);
		if (bpeid_to_bpe_[i] == "") {
		  KALDI_ERR << "Could not find bpe for integer " << i << "in the bpe "
			  << "symbol table, mismatched symbol table or you have discoutinuous "
			  << "integers in your symbol table?";
		}
	}
	for (int32 i = 0; i < bpe_symbols->NumSymbols(); i++) {
		if (bpe_symbols->Find(i) == "") {
		  KALDI_ERR << "Could not find bpe for integer " << i << "in the bpe "
			  << "symbol table, mismatched symbol table or you have discoutinuous "
			  << "integers in your symbol table?";
		}
		bpe_to_bpeid_[bpe_symbols->Find(i)] = i;
	}

    /// pinyin symbols
	use_pinyin_ = false;
	if (config_.pinyin2bpeid_rxfilename != "") {
		SequentialInt32VectorReader bpeid_reader(config_.pinyin2bpeid_rxfilename);
		for (; !bpeid_reader.Done(); bpeid_reader.Next()) {
			pinyin_to_bpe_.push_back(bpeid_reader.Value());
		}
		use_pinyin_ = true;
	}
}
#endif

void CTCDecoderWord::Initialize() {
	if (lstmlm_ != NULL) {
		rd_ = lstmlm_->GetRDim();
		cd_ = lstmlm_->GetCDim();
	}

	int beam_size = config_.beam + config_.word_beam;
	cur_beam_.resize(beam_size, PrefixSeqWord());
	cur_bpe_beam_size_ = 0;
	cur_word_beam_size_ = 0;

	// rnn lm
    if (config_.rnnlm_scale != 0) {
    	LstmlmHistroy his(rd_, cd_, kSetZero);
    	rnnlm_his_.resize(2*beam_size, his);
    	out_words_.resize(beam_size);
    	out_preseq_.resize(beam_size);
    	out_words_score_.resize(beam_size);
    	in_words_.reserve(beam_size);
    	rnnlm_his_in_.reserve(beam_size);
    	rnnlm_his_out_.reserve(beam_size);
        rnnlm_his_table_.resize(2*beam_size, false);
    }
}


void CTCDecoderWord::InitDecoding(int topk) {
    // Elements in the beam are (prefix, (p_blank, p_no_blank))
    // Initialize the beam with the empty sequence, a probability of
    // 1 for ending in blank and zero for ending in non-blank (in log space).
    int beam_size = config_.beam + config_.word_beam;
	if (config_.rnnlm_scale != 0) {
        std::fill(rnnlm_his_table_.begin(), rnnlm_his_table_.end(), false);
        for (int i = 0; i < beam_size; i++) {
    	    cur_beam_[i].lmhis = &rnnlm_his_[i];
    	    cur_beam_[i].next_lmhis = &rnnlm_his_[beam_size+i];
            cur_beam_[i].idx1 = i;
            cur_beam_[i].idx2 = beam_size+i;
            rnnlm_his_table_[i] = true;
            rnnlm_his_table_[beam_size+i] = true;
        }
    }

	PrefixSeqWord *seq = &cur_beam_[0];
    seq->Reset();
	seq->PrefixBpeAppend(config_.blank);
	seq->PrefixWordAppend(config_.sos);
	seq->logp_blank = 0.0;
	cur_bpe_beam_size_ = 1;
	cur_word_beam_size_ = 0;

	// next beam buffer
	next_bpe_beam_.resize(beam_size*topk, PrefixSeqWord());
	next_bpe_beam_size_ = 0;

	next_word_beam_.resize(beam_size*topk, PrefixSeqWord());
	next_word_beam_size_ = 0;

    //beam_union_.reserve(beam_size*topk);

	if (config_.rnnlm_scale != 0) {
    	// first input <s>
    	seq->lmhis->SetZero();

		for (int i = 0; i < beam_size; i++) {
			out_words_[i].reserve(topk);
			out_preseq_[i].reserve(topk);
			out_words_score_[i].reserve(topk);
		}
	}

#if HAVE_KENLM == 1
    if (kenlm_arpa_ != NULL)
	    seq->ken_state = kenlm_arpa_->BeginSentenceState();

    int nsub = sub_kenlm_apra_.size();
    seq->sub_ken_state.resize(nsub);
    for (int i = 0; i < nsub; i++)
        seq->sub_ken_state[i] = sub_kenlm_apra_[i]->BeginSentenceState();
#endif
}

bool CTCDecoderWord::GetBestPath(std::vector<int> &words, BaseFloat &logp, BaseFloat &logp_lm) {
    PrefixSeqWord *seq = &cur_beam_[0];

	if (seq == NULL) return false;

	logp = seq->logp;
    logp_lm = seq->logp_lm;


    int size = seq->prefix_word.size();
    words.resize(size);
	for (int i = 0; i < size; i++)
		words[i] = seq->prefix_word[i];

	return true;
}

void CTCDecoderWord::GreedySearch(const Matrix<BaseFloat> &loglikes) {
    int nframe = loglikes.NumRows(), k;
	int likes_size = loglikes.NumCols();
    int topk = likes_size/2;
    PrefixSeqWord *pre_seq;
    BaseFloat logp;

    InitDecoding(topk);
    // decode one utterance
    pre_seq = &cur_beam_[0];
    for (int n = 0; n < nframe; n++) {
        logp = loglikes.Row(n).Range(0, topk).Max(&k);
        k = loglikes(n, topk+k);
        pre_seq->logp_blank += logp;
        pre_seq->prefix_bpe.push_back(k);
    }
    std::vector<int> words;
    words.push_back(config_.blank);
    for (int i = 1; i < pre_seq->prefix_bpe.size(); i++) {
        if (pre_seq->prefix_bpe[i] != config_.blank && pre_seq->prefix_bpe[i] != pre_seq->prefix_bpe[i-1])
            words.push_back(pre_seq->prefix_bpe[i]);
    }
    pre_seq->prefix_bpe = words;
}


void CTCDecoderWord::BeamMerge(std::vector<PrefixSeqWord*> &bpe_beam,
		std::vector<PrefixSeqWord*> &word_beam, bool skip_blank) {
	//KALDI_ASSERT(next_beam_size_%cur_beam_size_ == 0);
	//BeamType beam;

    if (skip_blank) {
	    for (int i = 0; i < cur_bpe_beam_size_; i++)
	    	bpe_beam.push_back(&cur_beam_[i]);

		for (int i = cur_bpe_beam_size_; i < cur_bpe_beam_size_+cur_word_beam_size_; i++)
			word_beam.push_back(&cur_beam_[i]);
        return ;
    }

    /*
    bpe_beam.reserve(next_bpe_beam_size_);
	for (int i = 0; i < next_bpe_beam_size_; i++)
		bpe_beam.push_back(&next_bpe_beam_[i]);

	word_beam.reserve(next_word_beam_size_);
	for (int i = 0; i < next_word_beam_size_; i++)
		word_beam.push_back(&next_word_beam_[i]);
    */

    std::vector<int> prefix_key;
    prefix_key.reserve(PREFIX_BPE_MAX_LEN+PREFIX_WORD_MAX_LEN);
	beam_union_.clear();
	PrefixSeqWord *preseq, *n_preseq;
	for (int i = 0; i < next_bpe_beam_size_; i++) {
		n_preseq = &next_bpe_beam_[i];
        prefix_key.clear();
        prefix_key.insert(prefix_key.begin(), n_preseq->prefix_bpe.begin(), n_preseq->prefix_bpe.end());
        prefix_key.insert(prefix_key.end(), n_preseq->prefix_word.begin(), n_preseq->prefix_word.end());
		auto it = beam_union_.find(prefix_key);
	    if (it != beam_union_.end()) {
            preseq = it->second;
			preseq->logp_nblank = LogAdd(preseq->logp_nblank, n_preseq->logp_nblank);
			preseq->logp_blank = LogAdd(preseq->logp_blank, n_preseq->logp_blank);
			preseq->logp = preseq->logp_lm + LogAdd(preseq->logp_blank, preseq->logp_nblank);
        } else {
            beam_union_[prefix_key] = n_preseq;
        }
	}

	for (auto &seq : beam_union_) {
		bpe_beam.push_back(seq.second);
	}


	beam_union_.clear();
	for (int i = 0; i < next_word_beam_size_; i++) {
		n_preseq = &next_word_beam_[i];
		auto it = beam_union_.find(n_preseq->prefix_word);
		if (it != beam_union_.end()) {
            preseq = it->second;
			preseq->logp_nblank = LogAdd(preseq->logp_nblank, n_preseq->logp_nblank);
			preseq->logp_blank = LogAdd(preseq->logp_blank, n_preseq->logp_blank);
		    preseq->logp = preseq->logp_lm + LogAdd(preseq->logp_blank, preseq->logp_nblank);
		} else {
			beam_union_[n_preseq->prefix_word] = n_preseq;
		}
	}

	for (auto &seq : beam_union_) {
		word_beam.push_back(seq.second);
	}
}


void CTCDecoderWord::DeleteInWordBeam(std::vector<PrefixSeqWord> &beam, int size) {
    std::vector<PrefixSeqWord*> tmp;
    tmp.reserve(size);
    for (int i = 0; i < size; i++) {
        if (beam[i].trie_node == NULL)
            tmp.push_back(&beam[i]);
    }
    std::sort(tmp.begin(), tmp.end(), CTCDecoderWordUtil::compare_PrefixSeqWord_reverse);
    std::vector<PrefixSeqWord> new_beam(tmp.size());
    for (int i = 0; i < tmp.size(); i++)
        new_beam[i] = *tmp[i];
    beam.insert(beam.begin(), new_beam.begin(), new_beam.end());
}

std::string CTCDecoderWord::DebugBeam(std::vector<PrefixSeqWord> &beam, int n, int frame_idx) {
    std::ostringstream ostr;
    int size = beam.size() > n ? n : beam.size();
    ostr << "###No." << frame_idx << " frame beams info:\n";
    for (int i = 0; i < size; i++) {
        PrefixSeqWord &seq = beam[i];
        ostr << "BPE: ";
        for (int j = 1; j < seq.prefix_bpe.size(); j++)
             ostr << bpeid_to_bpe_[seq.prefix_bpe[j]] << ' ';

        ostr << ", Word: ";
        for (int j = 1; j < seq.prefix_word.size(); j++)
             ostr << wordid_to_word_[seq.prefix_word[j]] << ' ';
         ostr << "; score = " << seq.logp << ", ctc_score = " << seq.logp-seq.logp_lm <<
         ", lm_score = " << seq.logp_lm << ", trie_node = " << seq.trie_node << std::endl;
    }
    return ostr.str();
}

void CTCDecoderWord::BeamSearchTopk(const Matrix<BaseFloat> &loglikes) {
	// decode one utterance
	int nframe = loglikes.NumRows();
	int likes_size = loglikes.NumCols();
	int vocab_size = config_.vocab_size;
	PrefixSeqWord *preseq, *n_preseq;
	std::vector<int> prefix, n_prefix;
	std::vector<float> next_words(vocab_size);
	std::vector<float> next_bpes(likes_size);
	float logp = 0, logp_b = 0, logp_lm = 0, n_p_b, n_p_nb;
	float ngram_logp = 0, sub_ngram_logp = 0, ranklm_logp = 0,
			rscale = config_.rnnlm_scale, blank_penalty = log(config_.blank_penalty);
	int end_t, index, topk = likes_size/2, key, cur_his = 0, start = 0;
	int beam_size = config_.beam + config_.word_beam, cur_beam_size = 0, size;
    bool skip_blank = false, uselm;
    TrieNode *node, *next_node;

	InitDecoding(topk);
    cur_beam_size = cur_bpe_beam_size_+cur_word_beam_size_;

	// decode one utterance
	for (int n = start; n < nframe; n++) {
		logp_b = loglikes(n, config_.blank);
		uselm = false;
		if (config_.lm_scale > 0.0) {
		   if (config_.blank_threshold <= 0)
				uselm = true;
		   else if (config_.blank_threshold > 0 && Exp(logp_b) <= config_.blank_threshold)
				uselm = true;
		}

		// blank pruning
		// Only the probability of ending in blank gets updated.
		if (config_.blank_threshold > 0 && Exp(logp_b) > config_.blank_threshold) {
			logp = logp_b + blank_penalty; // -2.30259
			for (int i = 0; i < cur_beam_size; i++) {
				preseq = &cur_beam_[i];
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
			// Top K pruning, the nth bigest words
			for (int k = 1; k < topk; k++) {
				logp = loglikes(n, k);
				key = loglikes(n, topk+k);
				if (key == config_.blank) logp += blank_penalty; // -2.30259
				if (key < vocab_size && key >= 0) {
					if (!use_pinyin_) {
						next_words[key] = logp;
					} else {
						for (int i = 0; i < pinyin_to_bpe_[key].size(); i++)
							next_words[pinyin_to_bpe_[key][i]] = logp;
					}
                }
			}
		}


		/// produce next beam
		/// not extended
        next_bpe_beam_size_ = 0;
        next_word_beam_size_ = 0;
        skip_blank = false;
		for (int i = 0; i < cur_beam_size; i++) {
			preseq = &cur_beam_[i];
			end_t = preseq->PrefixBpeBack();
            if (preseq->trie_node != NULL) {
			    n_preseq = &next_bpe_beam_[next_bpe_beam_size_];
			    next_bpe_beam_size_++;
            } else {
                n_preseq = &next_word_beam_[next_word_beam_size_];
                next_word_beam_size_++;
            }

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
        /*
        for (int k = 1; k < vocab_size; k++) {
            key = k;
            logp = next_words[k];
        */
        
        if (use_pinyin_) {
        int k = 0;
        for (int i = 0; i < topk; i++) {
            key = loglikes(n, topk+k);
            logp = loglikes(n, k);
            size = pinyin_to_bpe_[key].size();
            for (int j = 0; j < size; j++) {
               next_bpes[topk+k] = pinyin_to_bpe_[key][j];
               next_bpes[k] = logp;
               k++;
               if (k >= topk) break;
            }
            if (k >= topk) break;
        }
        }

		for (int k = 1; k < topk; k++) {
			key = use_pinyin_ ? next_bpes[topk+k] : loglikes(n, topk+k);
			logp = use_pinyin_? next_bpes[k] : loglikes(n, k);

			if (key == config_.blank || key >= vocab_size || key < 0 || logp == kLogZeroFloat)
				continue;

            if (use_pinyin_ && (next_word_beam_size_ + cur_beam_size > next_word_beam_.size() 
                         || next_bpe_beam_size_ + cur_beam_size > next_bpe_beam_.size()))
                continue;

			// for each beam appending word key
			for (int i = 0; i < cur_beam_size; i++) {
				preseq = &cur_beam_[i];
				end_t = preseq->PrefixBpeBack();

				n_p_b = kLogZeroFloat;
				if (key != end_t) {
					n_p_nb = LogAdd(preseq->logp_blank+logp, preseq->logp_nblank+logp);
				} else {
					// We don't include the previous probability of not ending
					// in blank (p_nb) if s is repeated at the end. The CTC
					// algorithm merges characters not separated by a blank.
					n_p_nb = preseq->logp_blank+logp;
				}


				/////////////
				// in word
				node = preseq->trie_node;
				if (node == NULL)
					node = word_trie_.GetRootNode();

				next_node = word_trie_.Trav(node, key);

				// not in word dict, pruning
				if (next_node == NULL)
					continue;

				#if HAVE_KENLM == 1
                if (rank_kenlm_arpa_ != NULL) {
					index = rank_kenlm_vocab_->Index(bpeid_to_bpe_[key]);
					ranklm_logp = rank_kenlm_arpa_->Score(preseq->ken_state, index, n_preseq->ken_state);
					// Convert to natural log.
					ranklm_logp *= M_LN10;
                }
				#endif

				// is a word node
				if (next_node->is_word_) {
				    n_preseq = &next_word_beam_[next_word_beam_size_];
					next_word_beam_size_++;
					*n_preseq = *preseq;

					int word_id = next_node->info_->id_;

					// *NB* this would be a good place to include an LM score.
					if (config_.lm_scale > 0.0) {
						// rnn lm score
						if (rscale != 0) {
							out_words_[i].push_back(word_id);
							out_preseq_[i].push_back(n_preseq);
							n_preseq->lmhis = preseq->next_lmhis;
                            n_preseq->idx1 = preseq->idx2;
						}

						// ngram lm score
						if (rscale < 1.0) {
						#if HAVE_KENLM == 1
							if (config_.use_kenlm && kenlm_arpa_ != NULL) {
								index = kenlm_vocab_->Index(wordid_to_word_[word_id]);
								ngram_logp = kenlm_arpa_->Score(preseq->ken_state, index, n_preseq->ken_state);
								// Convert to natural log.
								ngram_logp *= M_LN10;

								n_preseq->sub_ken_state.resize(sub_kenlm_apra_.size());
								for (int i = 0; i < sub_kenlm_apra_.size(); i++) {
									index = sub_kenlm_vocab_[i]->Index(wordid_to_word_[word_id]);
									sub_ngram_logp = sub_kenlm_apra_[i]->Score(preseq->sub_ken_state[i], index, n_preseq->sub_ken_state[i]);
								    sub_ngram_logp *= M_LN10;
									ngram_logp = LogAdd(ngram_logp, sub_ngram_logp);
									//ngram_logp = std::max(ngram_logp, sub_ngram_logp);
								}
							}
                            n_preseq->ngram_logp = ngram_logp;
						#endif
						}
						// fusion score
						//logp_lm = config_.lm_scale*Log(rscale*Exp(rnnlm_logp) + (1.0-rscale)*Exp(ngram_logp));
					}

					n_preseq->PrefixBpeAppend(key);
					n_preseq->logp_blank = n_p_b;
					n_preseq->logp_nblank = n_p_nb;
                    if (rscale == 0) {
                        logp_lm = config_.lm_scale*n_preseq->ngram_logp;
                        n_preseq->logp_lm += logp_lm;
                    }
					//n_preseq->logp_lm += logp_lm;
					n_preseq->ranklm_logp += config_.lm_scale*ranklm_logp;
					n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_p_b, n_p_nb);

					n_preseq->PrefixWordAppend(word_id);
					n_preseq->trie_node = NULL;
				} 

                // is not a word node
                if ((next_node->is_word_&&next_node->NumChild() > 0) || next_node->NumChild() > 0) {
                    logp_lm = 0; // LOG_1

					n_preseq = &next_bpe_beam_[next_bpe_beam_size_];
					next_bpe_beam_size_++;
					*n_preseq = *preseq;

					n_preseq->PrefixBpeAppend(key);
					n_preseq->logp_blank = n_p_b;
					n_preseq->logp_nblank = n_p_nb;
					//n_preseq->logp_lm += logp_lm;
					n_preseq->ranklm_logp += config_.lm_scale*ranklm_logp;
					n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_p_b, n_p_nb);

					n_preseq->trie_node = next_node;
				}
			}
		}

		// rnn lm score
		if (uselm && rscale != 0) {
			// always padding to beam streams
			int i = 0, bz = 0, size = 0;
			while (bz < beam_size) {
				preseq = &cur_beam_[i];
				in_words_.push_back(preseq->PrefixWordBack());
				rnnlm_his_in_.push_back(preseq->lmhis);
				rnnlm_his_out_.push_back(preseq->next_lmhis);
				bz++;
				if (bz < cur_beam_size) i++;
			}

			lstmlm_->ForwardMseqClass(in_words_, rnnlm_his_in_,
										rnnlm_his_out_, out_words_, out_words_score_);

			for (int i = 0; i < out_words_.size(); i++) {
				size = out_words_[i].size();
				if (size == 0)
					continue;

				for (int j = 0; j < size; j++) {
					n_preseq = out_preseq_[i][j];
					n_preseq->rnnlm_logp = out_words_score_[i][j];
				    logp_lm = config_.lm_scale*Log(rscale*Exp(n_preseq->rnnlm_logp) + (1.0-rscale)*Exp(n_preseq->ngram_logp));
					n_preseq->logp_lm += logp_lm;
					n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_preseq->logp_blank, n_preseq->logp_nblank);
				}

				out_words_[i].clear();
				out_preseq_[i].clear();
				out_words_score_[i].clear();
			}

			in_words_.clear();
			rnnlm_his_in_.clear();
			rnnlm_his_out_.clear();
		}

		std::vector<PrefixSeqWord*> bpe_beam;
		std::vector<PrefixSeqWord*> word_beam;
		BeamMerge(bpe_beam, word_beam, skip_blank);
		// select best TopN beam
		std::sort(bpe_beam.begin(), bpe_beam.end(), CTCDecoderWordUtil::compare_PrefixSeqBpe_reverse);
		//std::sort(bpe_beam.begin(), bpe_beam.end(), CTCDecoderWordUtil::compare_PrefixSeqRank_reverse);
		size = bpe_beam.size();
		cur_bpe_beam_size_ = size >= config_.beam ? config_.beam : size;
		for (int i = 0; i < cur_bpe_beam_size_; i++)
			cur_beam_[i] = *bpe_beam[i];

		std::sort(word_beam.begin(), word_beam.end(), CTCDecoderWordUtil::compare_PrefixSeqWord_reverse);
		size = word_beam.size();
		cur_word_beam_size_ = size >= config_.word_beam ? config_.word_beam : size;
		for (int i = 0; i < cur_word_beam_size_; i++)
			cur_beam_[i+cur_bpe_beam_size_] = *word_beam[i];

        cur_beam_size = cur_bpe_beam_size_+cur_word_beam_size_;

        // rearrange rnnlm history
		if (uselm && rscale != 0) {
            std::fill(rnnlm_his_table_.begin(), rnnlm_his_table_.end(), false);
            for (int i = 0; i < cur_beam_size; i++)
                rnnlm_his_table_[cur_beam_[i].idx1] = true;
            int j = 0;
            for (int i = 0; i < beam_size*2; i++) {
                if (!rnnlm_his_table_[i]) {
                    cur_beam_[j].idx2 = i;
                    cur_beam_[j].next_lmhis = &rnnlm_his_[i];
                    rnnlm_his_table_[i] = true;
                    j++;
                }
               if (j >= cur_beam_size) break;
            }
        }

        if (kaldi::g_kaldi_verbose_level >= 1)
            KALDI_VLOG(1) << DebugBeam(cur_beam_, cur_beam_size, n);

        cur_his = (cur_his+1)%2;
	}

	DeleteInWordBeam(cur_beam_, cur_beam_size);
}


} // end namespace kaldi.
