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
						std::vector<KenModel *> &sub_kenlm_apra):
		config_(config), lstmlm_(lstmlm), kenlm_arpa_(kenlm_arpa), sub_kenlm_apra_(sub_kenlm_apra) {
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
    if (config_.rnnlm_scale > 0) {
    	LstmlmHistroy his(rd_, cd_, kSetZero);
    	rnnlm_his_.resize(2*config_.word_beam, his);
    	rnnlm_logp_.resize(config_.word_beam);
    	for (int i = 0; i < beam_size; i++) {
    		cur_beam_[i].lmhis = &rnnlm_his_[i];
    		cur_beam_[i].next_lmhis = &rnnlm_his_[beam_size+i];
    		cur_beam_[i].lmlogp = &rnnlm_logp_[i];
    	}
    }
}


void CTCDecoderWord::InitDecoding(int topk) {
    // Elements in the beam are (prefix, (p_blank, p_no_blank))
    // Initialize the beam with the empty sequence, a probability of
    // 1 for ending in blank and zero for ending in non-blank (in log space).
	PrefixSeqWord *seq = &cur_beam_[0];
    seq->Reset();
	seq->PrefixBpeAppend(config_.blank);
	seq->PrefixWordAppend(config_.blank);
	seq->logp_blank = 0.0;
	cur_bpe_beam_size_ = 1;
	cur_word_beam_size_ = 0;

    if (config_.rnnlm_scale != 0) {
    	// first input <s>
    	seq->lmhis->SetZero();
    }

	// next beam buffer
    int beam_size = config_.beam + config_.word_beam;
	next_bpe_beam_.resize(beam_size*topk, PrefixSeqWord());
	next_bpe_beam_size_ = 0;

	next_word_beam_.resize(beam_size*topk, PrefixSeqWord());
	next_word_beam_size_ = 0;

    //beam_union_.reserve(beam_size*topk);

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
	std::vector<int> next_scene_bpes(vocab_size);
	float logp = 0, logp_b = 0, logp_lm = 0, n_p_b, n_p_nb;
	float ngram_logp = kLogZeroFloat, rnnlm_logp = kLogZeroFloat, sub_ngram_logp = kLogZeroFloat,
			rscale = config_.rnnlm_scale,
            blank_penalty = log(config_.blank_penalty);
	int end_t, index, topk = likes_size/2, key, cur_his = 0, start = 0;
    bool skip_blank = false, uselm;
    TrieNode *node, *next_node;

    // rnnlm
    std::vector<int> in_words;
    std::vector<Vector<BaseFloat>*> nnet_out;
    std::vector<LstmlmHistroy*> context_in, context_out;

	InitDecoding(topk);

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
				preseq = &cur_beam_[i];
				in_words.push_back(preseq->PrefixWordBack());
				context_in.push_back(preseq->lmhis);
				context_out.push_back(preseq->next_lmhis);
				nnet_out.push_back(preseq->lmlogp);
				bz++;
				if (bz < cur_word_beam_size_) i++;
			}

			// beam streams parallel process
			lstmlm_->ForwardMseq(in_words, context_in, nnet_out, context_out);

			// get the valid streams
			for (int i = 0; i < cur_word_beam_size_; i++) {
				preseq = &cur_beam_[cur_bpe_beam_size_+i];
				preseq->lmlogp->ApplyLog();
			}
		}


		// blank pruning
		// Only the probability of ending in blank gets updated.
		if (config_.blank_threshold > 0 && Exp(logp_b) > config_.blank_threshold) {
			logp = logp_b + blank_penalty; // -2.30259
			for (int i = 0; i < cur_bpe_beam_size_+cur_word_beam_size_; i++) {
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
					next_words[key] = logp;
				}
			}
		}


		/// produce next beam
		/// not extended
        next_bpe_beam_size_ = 0;
        next_word_beam_size_ = 0;
        skip_blank = false;
		for (int i = 0; i < cur_bpe_beam_size_+cur_word_beam_size_; i++) {
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
		for (int k = 1; k < topk; k++) {
			key = loglikes(n, topk+k);
			logp = loglikes(n, k);
			if (key == config_.blank || key >= vocab_size || key < 0)
				continue;

			// for each beam appending word key
			for (int i = 0; i < cur_bpe_beam_size_+cur_word_beam_size_; i++) {
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
							rnnlm_logp = (*preseq->lmlogp)(key);
							// rnnlm history
							n_preseq->lmhis = preseq->next_lmhis;
							// n_preseq->next_lmhis = preseq->lmhis;
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
						#endif
						}
						// fusion score
						logp_lm = config_.lm_scale*Log(rscale*Exp(rnnlm_logp) + (1.0-rscale)*Exp(ngram_logp));
					}

					n_preseq->PrefixBpeAppend(key);
					n_preseq->logp_blank = n_p_b;
					n_preseq->logp_nblank = n_p_nb;
					n_preseq->logp_lm += logp_lm;
					n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_p_b, n_p_nb);

					n_preseq->PrefixWordAppend(word_id);
					n_preseq->trie_node = NULL;

					if (next_node->NumChild() > 0) {
						logp_lm = 0; // LOG_1

						n_preseq = &next_bpe_beam_[next_bpe_beam_size_];
						next_bpe_beam_size_++;
						*n_preseq = *preseq;

						n_preseq->PrefixBpeAppend(key);
						n_preseq->logp_blank = n_p_b;
						n_preseq->logp_nblank = n_p_nb;
						n_preseq->logp_lm += logp_lm;
						n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_p_b, n_p_nb);

						n_preseq->trie_node = next_node;
					}
				} else { // is not a word node
                    logp_lm = 0; // LOG_1

					n_preseq = &next_bpe_beam_[next_bpe_beam_size_];
					next_bpe_beam_size_++;
					*n_preseq = *preseq;

					n_preseq->PrefixBpeAppend(key);
					n_preseq->logp_blank = n_p_b;
					n_preseq->logp_nblank = n_p_nb;
					n_preseq->logp_lm += logp_lm;
					n_preseq->logp = n_preseq->logp_lm +  LogAdd(n_p_b, n_p_nb);

					n_preseq->trie_node = next_node;
				}

			}
		}

		std::vector<PrefixSeqWord*> bpe_beam;
		std::vector<PrefixSeqWord*> word_beam;
		BeamMerge(bpe_beam, word_beam, skip_blank);
		// select best TopN beam
		std::sort(bpe_beam.begin(), bpe_beam.end(), CTCDecoderWordUtil::compare_PrefixSeqBpe_reverse);
		int size = bpe_beam.size();
		cur_bpe_beam_size_ = size >= config_.beam ? config_.beam : size;
		for (int i = 0; i < cur_bpe_beam_size_; i++) {
			cur_beam_[i] = *bpe_beam[i];
			if (rscale != 0) {
				cur_beam_[i].lmlogp = &rnnlm_logp_[i];
				cur_beam_[i].next_lmhis = &rnnlm_his_[config_.beam*cur_his+i];
            }
        }

		std::sort(word_beam.begin(), word_beam.end(), CTCDecoderWordUtil::compare_PrefixSeqWord_reverse);
		size = word_beam.size();
		cur_word_beam_size_ = size >= config_.word_beam ? config_.word_beam : size;
		for (int i = 0; i < cur_word_beam_size_; i++) {
			cur_beam_[i+cur_bpe_beam_size_] = *word_beam[i];
		}

        if (kaldi::g_kaldi_verbose_level >= 1)
            KALDI_VLOG(1) << DebugBeam(cur_beam_, cur_bpe_beam_size_+cur_word_beam_size_, n);

        cur_his = (cur_his+1)%2;
	}

	DeleteInWordBeam(cur_beam_, cur_bpe_beam_size_+cur_word_beam_size_);
}


} // end namespace kaldi.
