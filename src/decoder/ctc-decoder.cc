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
						KaldiLstmlmWrapper &lstmlm,
						ConstArpaLm &const_arpa):
		config_(config), lstmlm_(lstmlm), const_arpa_(const_arpa) {
	rd_ = lstmlm.GetRDim();
	cd_ = lstmlm.GetCDim();

	use_pinyin_ = false;
	if (config_.pinyin2words_id_rxfilename != "") {
		SequentialInt32VectorReader wordid_reader(config.pinyin2words_id_rxfilename);
		for (; !wordid_reader.Done(); wordid_reader.Next()) {
			pinyin2words_.push_back(wordid_reader.Value());
		}
		use_pinyin_ = true;
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

bool CTCDecoder::GetBestPath(std::vector<int> &words, BaseFloat &logp) {
	PrefixSeq *seq = pre_seq_list_.front();
	if (seq == NULL) return false;

	logp = -LogAdd(seq->logp_blank, seq->logp_nblank);
	words = seq->prefix;
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
	beam_[seq->prefix] = seq;
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

void CTCDecoder::BeamSearch(const Matrix<BaseFloat> &loglikes) {
	// decode one utterance
	int nframe = loglikes.NumRows();
	int likes_size = loglikes.NumCols();
	int vocab_size = lstmlm_.GetVocabSize();
	PrefixSeq *preseq, *n_preseq;
	std::vector<int> n_prefix, prefix;
	std::vector<float> next_words(vocab_size);
	Vector<BaseFloat> *lmlogp;
	LstmlmHistroy *his = NULL;
    std::vector<BaseFloat> next_step(likes_size);
    std::vector<int> in_words;
    std::vector<Vector<BaseFloat>*> nnet_out;
    std::vector<LstmlmHistroy*> context_in, context_out;
	float logp, logp_b, n_p_b, n_p_nb, ngram_logp = 0, rnnlm_logp = 0,
            rscale = config_.rnnlm_scale;
	int end_t, bz;
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
            lstmlm_.ForwardMseq(in_words, context_in, nnet_out, context_out);

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
			next_words[config_.blank] = logp_b;
		} else if (config_.am_topk > 0) {
			// Top K pruning, the nth bigest words
            memcpy(&next_step.front(), loglikes.RowData(n), next_step.size()*sizeof(BaseFloat));
            std::nth_element(next_step.begin(), next_step.begin()+config_.am_topk, next_step.end(), std::greater<BaseFloat>());
            for (int k = 0; k < likes_size; k++) {
            	logp = loglikes(n, k);
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
                        prefix = preseq->prefix;
                        prefix[0] = config_.sos; // <s>
					    ngram_logp = const_arpa_.GetNgramLogprob(k, prefix);
                    }
                    // fusion score
					// n_preseq->logp_nblank = n_p_nb + config_.lm_scale*(rscale*rnnlm_logp + (1.0-rscale)*ngram_logp);
					n_preseq->logp_nblank = n_p_nb + config_.lm_scale*Log(rscale*Exp(rnnlm_logp) + (1.0-rscale)*Exp(ngram_logp));
				} else {
					n_preseq->logp_nblank = n_p_nb;
				}
				next_beam_[n_preseq->prefix] = n_preseq;
			}
		}


		// Sort and trim the beam before moving on to the next time-step.
		pre_seq_list_.clear();
		for (auto it = next_beam_.begin(); it != next_beam_.end(); it++) {
			preseq = it->second;
			pre_seq_list_.push_back(preseq);
		}
		pre_seq_list_.sort(LstmlmUtil::compare_PrefixSeq_reverse);

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
				lstmlm_.Forward(preseq->prefix.back(), *preseq->lmhis, lmlogp, his);
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
		pre_seq_list_.sort(LstmlmUtil::compare_PrefixSeq_reverse);

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

} // end namespace kaldi.
