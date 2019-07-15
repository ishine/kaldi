// decoder/rnnt-decoder.cc

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
#include "decoder/rnnt-decoder.h"

namespace kaldi {

RNNTDecoder::RNNTDecoder(KaldiLstmlmWrapper &rnntlm,
						RNNTDecoderOptions &config):
		config_(config), rnntlm_(rnntlm) {
	A_ = new std::list<Sequence*>;
	B_ = new std::list<Sequence*>;
	rd_ = rnntlm.GetRDim();
	cd_ = rnntlm.GetCDim();
}

void RNNTDecoder::FreeList(std::list<Sequence* > *list) {
	for (auto &seq : *list) {
		FreeSeq(seq);
		delete seq;
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

void RNNTDecoder::FreeSeq(Sequence *seq) {
	if (seq == NULL) return;
	for (int i = 0; i < seq->pred.size(); i++)
		FreePred(seq->pred[i]);
    FreeHis(seq->lmhis);
}

void RNNTDecoder::FreePred(Pred *pred) {
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

void RNNTDecoder::FreeHis(LstmlmHistroy *his) {
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

Vector<BaseFloat>* RNNTDecoder::MallocPred() {
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

LstmlmHistroy* RNNTDecoder::MallocHis() {
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

void RNNTDecoder::CopyPredList(std::vector<Pred*> &predlist) {
	for (int i = 0; i < predlist.size(); i++) {
		auto it = pred_buffer_.find(predlist[i]);
		if (it != pred_buffer_.end())
			pred_buffer_[predlist[i]]++;
	}
}

void RNNTDecoder::CopyHis(LstmlmHistroy* his) {
	auto it = his_buffer_.find(his);
	if (it != his_buffer_.end())
		his_buffer_[his]++;
}

void RNNTDecoder::DeepCopySeq(Sequence *seq) {
	if (seq == NULL) return;

	CopyPredList(seq->pred);
	CopyHis(seq->lmhis);
}

void RNNTDecoder::CleanBuffer() {
    for (auto it = pred_buffer_.begin(); it != pred_buffer_.end(); it++)
        delete it->first;
    for (auto it = his_buffer_.begin(); it != his_buffer_.end(); it++)
        delete it->first;
    pred_buffer_.clear();
    his_buffer_.clear();
}

void RNNTDecoder::InitDecoding() {
	// initialization
	// for (auto &seq : *B_) delete seq;
	FreeList(A_);
	A_->clear();

	FreeList(B_);
	B_->clear();

    CleanBuffer();

    // first input <s>
    LstmlmHistroy *sos_h = new LstmlmHistroy(rd_, cd_, kSetZero);
	Sequence *seq = new Sequence(sos_h, config_.blank);
	DeepCopySeq(seq);
	B_->push_back(seq);
}

bool RNNTDecoder::GetBestPath(std::vector<int> &words, BaseFloat &logp) {
	Sequence *seq = B_->front();
	if (seq == NULL) return false;

	logp = -seq->logp;
	words = seq->k;
	return true;
}
void RNNTDecoder::GreedySearch(const Matrix<BaseFloat> &loglikes) {
	int nframe = loglikes.NumRows();
	int len, k;
	Vector<BaseFloat> *pred, logprob(rnntlm_.GetVocabSize());
	Sequence *y_hat;
	LstmlmHistroy *his;
	BaseFloat logp;

	InitDecoding();
    /*
    if (config_.blank_posterior_scale >= 0) {
        loglikes.ColRange(0, 1).Scale(config_.blank_posterior_scale);
    }
    */
	// decode one utterance
	y_hat = B_->front();

	// <sos> first
	pred = MallocPred();
	his = MallocHis();
	len = y_hat->k.size();
	rnntlm_.Forward(y_hat->k[len-1], *y_hat->lmhis, pred, his);

	for (int n = 0; n < nframe; n++) {
		// log probability for each rnnt output k
	    logprob.CopyFromVec(*pred);
		logprob.AddVec(1.0, loglikes.Row(n));
		//logprob.ApplyLogSoftMax();
        logprob.ApplySoftMax();
        if (config_.blank_posterior_scale >= 0)
            logprob(0) *= config_.blank_posterior_scale;
        logprob.ApplyLog();

		logp = logprob.Max(&k);
		y_hat->logp += logp;
		if (k != config_.blank) {
			y_hat->k.push_back(k);
			y_hat->pred.push_back(pred);
			FreeHis(y_hat->lmhis);
			y_hat->lmhis = his;

			pred = MallocPred();
			his = MallocHis();
			len = y_hat->k.size();
			rnntlm_.Forward(y_hat->k[len-1], *y_hat->lmhis, pred, his);
		}
	}
}

void RNNTDecoder::BeamSearch(const Matrix<BaseFloat> &loglikes) {
	// decode one utterance
	int nframe = loglikes.NumRows();
	int vocab_size, len;
	Sequence *seqi, *seqj;
	Vector<BaseFloat> *pred, logprob(rnntlm_.GetVocabSize());
	LstmlmHistroy *his;

	InitDecoding();
	// decode one utterance
	for (int n = 0; n < nframe; n++) {
		B_->sort(LstmlmUtil::compare_len_reverse);
		//for (auto &seq : *A_) delete seq;
		FreeList(A_);
		delete A_;

		A_ = B_;
		B_ = new std::list<Sequence*>;

		if (config_.use_prefix) {
			for (auto iterj = A_->begin(); iterj != A_->end(); iterj++) {
				auto iteri = iterj; iteri++;
				for (; iteri != A_->end(); iteri++) {
					seqi = *iteri; seqj = *iterj;
					if (!LstmlmUtil::isprefix(seqi->k, seqj->k))
						continue;

					int leni = seqi->k.size();
					int lenj = seqj->k.size();

					pred = MallocPred();
					rnntlm_.Forward(seqi->k[leni-1], *seqi->lmhis, pred, NULL);

					logprob.CopyFromVec(*pred);
					logprob.AddVec(1.0, loglikes.Row(n));
					//logprob.ApplyLogSoftMax();
                    logprob.ApplySoftMax();
                    if (config_.blank_posterior_scale >= 0)
                        logprob(0) *= config_.blank_posterior_scale;
                    logprob.ApplyLog();

					BaseFloat curlogp = seqi->logp + logprob(seqj->k[leni]);
					for (int m = leni; m < lenj-1; m++) {
						logprob.CopyFromVec(*seqj->pred[m]);
						logprob.AddVec(1.0, loglikes.Row(n));
						//logprob.ApplyLogSoftMax();
                        logprob.ApplySoftMax();
                        if (config_.blank_posterior_scale >= 0)
                            logprob(0) *= config_.blank_posterior_scale;
                        logprob.ApplyLog();

						curlogp += seqj->k[m+1];
					}
					seqj->logp = LogAdd(seqj->logp, curlogp);
					FreePred(pred);
				}
			}
		}

		while (true) {
			// y* = most probable in A
			Sequence *y_hat, *y_a, *y_b;
			auto it = std::max_element(A_->begin(), A_->end(), LstmlmUtil::compare_logp);
			A_->erase(it);
			y_hat = *it;

			// get rnnt lm current output and hidden state
			pred = MallocPred();
			his = MallocHis();
			len = y_hat->k.size();
			rnntlm_.Forward(y_hat->k[len-1], *y_hat->lmhis, pred, his);

			// log probability for each rnnt output k
			logprob.CopyFromVec(*pred);
			logprob.AddVec(1.0, loglikes.Row(n));
			//logprob.ApplyLogSoftMax();
            logprob.ApplySoftMax();
            if (config_.blank_posterior_scale >= 0)
                logprob(0) *= config_.blank_posterior_scale;
            logprob.ApplyLog();

			vocab_size = logprob.Dim();
			for (int k = 0; k < vocab_size; k++) {
				Sequence *y_k = new Sequence(*y_hat);

				y_k->logp += logprob(k);
				if (k == config_.blank) {
					DeepCopySeq(y_hat);
					B_->push_back(y_k);
					continue;
				}

				// next t add to A
				y_k->k.push_back(k);
				y_k->lmhis = his;
				CopyHis(his);
				if (config_.use_prefix) {
					y_k->pred.push_back(pred);
					CopyPredList(y_k->pred);
				}
				A_->push_back(y_k);
			}
            
			// free link count
			FreeSeq(y_hat);
			FreePred(pred);
			FreeHis(his);
			y_a = *std::max_element(A_->begin(), A_->end(), LstmlmUtil::compare_logp);
			y_b = *std::max_element(B_->begin(), B_->end(), LstmlmUtil::compare_logp);
			if (B_->size() >= config_.beam && y_b->logp >= y_a->logp) break;
		}

		// beam width
		B_->sort(LstmlmUtil::compare_logp_reverse);
		// free memory
		int idx = 0;
		for (auto it = B_->begin(); it != B_->end(); it++) {
			if (idx >= config_.beam) FreeSeq(*it);
			idx++;
		}
		B_->resize(config_.beam);
	}
}

} // end namespace kaldi.
