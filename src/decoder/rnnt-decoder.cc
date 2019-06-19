// decoder/faster-decoder.cc

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

RNNTDecoder::RNNTDecoder(KaldiRNNTlmWrapper &rnntlm,
						RNNTDecoderOptions &config):
		config_(config), rnntlm_(rnntlm) {
	A_ = new std::list<Sequence*>;
	B_ = new std::list<Sequence*>;
	rd_ = rnntlm.GetRDim();
	cd_ = rnntlm.GetCDim();
}

void RNNTDecoder::FreeList(std::list<Sequence* > *list) {
	for (auto &seq : *list) {
        FreeSeqence(seq);
		delete seq;
	}

	int idx = 0;
	if (pred_buffer_.size() > config_.max_mem) {
		for (auto it = pred_buffer_.begin(); it != pred_buffer_.end(); it++) {
			if (idx >= config_.max_mem) delete (*it);
			idx++;
		}
		pred_buffer_.resize(config_.max_mem);
	}

	idx = 0;
	if (his_buffer_.size() > config_.max_mem) {
		for (auto it = his_buffer_.begin(); it != his_buffer_.end(); it++) {
			if (idx >= config_.max_mem) delete (*it);
			idx++;
		}
		his_buffer_.resize(config_.max_mem);
	}
}

void RNNTDecoder::FreeSeqence(Sequence *seq) {
    if (seq == NULL) return;
	for (int i = 0; i < seq->pred.size(); i++)
		pred_buffer_.push_back(seq->pred[i]);
	his_buffer_.push_back(seq->lmhis);
}

Vector<BaseFloat>* RNNTDecoder::MallocPred() {
	Vector<BaseFloat> *pred = NULL;
	if (pred_buffer_.size() > 0) {
		pred = pred_buffer_.front();
		pred_buffer_.pop_front();
	} else {
		pred = new Vector<BaseFloat>;
	}
	return pred;
}

LstmLmHistroy* RNNTDecoder::MallocHis(MatrixResizeType resize_type = kSetZero) {
	LstmLmHistroy *his = NULL;
	if (his_buffer_.size() > 0) {
		his = his_buffer_.front();
		his_buffer_.pop_front();
	} else {
		his = new LstmLmHistroy(rd_, cd_, resize_type);
	}
	return his;
}

void RNNTDecoder::InitDecoding() {
	// initialization
	// for (auto &seq : *B_) delete seq;
	FreeList(B_);
	B_->clear();
	LstmLmHistroy *sos_h = MallocHis(kSetZero);
	Sequence *seq = new Sequence(sos_h, config_.blank);
	B_->push_back(seq);
}

bool RNNTDecoder::GetBestPath(std::vector<int> &words, BaseFloat &logp) {
	Sequence *seq = B_->front();
	if (seq == NULL) return false;

	logp = -seq->logp;
	words = seq->k;
	return true;
}

void RNNTDecoder::BeamSearch(const Matrix<BaseFloat> &loglikes) {
	// decode one utterance
	int nframe = loglikes.NumRows();
	int vocab_size, len;
	Sequence *seqi, *seqj;
	Vector<BaseFloat> *pred, logprob(rnntlm_.GetVocabSize());
	LstmLmHistroy *his;

	InitDecoding();
	// decode one utterance
	for (int n = 0; n < nframe; n++) {
		B_->sort(RNNTUtil::compare_len_reverse);
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
					if (!RNNTUtil::isprefix(seqi->k, seqj->k))
						continue;

					int leni = seqi->k.size();
					int lenj = seqj->k.size();

					pred = MallocPred();
					his = MallocHis(kUndefined);
					rnntlm_.Forward(seqi->k[leni-1], *seqi->lmhis, *pred, *his);

					logprob.CopyFromVec(*pred);
					logprob.AddVec(1.0, loglikes.Row(n));
					logprob.ApplyLogSoftMax();
					BaseFloat curlogp = seqi->logp + logprob(seqj->k[leni]);
					for (int m = leni; m < lenj-1; m++) {
						logprob.CopyFromVec(*seqj->pred[m]);
						logprob.AddVec(1.0, loglikes.Row(n));
						logprob.ApplyLogSoftMax();
						curlogp += seqj->k[m+1];
					}
					seqj->logp = LogAdd(seqj->logp, curlogp);
				}
			}
		}

		while (true) {
			// y* = most probable in A
			Sequence *y_hat, *y_b;
			auto it = std::max_element(A_->begin(), A_->end(), RNNTUtil::compare_logp);
			y_hat = *it;
			A_->erase(it);

			// get rnnt lm current output and hidden state
			len = y_hat->k.size();
			pred = MallocPred();
			his = MallocHis(kUndefined);
			rnntlm_.Forward(y_hat->k[len-1], *y_hat->lmhis, *pred, *his);

			// log probability for each rnnt output k
			logprob.CopyFromVec(*pred);
			logprob.AddVec(1.0, loglikes.Row(n));
			logprob.ApplyLogSoftMax();

			vocab_size = logprob.Dim();
			for (int k = 0; k < vocab_size; k++) {
				Sequence *y_k = new Sequence(*y_hat);
				y_k->logp += logprob(k);
				if (k == config_.blank) {
					B_->push_back(y_k);
					continue;
				}
				// next t add to A
				y_k->lmhis = his;
				y_k->k.push_back(k);
				if (config_.use_prefix) {
					y_k->pred.push_back(pred);
				}
				A_->push_back(y_k);
			}
            
            FreeSeqence(y_hat);
            delete y_hat;
			y_hat = *std::max_element(A_->begin(), A_->end(), RNNTUtil::compare_logp);
			y_b = *std::max_element(B_->begin(), B_->end(), RNNTUtil::compare_logp);
			if (B_->size() >= config_.beam && y_b->logp >= y_hat->logp) break;
		}

		// beam width
		B_->sort(RNNTUtil::compare_logp_reverse);
		// free memory
		int idx = 0;
		for (auto it = B_->begin(); it != B_->end(); it++) {
			if (idx >= config_.beam) delete (*it);
			idx++;
		}
		B_->resize(config_.beam);
	}
}

} // end namespace kaldi.
