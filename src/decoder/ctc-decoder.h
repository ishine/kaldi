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
#include "lm/kaldi-lstmlm.h"
#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"


namespace kaldi {

struct CTCDecoderOptions {
  int beam;
  int blank;
  int   am_topk;
  float lm_scale;
  float blank_threshold;
  int max_mem;

  CTCDecoderOptions(): beam(5), blank(0), am_topk(-1),
		  	  	  	   lm_scale(0.0), blank_threshold(0.0), max_mem(50000)
                        { }
  void Register(OptionsItf *opts) {
	opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
	opts->Register("blank", &blank, "CTC bank id.");
	opts->Register("am-topk", &am_topk, "For each time step beam search, keep top K am output probability words.");
	opts->Register("lm-scale", &lm_scale, "Process extend language model log probability.");
	opts->Register("blank-threshold", &blank_threshold, "Procee am blank output probability, exceed threshold will be blank directly.");
	opts->Register("max-mem", &max_mem, "maximum memory in decoding.");
  }
};

class CTCDecoder {
	typedef Vector<BaseFloat> Pred;
	public:
		CTCDecoder(KaldiLstmlmWrapper &rnntlm, CTCDecoderOptions &config);
		void GreedySearch(const Matrix<BaseFloat> &loglikes);
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

	KALDI_DISALLOW_COPY_AND_ASSIGN(CTCDecoder);
};


} // end namespace kaldi.


#endif
