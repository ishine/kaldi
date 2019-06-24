// decoder/rnnt-decoder.h

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

#ifndef KALDI_DECODER_RNNT_DECODER_H_
#define KALDI_DECODER_RNNT_DECODER_H_

#include <list>
#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"
#include "lm/kaldi-rnntlm.h"


namespace kaldi {

struct RNNTDecoderOptions {
  int beam;
  int blank;
  bool use_prefix;
  int max_mem;

  RNNTDecoderOptions(): beam(5), blank(0),
		  	  	  	  	use_prefix(false), max_mem(20000)
                        { }
  void Register(OptionsItf *opts) {
	opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
	opts->Register("blank", &blank, "RNNT bank id.");
	opts->Register("use-prefix", &use_prefix, "Process prefix probability.");
	opts->Register("max-mem", &max_mem, "maximum memory in decoding.");
  }
};

class RNNTDecoder {
	typedef Vector<BaseFloat> Pred;
	public:
		RNNTDecoder(KaldiRNNTlmWrapper &rnntlm, RNNTDecoderOptions &config);
		void GreedySearch(const Matrix<BaseFloat> &loglikes);
		void BeamSearch(const Matrix<BaseFloat> &loglikes);
		bool GetBestPath(std::vector<int> &words, BaseFloat &logp);

	protected:
		void InitDecoding();
		void FreeList(std::list<Sequence* > *list);
		void FreeSeq(Sequence *seq);
		void FreePred(Pred *pred);
		void FreeHis(LstmLmHistroy *his);
		Pred* MallocPred();
		LstmLmHistroy* MallocHis();
		void CopyPredList(std::vector<Pred*> &predlist);
		void CopyHis(LstmLmHistroy* his);
		void DeepCopySeq(Sequence *seq);
        void CleanBuffer();


		RNNTDecoderOptions &config_;
		KaldiRNNTlmWrapper &rnntlm_;
		std::list<Sequence* > *A_;
		std::list<Sequence* > *B_;
		std::unordered_map<Vector<BaseFloat> *, int> pred_buffer_;
		std::unordered_map<LstmLmHistroy *, int> his_buffer_;
		std::list<Vector<BaseFloat> *> pred_list_;
		std::list<LstmLmHistroy *>	his_list_;
		std::vector<int> rd_;
		std::vector<int> cd_;

	KALDI_DISALLOW_COPY_AND_ASSIGN(RNNTDecoder);
};


} // end namespace kaldi.


#endif
