// online0/online-lattice-faster-decoder.cc

// Copyright 2012 Cisco Systems (author: Matthias Paulik)
// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

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

#include "base/timer.h"
#include "fstext/fstext-utils.h"
#include "hmm/hmm-utils.h"

#include "online0/online-lattice-faster-decoder.h"

namespace kaldi {

template <typename FST>
void OnlineLatticeFasterDecoderTpl<FST>::ResetDecoder(bool full) {
  this->InitDecoding();

  //prev_immortal_tok_ = immortal_tok_ = dummy_token;
  utt_frames_ = 0;
  if (full)
    frame_ = 0;
  state_ = kStartFeats;
}

template <typename FST>
void OnlineLatticeFasterDecoderTpl<FST>::FinalizeDecoding() {
  this->FinalizeDecoding();
}

template <typename FST>
void OnlineLatticeFasterDecoderTpl<FST>::GetBestPath(bool end_of_utterance, Lattice *best_path) const {
  this->GetBestPath(best_path, end_of_utterance);
}

template <typename FST>
void OnlineLatticeFasterDecoderTpl<FST>::GetLattice(bool end_of_utterance,
		CompactLattice *clat, const TransitionModel *trans_model) const {
	if (this->NumFramesDecoded() == 0)
	  KALDI_ERR << "You cannot get a lattice if you decoded no frames.";
	Lattice raw_lat;
	this->GetRawLattice(&raw_lat, end_of_utterance);

	if (!opts_.determinize_lattice)
	KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

	if (opts_.cutoff == "hybrid")
		DeterminizeLatticePhonePrunedWrapper(*trans_model, &raw_lat, opts_.lattice_beam, clat, opts_.det_opts);
	else
		DeterminizeLatticePhonePrunedCtcWrapper(&raw_lat, opts_.lattice_beam, clat, opts_.det_opts);
}

template <typename FST>
typename OnlineLatticeFasterDecoderTpl<FST>::DecodeState
OnlineLatticeFasterDecoderTpl<FST>::Decode(DecodableInterface *decodable) {
  if (state_ == kEndFeats) // new utterance
    return state_;

  this->AdvanceDecoding(decodable, opts_.batch_size);

  frame_ = this->NumFramesDecoded();
  state_ = kEndBatch;
  if (decodable->IsLastFrame(frame_ - 1))
	  state_ = kEndFeats;

  return state_;
}

} // namespace kaldi
