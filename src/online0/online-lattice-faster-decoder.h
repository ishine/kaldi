// online0/online-lattice-faster-decoder.h

// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

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

#ifndef ONLINE0_ONLINE_LATTICE_FASTER_DECODER_H_
#define ONLINE0_ONLINE_LATTICE_FASTER_DECODER_H_

#include "util/stl-utils.h"
#include "hmm/transition-model.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "online0/online-util.h"

namespace kaldi {

// Extends the definition of FasterDecoder's options to include additional
// parameters. The meaning of the "beam" option is also redefined as
// the _maximum_ beam value allowed.
struct OnlineLatticeFasterDecoderOptions : public LatticeFasterDecoderConfig {
	  BaseFloat rt_min; // minimum decoding runtime factor
	  BaseFloat rt_max; // maximum decoding runtime factor
	  int32 batch_size; // number of features decoded in one go
	  int32 inter_utt_sil; // minimum silence (#frames) to trigger end of utterance
	  int32 max_utt_len_; // if utt. is longer, we accept shorter silence as utt. separators
	  int32 update_interval; // beam update period in # of frames
	  BaseFloat beam_update; // rate of adjustment of the beam
	  BaseFloat max_beam_update; // maximum rate of beam adjustment
	  std::string cutoff;

	  OnlineLatticeFasterDecoderOptions() :
	    rt_min(0.7), rt_max(0.75), batch_size(18),
	    inter_utt_sil(50), max_utt_len_(1500),
	    update_interval(3), beam_update(0.01),
	    max_beam_update(0.05), cutoff("hybrid") {}

	  void Register(OptionsItf *opts, bool full = true) {
	    LatticeFasterDecoderConfig::Register(opts);
	    opts->Register("rt-min", &rt_min,
	                   "Approximate minimum decoding run time factor");
	    opts->Register("rt-max", &rt_max,
	                   "Approximate maximum decoding run time factor");
	    opts->Register("batch-size", &batch_size,
	                   "number of features decoded in one go");
	    opts->Register("update-interval", &update_interval,
	                   "Beam update interval in frames");
	    opts->Register("beam-update", &beam_update, "Beam update rate");
	    opts->Register("max-beam-update", &max_beam_update, "Max beam update rate");
	    opts->Register("inter-utt-sil", &inter_utt_sil,
	                   "Maximum # of silence frames to trigger new utterance");
	    opts->Register("max-utt-length", &max_utt_len_,
	                   "If the utterance becomes longer than this number of frames, "
	                   "shorter silence is acceptable as an utterance separator");
	    opts->Register("cutoff", &cutoff,
	    					"token cutoff algorithm, e.g. ctc or hmm-dnn hybrid");
	  }
};

template <typename FST>
class OnlineLatticeFasterDecoderTpl : public LatticeFasterOnlineDecoderTpl<FST> {
public:
	// Codes returned by Decode() to show the current state of the decoder
	enum DecodeState {
		kStartFeats = 1, // Start from the Decodable
		kEndFeats = 2, // No more scores are available from the Decodable
		kEndBatch = 3, // End of batch - end of utterance not reached yet
	};

    using Arc = typename FST::Arc;
    using Label = typename Arc::Label;
    using StateId = typename Arc::StateId;
    using Weight = typename Arc::Weight;
    //using Token = decoder::BackpointerToken;
    using Token = typename kaldi::LatticeFasterOnlineDecoderTpl<FST>::Token;

	OnlineLatticeFasterDecoderTpl(const FST &fst, const OnlineLatticeFasterDecoderOptions &opts):
								LatticeFasterOnlineDecoderTpl<FST>(fst, opts),
								opts_(opts), 
								state_(kStartFeats), frame_(0), utt_frames_(0),
								immortal_tok_(NULL), prev_immortal_tok_(NULL) {
								}

	DecodeState Decode(DecodableInterface *decodable);

	/// Finalizes the decoding. Cleans up and prunes remaining tokens, so the
	/// GetLattice() call will return faster.  You must not call this before
	/// calling (TerminateDecoding() or InputIsFinished()) and then Wait().
	void FinalizeDecoding();

	/// Gets the lattice.  The output lattice has any acoustic scaling in it
	/// (which will typically be desirable in an online-decoding context); if you
	/// want an un-scaled lattice, scale it using ScaleLattice() with the inverse
	/// of the acoustic weight.  "end_of_utterance" will be true if you want the
	/// final-probs to be included.
	void GetLattice(bool end_of_utterance, CompactLattice *clat,
			const TransitionModel *trans_model = NULL) const;

	int32 frame() { return frame_; }

	void ResetDecoder(bool full);

private:



	const OnlineLatticeFasterDecoderOptions &opts_;
	DecodeState state_; // the current state of the decoder
	int32 frame_; // the next frame to be processed
	int32 utt_frames_; // # frames processed from the current utterance
	Token *immortal_tok_;      // "immortal" token means it's an ancestor of ...
	Token *prev_immortal_tok_; // ... all currently active tokens
	KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineLatticeFasterDecoderTpl);
};

typedef OnlineLatticeFasterDecoderTpl<fst::StdFst> OnlineLatticeFasterDecoder;

} // namespace kaldi



#endif /* ONLINE0_ONLINE_LATTICE_FASTER_DECODER_H_ */
