#ifndef KALDI_IOT_ONLINE_DECODER_H_
#define KALDI_IOT_ONLINE_DECODER_H_

#include <string>
#include <vector>
#include <deque>

#include "hmm/transition-model.h"
#include "hmm/posterior.h"

#include "nnet3/decodable-online-looped.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "itf/online-feature-itf.h"
#include "online2/online-endpoint.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"

#include "iot/dec-core.h"

namespace kaldi {
namespace iot {

class OnlineDecoder {
 public:

  // Constructor. The pointer 'features' is not being given to this class to own
  // and deallocate, it is owned externally.
  OnlineDecoder(const DecCoreConfig &decoder_opts,
                const TransitionModel &trans_model,
                const nnet3::DecodableNnetSimpleLoopedInfo &info,
                Wfst *fst,
                OnlineNnet2FeaturePipeline *features);

  /// advance the decoding as far as we can.
  void AdvanceDecoding();

  /// Finalizes the decoding. Cleans up and prunes remaining tokens, so the
  /// GetLattice() call will return faster.  You must not call this before
  /// calling (TerminateDecoding() or InputIsFinished()) and then Wait().
  void FinalizeDecoding();

  int32 NumFramesDecoded() const;

  /// Gets the lattice.  The output lattice has any acoustic scaling in it
  /// (which will typically be desirable in an online-decoding context); if you
  /// want an un-scaled lattice, scale it using ScaleLattice() with the inverse
  /// of the acoustic weight.  "end_of_utterance" will be true if you want the
  /// final-probs to be included.
  void GetLattice(bool end_of_utterance, CompactLattice *clat) const;

  /// Outputs an FST corresponding to the single best path through the current
  /// lattice. If "use_final_probs" is true AND we reached the final-state of
  /// the graph then it will include those as final-probs, else it will treat
  /// all final-probs as one.
  void GetBestPath(bool end_of_utterance,
                   Lattice *best_path) const;


  /// This function calls EndpointDetected from online-endpoint.h,
  /// with the required arguments.
  //bool EndpointDetected(const OnlineEndpointConfig &config);

  const DecCore &Decoder() const { return decoder_; }

  ~OnlineDecoder() { }
 private:

  const DecCoreConfig &decoder_opts_;

  // this is remembered from the constructor; it's ultimately
  // derived from calling FrameShiftInSeconds() on the feature pipeline.
  BaseFloat input_feature_frame_shift_in_seconds_;

  // we need to keep a reference to the transition model around only because
  // it's needed by the endpointing code.
  const TransitionModel &trans_model_;

  nnet3::DecodableAmNnetLoopedOnline decodable_;

  DecCore decoder_;
};


}  // namespace iot
}  // namespace kaldi

#endif
