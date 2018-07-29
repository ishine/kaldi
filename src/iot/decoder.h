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
#include "online2/online-nnet2-feature-pipeline.h"
#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"

#include "iot/dec-core.h"
#include "iot/end-pointer.h"

namespace kaldi {
namespace iot {

class Decoder {
 public:
  // Constructor. The pointer 'features' is not being given to this class to own
  // and deallocate, it is owned externally.
  Decoder(Wfst *fst, 
          const TransitionModel &trans_model,
          const nnet3::DecodableNnetSimpleLoopedInfo &info,
          OnlineNnet2FeaturePipeline *features,
          const DecCoreConfig &dec_core_config);

  void StartSession(const char* session_key = NULL);
  void Advance();
  void StopSession();
  int32 NumFramesDecoded() const;

  void EnableEndPointer(EndPointerConfig &end_pointer_config);
  bool EndpointDetected();

  /// Gets the lattice.  The output lattice has any acoustic scaling in it
  /// (which will typically be desirable in an online-decoding context); if you
  /// want an un-scaled lattice, scale it using ScaleLattice() with the inverse
  /// of the acoustic weight.  "end_of_utterance" will be true if you want the
  /// final-probs to be included.
  void GetLattice(bool use_final_prob, CompactLattice *clat) const;

  /// Outputs an FST corresponding to the single best path through the current
  /// lattice. If "use_final_probs" is true AND we reached the final-state of
  /// the graph then it will include those as final-probs, else it will treat
  /// all final-probs as one.
  void GetBestPath(bool use_final_prob, Lattice *best_path) const;

  const DecCore &GetDecCore() const { return dec_core_; }

  ~Decoder();

 private:
  const TransitionModel &trans_model_;

  nnet3::DecodableAmNnetLoopedOnline decodable_;
  // this is remembered from the constructor; it's ultimately
  // derived from calling FrameShiftInSeconds() on the feature pipeline.
  BaseFloat feature_frame_shift_in_sec_;

  const DecCoreConfig &dec_core_config_;
  DecCore dec_core_;

  EndPointer *end_pointer_;
};


}  // namespace iot
}  // namespace kaldi

#endif
