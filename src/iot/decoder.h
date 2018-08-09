#ifndef KALDI_IOT_DECODER_H_
#define KALDI_IOT_DECODER_H_

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
  Decoder(Wfst *fst, 
          const TransitionModel &trans_model,
          nnet3::AmNnetSimple &am_nnet,
          const OnlineNnet2FeaturePipelineConfig &feature_config,
          const nnet3::NnetSimpleLoopedComputationOptions &decodable_config,
          const DecCoreConfig &core_config);

  void EnableEndPointer(EndPointerConfig &end_pointer_config);

  void StartSession(const char* session_key = NULL);

  void AcceptAudio(const void* data, int32 nbytes, AudioFormat audio_format);

  int32 NumFramesDecoded() const;

  bool EndpointDetected();

  void StopSession();

  // Gets acoustic-scaled lattice
  void GetLattice(bool use_final_prob, CompactLattice *clat) const;

  /// Outputs an FST corresponding to the single best path through the current
  /// lattice. If "use_final_probs" is true AND we reached the final-state of
  /// the graph then it will include those as final-probs, else it will treat
  /// all final-probs as one.
  void GetBestPath(bool use_final_prob, Lattice *best_path) const;

  /*
  const DecCore &GetDecCore() const { return core_; }
  */

  ~Decoder();

  OnlineNnet2FeaturePipeline* Feature() { return feature_; }
  OnlineNnet2FeaturePipelineInfo& FeatureInfo() { return feature_info_; }

 private:
  const TransitionModel &trans_model_;
  
  // feature
  OnlineNnet2FeaturePipelineInfo feature_info_;
  OnlineNnet2FeaturePipeline *feature_;

  // decodable
  nnet3::DecodableNnetSimpleLoopedInfo decodable_info_;
  nnet3::DecodableAmNnetLoopedOnline *decodable_;

  // core engine
  const DecCoreConfig &core_config_;
  DecCore core_;

  EndPointer *end_pointer_;
};


}  // namespace iot
}  // namespace kaldi

#endif
