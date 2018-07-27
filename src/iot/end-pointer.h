#ifndef KALDI_IOT_END_POINTER_H_
#define KALDI_IOT_END_POINTER_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/feature-functions.h"
#include "feat/feature-mfcc.h"
#include "feat/feature-plp.h"
#include "itf/online-feature-itf.h"
#include "lat/kaldi-lattice.h"
#include "hmm/transition-model.h"

#include "iot/dec-core.h"

namespace kaldi {
namespace iot {

struct EndPointerConfig {
  BaseFloat silence_timeout;
  BaseFloat max_utterance_length;
  BaseFloat min_trailing_silence;
  BaseFloat max_relative_cost;

  void Register(OptionsItf *opts) {
    ;
  }
};

class EndPointer {
 public:
  EndPointer(const EndPointerConfig &config)
    : config_(config)
  { }

  ~EndPointer() { }
  
  bool Detected(const DecCore &dec_core, BaseFloat frame_shift_in_sec);

 private:
  const EndPointerConfig &config_;
};

}  // namespace iot
}  // namespace kaldi

#endif  // KALDI_IOT_END_POINTER_
