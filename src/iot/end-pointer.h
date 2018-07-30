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
  std::string silence_phones;
  BaseFloat silence_timeout;
  BaseFloat max_utterance_length;
  BaseFloat min_trailing_silence;
  BaseFloat max_relative_cost;
  BaseFloat subsampled_frame_shift;

  void Register(OptionsItf *opts) {
    opts->Register("silence-phones", &silence_phones, "");
    opts->Register("silence-timeout", &silence_timeout, "");
    opts->Register("max-utterance-length", &max_utterance_length, "");
    opts->Register("min-trailing-silence", &min_trailing_silence, "");
    opts->Register("max-relative-cost", &max_relative_cost, "");
    opts->Register("subsampled-frame-shift", &subsampled_frame_shift, "");
  }
};

class EndPointer {
 public:
  EndPointer(const EndPointerConfig &config)
    : config_(config)
  { KALDI_ASSERT(config_.silence_phones == "1"); } // kSilPhoneId 1

  ~EndPointer() { }
  
  bool Detected(const DecCore &dec_core);

 private:
  const EndPointerConfig &config_;
};

}  // namespace iot
}  // namespace kaldi

#endif  // KALDI_IOT_END_POINTER_
