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

#include "iot/decoder.h"

namespace kaldi {
namespace iot {

class EndPointer {
 public:
  EndPointer() :
    min_trailing_silence_in_sec_(0.5f),
    max_relative_cost_(2.0f),
    max_utterance_length_in_sec_(20.0f)
  { }

  EndPointer(BaseFloat min_trailing_silence_in_sec,
             BaseFloat max_relative_cost,
             BaseFloat max_utterance_length_in_sec) :
    min_trailing_silence_in_sec_(min_trailing_silence_in_sec),
    max_relative_cost_(max_relative_cost),
    max_utterance_length_in_sec_(max_utterance_length_in_sec)
  { }

  ~EndPointer();
  
  bool Detected(Decoder &decoder);

 private:
  BaseFloat min_trailing_silence_in_sec_;
  BaseFloat max_relative_cost_;
  BaseFloat max_utterance_length_in_sec_;
};

}  // namespace iot
}  // namespace kaldi

#endif  // KALDI_IOT_END_POINTER_
