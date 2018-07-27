#include "iot/end-pointer.h"

namespace kaldi {
namespace iot {

bool EndPointer::Detected(const DecCore &dec_core, BaseFloat frame_shift_in_sec) {
  int32 trailing_silence  = frame_shift_in_sec * dec_core.TrailingSilenceFrames();
  int32 utterance_length  = frame_shift_in_sec * dec_core.NumFramesDecoded();
  BaseFloat relative_cost = dec_core.FinalRelativeCost();

  bool contain_only_silence = (trailing_silence == utterance_length);

  if (contain_only_silence && (utterance_length >= config_.silence_timeout)) {
    return true;
  }

  if (!contain_only_silence) {
    if (trailing_silence >= config_.min_trailing_silence && relative_cost <= config_.max_relative_cost) {
      return true;
    }

    if (utterance_length >= config_.max_utterance_length) {
      return true;
    }
  }

  return false;
}

}  // namespace iot
}  // namespace kaldi
