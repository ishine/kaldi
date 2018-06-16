#include "iot/online-decoder.h"

namespace kaldi {
namespace iot {

OnlineDecoder::OnlineDecoder(
    const DecCoreConfig &decoder_opts,
    const TransitionModel &trans_model,
    const nnet3::DecodableNnetSimpleLoopedInfo &info,
    Wfst *fst,
    OnlineNnet2FeaturePipeline *features):
    decoder_opts_(decoder_opts),
    input_feature_frame_shift_in_seconds_(features->FrameShiftInSeconds()),
    trans_model_(trans_model),
    decodable_(trans_model_, info,
               features->InputFeature(), features->IvectorFeature()),
    decoder_(fst, decoder_opts_) {
  decoder_.InitDecoding();
}

void OnlineDecoder::AdvanceDecoding() {
  decoder_.AdvanceDecoding(&decodable_);
}

void OnlineDecoder::FinalizeDecoding() {
  decoder_.FinalizeDecoding();
}

int32 OnlineDecoder::NumFramesDecoded() const {
  return decoder_.NumFramesDecoded();
}

void OnlineDecoder::GetLattice(bool end_of_utterance,
                                             CompactLattice *clat) const {
  if (NumFramesDecoded() == 0)
    KALDI_ERR << "You cannot get a lattice if you decoded no frames.";
  Lattice raw_lat;
  decoder_.GetRawLattice(&raw_lat, end_of_utterance);

  if (!decoder_opts_.determinize_lattice)
    KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

  BaseFloat lat_beam = decoder_opts_.lattice_beam;
  DeterminizeLatticePhonePrunedWrapper(
      trans_model_, &raw_lat, lat_beam, clat, decoder_opts_.det_opts);
}

void OnlineDecoder::GetBestPath(bool end_of_utterance,
                                              Lattice *best_path) const {
  decoder_.GetBestPath(best_path, end_of_utterance);
}

/*
bool OnlineDecoder::EndpointDetected(
    const OnlineEndpointConfig &config) {
  BaseFloat output_frame_shift =
      input_feature_frame_shift_in_seconds_ *
      decodable_.FrameSubsamplingFactor();
  return kaldi::EndpointDetected(config, trans_model_,
                                 output_frame_shift, decoder_);
}
*/

}
}