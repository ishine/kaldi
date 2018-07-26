#include "iot/decoder.h"

namespace kaldi {
namespace iot {

Decoder::Decoder(
    const DecCoreConfig &decoder_opts,
    const TransitionModel &trans_model,
    const nnet3::DecodableNnetSimpleLoopedInfo &info,
    Wfst *fst,
    OnlineNnet2FeaturePipeline *features):
    decoder_opts_(decoder_opts),
    input_feature_frame_shift_in_seconds_(features->FrameShiftInSeconds()),
    trans_model_(trans_model),
    decodable_(trans_model_, info, features->InputFeature(), features->IvectorFeature()),
    dec_core_(fst, decoder_opts_) 
{ }

void Decoder::Initialize() {
  dec_core_.InitDecoding();
}

void Decoder::Advance() {
  dec_core_.AdvanceDecoding(&decodable_);
}

void Decoder::Finalize() {
  dec_core_.FinalizeDecoding();
}

int32 Decoder::NumFramesDecoded() const {
  return dec_core_.NumFramesDecoded();
}

void Decoder::GetLattice(bool use_final_prob, CompactLattice *clat) const {
  if (NumFramesDecoded() == 0)
    KALDI_ERR << "You cannot get a lattice if you decoded no frames.";
  Lattice raw_lat;
  dec_core_.GetRawLattice(&raw_lat, use_final_prob);

  if (!decoder_opts_.determinize_lattice)
    KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

  BaseFloat lat_beam = decoder_opts_.lattice_beam;
  DeterminizeLatticePhonePrunedWrapper(trans_model_, &raw_lat, lat_beam, clat, decoder_opts_.det_opts);
}

void Decoder::GetBestPath(bool use_final_prob, Lattice *best_path) const {
  dec_core_.GetBestPath(best_path, use_final_prob);
}

/*
bool Decoder::EndpointDetected(
    const OnlineEndpointConfig &config) {
  BaseFloat output_frame_shift =
      input_feature_frame_shift_in_seconds_ *
      decodable_.FrameSubsamplingFactor();
  return kaldi::EndpointDetected(config, trans_model_, output_frame_shift, decoder_);
}
*/

}
}
