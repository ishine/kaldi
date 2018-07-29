#include "iot/decoder.h"

namespace kaldi {
namespace iot {

Decoder::Decoder(Wfst *fst,
                 const TransitionModel &trans_model,
                 const nnet3::DecodableNnetSimpleLoopedInfo &info,
                 OnlineNnet2FeaturePipeline *features,
                 const DecCoreConfig &dec_core_config,
                 const EndPointerConfig &end_pointer_config) :
  trans_model_(trans_model),
  decodable_(trans_model_, info, features->InputFeature(), features->IvectorFeature()),
  feature_frame_shift_in_sec_(features->FrameShiftInSeconds()),
  dec_core_config_(dec_core_config),
  dec_core_(fst, trans_model, dec_core_config_),
  end_pointer_config_(end_pointer_config),
  end_pointer_(end_pointer_config)
{ }


void Decoder::StartSession(const char* session_key) {
  dec_core_.InitDecoding();
}


void Decoder::Advance() {
  dec_core_.AdvanceDecoding(&decodable_);
}


void Decoder::StopSession() {
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

  if (!dec_core_config_.determinize_lattice)
    KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

  BaseFloat lat_beam = dec_core_config_.lattice_beam;
  DeterminizeLatticePhonePrunedWrapper(trans_model_, &raw_lat, lat_beam, clat, dec_core_config_.det_opts);
}


void Decoder::GetBestPath(bool use_final_prob, Lattice *best_path) const {
  dec_core_.GetBestPath(best_path, use_final_prob);
}


bool Decoder::EndpointDetected() {
  //BaseFloat frame_shift_in_sec = feature_frame_shift_in_sec_ * decodable_.FrameSubsamplingFactor();
  return end_pointer_.Detected(dec_core_);
}

}
}
