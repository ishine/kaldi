#include "iot/decoder.h"

namespace kaldi {
namespace iot {

Decoder::Decoder(Wfst *fst,
                 const TransitionModel &trans_model,
                 const nnet3::DecodableNnetSimpleLoopedInfo &info,
                 OnlineNnet2FeaturePipeline *features,
                 const DecCoreConfig &core_config) :
  trans_model_(trans_model),
  decodable_(trans_model_, info, features->InputFeature(), features->IvectorFeature()),
  core_config_(core_config),
  core_(fst, trans_model, core_config_),
  end_pointer_(NULL)
{ }


Decoder::~Decoder() {
  DELETE(end_pointer_);
}


void Decoder::EnableEndPointer(EndPointerConfig &end_pointer_config) {
  end_pointer_ = new EndPointer(end_pointer_config);
}


void Decoder::StartSession(const char* session_key) {
  core_.InitDecoding();
}


void Decoder::Advance() {
  core_.AdvanceDecoding(&decodable_);
}


int32 Decoder::NumFramesDecoded() const {
  return core_.NumFramesDecoded();
}


bool Decoder::EndpointDetected() {
  KALDI_ASSERT(end_pointer_ != NULL);
  return end_pointer_->Detected(core_);
}


void Decoder::StopSession() {
  core_.FinalizeDecoding();
}


void Decoder::GetLattice(bool use_final_prob, CompactLattice *clat) const {
  if (NumFramesDecoded() == 0)
    KALDI_ERR << "You cannot get a lattice if you decoded no frames.";
  Lattice raw_lat;
  core_.GetRawLattice(&raw_lat, use_final_prob);

  if (!core_config_.determinize_lattice)
    KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

  BaseFloat lat_beam = core_config_.lattice_beam;
  DeterminizeLatticePhonePrunedWrapper(trans_model_, &raw_lat, lat_beam, clat, core_config_.det_opts);
}


void Decoder::GetBestPath(bool use_final_prob, Lattice *best_path) const {
  core_.GetBestPath(best_path, use_final_prob);
}

} // namespace iot
} // namespace kaldi
