#include "iot/decoder.h"

namespace kaldi {
namespace iot {

Decoder::Decoder(Wfst *la_fst,
                 fst::DeterministicOnDemandFst<fst::StdArc> *lm_fst,
                 const TransitionModel &trans_model,
                 nnet3::AmNnetSimple &am_nnet,
                 const OnlineNnet2FeaturePipelineConfig &feature_config,
                 const nnet3::NnetSimpleLoopedComputationOptions &decodable_config,
                 const DecCoreConfig &core_config) :
  trans_model_(trans_model),
  feature_info_(feature_config),
  feature_(NULL),
  decodable_info_(decodable_config, &am_nnet),
  decodable_(NULL),
  core_config_(core_config),
  core_(la_fst, lm_fst, trans_model, core_config_),
  end_pointer_(NULL)
{ }


Decoder::~Decoder() {
  DELETE(feature_);
  DELETE(decodable_);
  DELETE(end_pointer_);
}


void Decoder::EnableEndPointer(EndPointerConfig &end_pointer_config) {
  end_pointer_ = new EndPointer(end_pointer_config);
}


void Decoder::StartSession(const char* session_key) {
  DELETE(feature_);
  feature_ = new OnlineNnet2FeaturePipeline(feature_info_);

  DELETE(decodable_);
  decodable_ = new nnet3::DecodableAmNnetLoopedOnline(trans_model_, 
                                                      decodable_info_, 
                                                      feature_->InputFeature(), 
                                                      feature_->IvectorFeature());
  core_.InitDecoding();
}


void Decoder::AcceptAudio(const void* data, int32 nbytes, AudioFormat audio_format) {
  KALDI_ASSERT(feature_ != NULL);

  int sample_rate;
  switch(audio_format) {
    case FLOAT_8K:
      sample_rate = 8000;
      break;
    case FLOAT_16K:
      sample_rate = 16000;
      break;
    default:
      sample_rate = -1;
  }
  KALDI_ASSERT(sample_rate != -1);

  BaseFloat* p = (BaseFloat*)data;
  int n = nbytes / sizeof(BaseFloat);
  
  SubVector<BaseFloat> audio_chunk(p, n);

  feature_->AcceptWaveform(sample_rate, audio_chunk);
  core_.AdvanceDecoding(decodable_);
}


int32 Decoder::NumFramesDecoded() const {
  return core_.NumFramesDecoded();
}


bool Decoder::EndpointDetected() {
  KALDI_ASSERT(end_pointer_ != NULL);
  return end_pointer_->Detected(core_);
}


void Decoder::StopSession() {
  feature_->InputFinished();
  core_.AdvanceDecoding(decodable_);
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
