// online0/online-ivector-feature.cc

// Copyright 2014  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "online0/online-ivector-feature.h"

namespace kaldi {

OnlineStreamIvectorExtractionInfo::OnlineStreamIvectorExtractionInfo(
    const OnlineStreamIvectorExtractionConfig &config) {
  Init(config);
}

void OnlineStreamIvectorExtractionInfo::Init(
	const OnlineStreamIvectorExtractionConfig &config) {

  ivector_period = config.ivector_period;
  num_gselect = config.num_gselect;
  min_post = config.min_post;
  posterior_scale = config.posterior_scale;
  max_count = config.max_count;
  num_cg_iters = config.num_cg_iters;
  use_most_recent_ivector = config.use_most_recent_ivector;
  greedy_ivector_extractor = config.greedy_ivector_extractor;
  if (greedy_ivector_extractor && !use_most_recent_ivector) {
    KALDI_WARN << "--greedy-ivector-extractor=true implies "
               << "--use-most-recent-ivector=true";
    use_most_recent_ivector = true;
  }
  max_remembered_frames = config.max_remembered_frames;
  normalize = config.normalize;

  std::string note = "(note: this may be needed "
      "in the file supplied to --ivector-extractor-config)";

  if (config.diag_ubm_rxfilename == "")
	  KALDI_ERR << "--diag-ubm option must be set " << note;
  ReadKaldiObject(config.diag_ubm_rxfilename, &diag_ubm);
  if (config.full_ubm_rxfilename == "")
	  KALDI_ERR << "--full-ubm option must be set " << note;
  ReadKaldiObject(config.full_ubm_rxfilename, &full_ubm);
  if (config.ivector_extractor_rxfilename == "")
	  KALDI_ERR << "--ivector-extractor option must be set " << note;
  ReadKaldiObject(config.ivector_extractor_rxfilename, &extractor);
  if (config.lda_transform_rxfilename != "") {
	  ReadKaldiObject(config.lda_transform_rxfilename, &lda_transform);
	  lda_linear_term = lda_transform.ColRange(0, lda_transform.NumCols()-1);
	  lda_constant_term.Resize(lda_transform.NumRows());
	  lda_constant_term.CopyColFromMat(lda_transform, lda_transform.NumCols()-1);
  }

  this->Check();
}


void OnlineStreamIvectorExtractionInfo::Check() const {
  KALDI_ASSERT(diag_ubm.Dim() == extractor.FeatDim());
  KALDI_ASSERT(full_ubm.Dim() == extractor.FeatDim());
  KALDI_ASSERT(ivector_period > 0);
  KALDI_ASSERT(num_gselect > 0);
  KALDI_ASSERT(min_post < 0.5);
  // posterior scale more than one does not really make sense.
  KALDI_ASSERT(posterior_scale > 0.0 && posterior_scale <= 1.0);
  KALDI_ASSERT(max_remembered_frames >= 0);
}

// The class constructed in this way should never be used.
OnlineStreamIvectorExtractionInfo::OnlineStreamIvectorExtractionInfo():
    ivector_period(0), num_gselect(0), min_post(0.0), posterior_scale(0.0),
	max_count(0.0), use_most_recent_ivector(true), greedy_ivector_extractor(false),
    max_remembered_frames(0), samp_freq(16000) { }

int32 OnlineStreamIvectorFeature::Dim() const {
	return info_.extractor.IvectorDim();
}

bool OnlineStreamIvectorFeature::IsLastFrame(int32 frame) const {
  // Note: it might be more logical to return, say, lda_->IsLastFrame()
  // since this is the feature the iVector extractor directly consumes,
  // but it will give the same answer as base_->IsLastFrame() anyway.
  // [note: the splicing component pads at begin and end so it always
  // returns the same number of frames as its input.]
  return base_feature_->IsLastFrame(frame);
}

int32 OnlineStreamIvectorFeature::NumFramesReady() const {
  KALDI_ASSERT(base_feature_ != NULL);
  return base_feature_->NumFramesReady();
}

BaseFloat OnlineStreamIvectorFeature::FrameShiftInSeconds() const {
  return base_feature_->FrameShiftInSeconds();
}

void OnlineStreamIvectorFeature::UpdateStatsForFrame(int32 t) {
  int32 feat_dim = base_feature_->Dim();
  Vector<BaseFloat> feat(feat_dim);  // features given to iVector extractor
  Vector<BaseFloat> log_likes;

  base_feature_->GetFrame(t, &feat);

  std::vector<int32> gselect;
  info_.diag_ubm.GaussianSelection(feat, info_.num_gselect, &gselect);
  //info_.diag_ubm.LogLikelihoods(feat, &log_likes);
  info_.full_ubm.LogLikelihoodsPreselect(feat, gselect, &log_likes);

  // "posterior" stores the pruned posteriors for Gaussians in the UBM.
  std::vector<std::pair<int32, BaseFloat> > posterior;
  tot_ubm_loglike_ += VectorToPosteriorEntry(log_likes, gselect, info_.min_post, &posterior);

  for (size_t i = 0; i < posterior.size(); i++)
	  posterior[i].second *= info_.posterior_scale;

  ivector_stats_.AccStats(info_.extractor, feat, posterior);
}

void OnlineStreamIvectorFeature::UpdateStatsUntilFrame(int32 frame) {
  KALDI_ASSERT(frame >= 0 && frame < this->NumFramesReady());

  int32 ivector_period = info_.ivector_period;
  int32 num_cg_iters = info_.num_cg_iters;

  for (; num_frames_stats_ <= frame; num_frames_stats_++) {
    int32 t = num_frames_stats_;
    UpdateStatsForFrame(t);
    if ((!info_.use_most_recent_ivector && t % ivector_period == 0) ||
        (info_.use_most_recent_ivector && t == frame)) {
      ivector_stats_.GetIvector(num_cg_iters, &current_ivector_);
      if (!info_.use_most_recent_ivector) {  // need to cache iVectors.
        int32 ivec_index = t / ivector_period;
        KALDI_ASSERT(ivec_index == static_cast<int32>(ivectors_history_.size()));
        ivectors_history_.push_back(new Vector<BaseFloat>(current_ivector_));
      }
    }
  }
}


void OnlineStreamIvectorFeature::GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
  int32 frame_to_update_until = (info_.greedy_ivector_extractor ?
		  	  	  	  	  	  	  base_feature_->NumFramesReady() - 1 : frame);

  UpdateStatsUntilFrame(frame_to_update_until);

  KALDI_ASSERT(feat->Dim() == this->Dim());

  // ivector
  Vector<BaseFloat> ivector(this->Dim());
  if (info_.use_most_recent_ivector) {
    KALDI_VLOG(5) << "due to --use-most-recent-ivector=true, using iVector "
                  << "from frame " << num_frames_stats_ << " for frame "
                  << frame;
    // use the most recent iVector we have, even if 'frame' is significantly in
    // the past.
    ivector.CopyFromVec(current_ivector_);
    // Subtract the prior-mean from the first dimension of the output feature so
    // it's approximately zero-mean.
    ivector(0) -= info_.extractor.PriorOffset();
  } else {
    int32 i = frame / info_.ivector_period;  // rounds down.
    // if the following fails, UpdateStatsUntilFrame would have a bug.
    KALDI_ASSERT(static_cast<size_t>(i) < ivectors_history_.size());
    ivector.CopyFromVec(*(ivectors_history_[i]));
    ivector(0) -= info_.extractor.PriorOffset();
  }

  feat->CopyFromVec(ivector);
}

void OnlineStreamIvectorFeature::LdaTransform(const VectorBase<BaseFloat> &ivector,
		Vector<BaseFloat> &transformed_ivector) {
	  // Normalize length of iVectors to equal sqrt(feature-dimension)
	  int32 lda_dim = info_.lda_transform.NumRows();
	  VectorBase<BaseFloat> norm_ivector(ivector);
	  BaseFloat norm = norm_ivector.Norm(2.0);
	  BaseFloat ratio = norm / sqrt(norm_ivector.Dim()); // how much larger it is
														  // than it would be, in
														  // expectation, if normally
	  if (ratio != 0.0 && info_.normalize)
		  norm_ivector.Scale(1.0 / ratio);

	  // ivector lda transform
	  transformed_ivector.Resize(lda_dim);
	  if (ivector.Dim() == info_.lda_transform.NumCols()) {
		  transformed_ivector.AddMatVec(1.0, info_.lda_transform, kNoTrans, norm_ivector, 0.0);
	  } else {
		  KALDI_ASSERT(norm_ivector.Dim() == info_.lda_transform.NumCols()-1);
		  transformed_ivector.CopyFromVec(info_.lda_constant_term);
		  transformed_ivector.AddMatVec(1.0, info_.lda_linear_term, kNoTrans, norm_ivector, 1.0);
	  }

	  // Normalize
	  norm = transformed_ivector.Norm(2.0);
	  ratio = norm / sqrt(transformed_ivector.Dim());
	  if (ratio != 0.0 && info_.normalize)
		  transformed_ivector.Scale(1.0 / ratio);
}

int32 OnlineStreamIvectorFeature::LdaDim() {
	return info_.lda_transform.NumRows();
}

void OnlineStreamIvectorFeature::PrintDiagnostics() const {
  if (num_frames_stats_ == 0) {
    KALDI_VLOG(3) << "Processed no data.";
  } else {
    KALDI_VLOG(3) << "UBM log-likelihood was "
                  << (tot_ubm_loglike_ / NumFrames())
                  << " per frame, over " << NumFrames()
                  << " frames.";

    Vector<BaseFloat> temp_ivector(current_ivector_);
    temp_ivector(0) -= info_.extractor.PriorOffset();

    KALDI_VLOG(3) << "By the end of the utterance, objf change/frame "
                  << "from estimating iVector (vs. default) was "
                  << ivector_stats_.ObjfChange(current_ivector_)
                  << " and iVector length was "
                  << temp_ivector.Norm(2.0);
  }
}

OnlineStreamIvectorFeature::~OnlineStreamIvectorFeature() {
  // Delete objects owned here.
  // base_ is not owned here so don't delete it.
  for (size_t i = 0; i < ivectors_history_.size(); i++)
    delete ivectors_history_[i];
}


OnlineStreamIvectorFeature::OnlineStreamIvectorFeature(
    const OnlineStreamIvectorExtractionInfo &info,
    OnlineStreamBaseFeature *base_feature):
    info_(info), base_feature_(base_feature),
    ivector_stats_(info_.extractor.IvectorDim(),
                   info_.extractor.PriorOffset(),
                   info_.max_count),
    num_frames_stats_(0), tot_ubm_loglike_(0.0) {

  info.Check();
  KALDI_ASSERT(base_feature_ != NULL);

  // Set the iVector to its default value, [ prior_offset, 0, 0, ... ].
  current_ivector_.Resize(info_.extractor.IvectorDim());
  current_ivector_(0) = info_.extractor.PriorOffset();
}

BaseFloat OnlineStreamIvectorFeature::UbmLogLikePerFrame() const {
  if (NumFrames() == 0) return 0;
  else return tot_ubm_loglike_ / NumFrames();
}

BaseFloat OnlineStreamIvectorFeature::ObjfImprPerFrame() const {
  return ivector_stats_.ObjfChange(current_ivector_);
}

void OnlineStreamIvectorFeature::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
  base_feature_->AcceptWaveform(sampling_rate, waveform);
}

void OnlineStreamIvectorFeature::InputFinished() {
  base_feature_->InputFinished();
}

}  // namespace kaldi
