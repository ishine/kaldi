// online0/online-ivector-feature.h

// Copyright 2013-2014   Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_ONLINE0_ONLINE_IVECTOR_FEATURE_H_
#define KALDI_ONLINE0_ONLINE_IVECTOR_FEATURE_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "gmm/diag-gmm.h"
#include "ivector/ivector-extractor.h"
#include "online0/online-feature.h"
#include "online0/online-nnet-feature-pipeline.h"


namespace kaldi {
/// @addtogroup  onlinefeat OnlineFeatureExtraction
/// @{

/// @file
/// This file contains code for online iVector extraction in a form compatible
/// with OnlineFeatureInterface.  It's used in online-nnet2-feature-pipeline.h.

/// This class includes configuration variables relating to the online iVector
/// extraction, but not including configuration for the "base feature",
/// i.e. MFCC/PLP/filterbank, which is an input to this feature.  This
/// configuration class can be used from the command line, but before giving it
/// to the code we create a config class called
/// OnlineIvectorExtractionInfo which contains the actual configuration
/// classes as well as various objects that are needed.  The principle is that
/// any code should be callable from other code, so we didn't want to force
/// configuration classes to be read from disk.
struct OnlineStreamIvectorExtractionConfig {

  OnlineNnetFeaturePipelineConfig base_feature_cfg;
  std::string diag_ubm_rxfilename;  // reads type DiagGmm.
  std::string full_ubm_rxfilename;  // reads tyep fgmm.
  std::string ivector_extractor_rxfilename;  // reads type IvectorExtractor
  std::string lda_transform_rxfilename; // read ivector lda transform matrix

  // the following four configuration values should in principle match those
  // given to the script extract_ivectors_online.sh, although none of them are
  // super-critical.
  int32 ivector_period;  // How frequently we re-estimate iVectors.
  int32 num_gselect;  // maximum number of posteriors to use per frame for
                      // iVector extractor.
  BaseFloat min_post;  // pruning threshold for posteriors for the iVector
                       // extractor.
  BaseFloat posterior_scale;  // Scale on posteriors used for iVector
                              // extraction; can be interpreted as the inverse
                              // of a scale on the log-prior.
  BaseFloat max_count;  // Maximum stats count we allow before we start scaling
                        // down stats (if nonzero).. this prevents us getting
                        // atypical-looking iVectors for very long utterances.
                        // Interpret this as a number of frames times
                        // posterior_scale, typically 1/10 of a frame count.

  int32 num_cg_iters;  // set to 15.  I don't believe this is very important, so it's
                       // not configurable from the command line for now.
  

  // If use_most_recent_ivector is true, we always return the most recent
  // available iVector rather than the one for the current frame.  This means
  // that if audio is coming in faster than we can process it, we will return a
  // more accurate iVector. 
  bool use_most_recent_ivector;

  // If true, always read ahead to NumFramesReady() when getting iVector stats.
  bool greedy_ivector_extractor;

  // max_remembered_frames is the largest number of frames it will remember
  // between utterances of the same speaker; this affects the output of
  // GetAdaptationState(), and has the effect of limiting the number of frames
  // of both the CMVN stats and the iVector stats.  Setting this to a smaller
  // value means the adaptation is less constrained by previous utterances
  // (assuming you provided info from a previous utterance of the same speaker
  // by calling SetAdaptationState()).
  BaseFloat max_remembered_frames;
  
  OnlineStreamIvectorExtractionConfig(): ivector_period(10), num_gselect(5),
                                   min_post(0.025), posterior_scale(0.1),
                                   max_count(0.0), num_cg_iters(15),
                                   use_most_recent_ivector(true),
                                   greedy_ivector_extractor(false),
                                   max_remembered_frames(1000) { }
  
  void Register(OptionsItf *opts) {
    base_feature_cfg.Register(opts);

    opts->Register("diag-ubm", &diag_ubm_rxfilename, "Filename of diagonal UBM "
                   "used to obtain posteriors for iVector extraction, e.g. "
                   "final.dubm");
    opts->Register("full-ubm", &full_ubm_rxfilename, "Filename of full covariance UBM "
                   "used to obtain posteriors for iVector extraction in speaker-id systems, e.g. "
                   "final.ubm");
    opts->Register("ivector-extractor", &ivector_extractor_rxfilename,
                   "Filename of iVector extractor, e.g. final.ie");
    opts->Register("ivector-lda-transform", &lda_transform_rxfilename,
                   "Filename of iVector lda transform matrix, e.g. transform.mat");
    opts->Register("ivector-period", &ivector_period, "Frequency with which "
                   "we extract iVectors for neural network adaptation");
    opts->Register("num-gselect", &num_gselect, "Number of Gaussians to select "
                   "for iVector extraction");
    opts->Register("min-post", &min_post, "Threshold for posterior pruning in "
                   "iVector extraction");
    opts->Register("posterior-scale", &posterior_scale, "Scale for posteriors in "
                   "iVector extraction (may be viewed as inverse of prior scale)");
    opts->Register("max-count", &max_count, "Maximum data count we allow before "
                   "we start scaling the stats down (if nonzero)... helps to make "
                   "iVectors from long utterances look more typical.  Interpret "
                   "as a frame-count times --posterior-scale, typically 1/10 of "
                   "a number of frames.  Suggest 100.");
    opts->Register("use-most-recent-ivector", &use_most_recent_ivector, "If true, "
                   "always use most recent available iVector, rather than the "
                   "one for the designated frame.");
    opts->Register("greedy-ivector-extractor", &greedy_ivector_extractor, "If "
                   "true, 'read ahead' as many frames as we currently have available "
                   "when extracting the iVector.  May improve iVector quality.");
    opts->Register("max-remembered-frames", &max_remembered_frames, "The maximum "
                   "number of frames of adaptation history that we carry through "
                   "to later utterances of the same speaker (having a finite "
                   "number allows the speaker adaptation state to change over "
                   "time).  Interpret as a real frame count, i.e. not a count "
                   "scaled by --posterior-scale.");
  }
};

/// This struct contains various things that are needed (as const references)
/// by class OnlineIvectorExtractor.
struct OnlineStreamIvectorExtractionInfo {
  DiagGmm diag_ubm;
  FullGmm full_ubm;
  IvectorExtractor extractor;
  Matrix<BaseFloat> lda_transform;
  Matrix<BaseFloat> lda_linear_term;
  Vector<BaseFloat> lda_constant_term;

  // the following configuration variables are copied from
  // OnlineIvectorExtractionConfig, see comments there.
  int32 ivector_period;
  int32 num_gselect;
  BaseFloat min_post;
  BaseFloat posterior_scale;
  BaseFloat max_count;
  int32 num_cg_iters;
  bool use_most_recent_ivector;
  bool greedy_ivector_extractor;
  BaseFloat max_remembered_frames;

  BaseFloat samp_freq;

  OnlineStreamIvectorExtractionInfo(const OnlineStreamIvectorExtractionConfig &config);

  void Init(const OnlineStreamIvectorExtractionConfig &config);

  // This constructor creates a version of this object where everything
  // is empty or zero.
  OnlineStreamIvectorExtractionInfo();

  void Check() const;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(OnlineStreamIvectorExtractionInfo);
};



/// OnlineIvectorFeature is an online feature-extraction class that's responsible
/// for extracting iVectors from raw features such as MFCC, PLP or filterbank.
/// Internally it processes the raw features using two different pipelines, one
/// online-CMVN+splice+LDA, and one just splice+LDA. It gets GMM posteriors from
/// the CMVN-normalized features, and with those and the unnormalized features
/// it obtains iVectors.

class OnlineStreamIvectorFeature: public OnlineStreamBaseFeature {
 public:
  /// Constructor.  base_feature is for example raw MFCC or PLP or filterbank
  /// features, whatever was used to train the iVector extractor.
  /// "info" contains all the configuration information as well as
  /// things like the iVector extractor that we won't be modifying.
  /// Caution: the class keeps a const reference to "info", so don't
  /// delete it while this class or others copied from it still exist.
  explicit OnlineStreamIvectorFeature(const OnlineStreamIvectorExtractionInfo &info,
		  	  	  	  	  	  	  	  OnlineStreamBaseFeature *base_feature);

  virtual ~OnlineStreamIvectorFeature();

  // Member functions from OnlineFeatureInterface:

  /// Dim() will return the iVector dimension.
  virtual int32 Dim() const;
  virtual bool IsLastFrame(int32 frame) const;
  virtual int32 NumFramesReady() const;
  virtual BaseFloat FrameShiftInSeconds() const;
  virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat);
  virtual void Reset() {};

	/// Accept more data to process.  It won't actually process it until you call
	/// GetFrame() [probably indirectly via (decoder).AdvanceDecoding()], when you
	/// call this function it will just copy it).  sampling_rate is necessary just
	/// to assert it equals what's in the config.
	void AcceptWaveform(BaseFloat sampling_rate, const VectorBase<BaseFloat> &waveform);

	// InputFinished() tells the class you won't be providing any
	// more waveform.  This will help flush out the last few frames
	// of delta or LDA features, and finalize the pitch features
	// (making them more accurate).
	void InputFinished();

  // Some diagnostics (not present in generic interface):
  // UBM log-like per frame:
  BaseFloat UbmLogLikePerFrame() const;
  // Objective improvement per frame from iVector estimation, versus default iVector
  // value, measured at utterance end.
  BaseFloat ObjfImprPerFrame() const;

  // returns number of frames seen (but not counting the posterior-scale).
  BaseFloat NumFrames() const {
    return ivector_stats_.NumFrames() / info_.posterior_scale;
  }

  void PrintDiagnostics() const;
  
 private:
  // the stats for frame "frame".
  void UpdateStatsForFrame(int32 frame);

  // This is the original UpdateStatsUntilFrame that is called when there is
  // no data-weighting involved.
  void UpdateStatsUntilFrame(int32 frame);

  const OnlineStreamIvectorExtractionInfo &info_;

  OnlineStreamBaseFeature *base_feature_;

  /// the iVector estimation stats
  OnlineIvectorEstimationStats ivector_stats_;

  /// num_frames_stats_ is the number of frames of data we have already
  /// accumulated from this utterance and put in ivector_stats_.  Each frame t <
  /// num_frames_stats_ is in the stats.  In case you are doing the
  /// silence-weighted iVector estimation, with UpdateFrameWeights() being
  /// called, this variable is still used but you may later have to revisit
  /// earlier frames to adjust their weights... see the code.
  int32 num_frames_stats_;
  
  /// The following is only needed for diagnostics.
  double tot_ubm_loglike_;
  
  /// Most recently estimated iVector, will have been
  /// estimated at the greatest time t where t <= num_frames_stats_ and
  /// t % info_.ivector_period == 0.
  Vector<double> current_ivector_;
  
  /// if info_.use_most_recent_ivector == false, we need to store
  /// the iVector we estimated each info_.ivector_period frames so that
  /// GetFrame() can return the iVector that was active on that frame.
  /// ivectors_history_[i] contains the iVector we estimated on
  /// frame t = i * info_.ivector_period.
  std::vector<Vector<BaseFloat>* > ivectors_history_;
};

/// @} End of "addtogroup onlinefeat"
}  // namespace kaldi

#endif  // KALDI_ONLINE0_ONLINE_IVECTOR_FEATURE_H_

