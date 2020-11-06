// decoder/lattice-biglm-faster-online-bucket-decoder.h

// Copyright 2020 ASLP (Author: Hang Lyu)

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

// see note at the top of lattice-faster-decoder.h, about how to maintain this
// file in sync with lattice-faster-decoder.h


#ifndef KALDI_DECODER_LATTICE_BIGLM_FASTER_ONLINE_BUCKET_DECODER_H_
#define KALDI_DECODER_LATTICE_BIGLM_FASTER_ONLINE_BUCKET_DECODER_H_

#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
// Use the same configuration class as LatticeFasterDecoder.
#include "decoder/lattice-biglm-faster-bucket-decoder.h"

namespace kaldi {
/** LatticeBiglmFasterOnlineBucketDecoderTpl is as
    LatticeBiglmFasterBucketDecoder but also supports an efficient way to get
    the best path (see the function BestPathEnd()), which is useful in
    endpointing and in situation where you might want to frequently access the
    best path.

    The 'Token' type is required to be BackpointerToken, so this is only
    templated on the FST type.

    Actually, for the 'Bucket' decoder, we use the BackwardLinks of buckets to
    achieve the partial results and use the BackpointerTokens to collect the
    final best path after FinalizeDecoding.
**/
template <typename FST>
class LatticeBiglmFasterOnlineBucketDecoderTpl:
    public LatticeBiglmFasterBucketDecoderTpl<
      FST, kaldi::biglmbucketdecoder::BackpointerToken<FST> > {
 public:
  double process_time = 0.0;

  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using Token = typename kaldi::biglmbucketdecoder::BackpointerToken<FST>;
  using Bucket = typename kaldi::biglmbucketdecoder::TokenBucket<FST, Token>;
  using BackwardLinkT =
    typename kaldi::biglmbucketdecoder::BackwardLink<Token>;
  using BackwardBucketLinkT =
    typename kaldi::biglmbucketdecoder::BackwardLink<Bucket>;
  using ForwardLinkT =
    typename kaldi::biglmbucketdecoder::ForwardLink<Token>;
  using ForwardBucketLinkT =
    typename kaldi::biglmbucketdecoder::ForwardLink<Bucket>;

  struct BestPathIterator {
    // The category is used to decide the specific type of the 'elem' pointer
    enum Category { kUnknown = 0, kToken  = 1, kBucket = 2 };

    // Note: 'frame' is the frame-index of the frame you'll get the
    // transition-id for next time, if you call TraceBackBestPath on this
    // iterator (assuming it's not an epsilon transition). Note that this
    // is one less than you might reasonably expect, e.g. it's -1 for the
    // nonemitting transitions before the first frame.
    BestPathIterator(void *e, int32 f, Category c):
      elem(e), frame(f), category(c) { }
    bool Done() const { return elem == NULL; }

    void *elem;
    int32 frame;
    Category category;
  };


  // Instantiate this class once for each thing you have to decode.
  // This version of the constructor does not take ownership of 'fst'.
  LatticeBiglmFasterOnlineBucketDecoderTpl(
      const FST &fst,
      const LatticeBiglmFasterBucketDecoderConfig &config,
      fst::DeterministicOnDemandFst<Arc> *lm_diff_fst):
    LatticeBiglmFasterBucketDecoderTpl<FST, Token>(fst, config, lm_diff_fst) { }


  // This version of the constructor takes ownership of the fst, and will delete
  // it when this object is destroyed.
  LatticeBiglmFasterOnlineBucketDecoderTpl(
      const LatticeBiglmFasterBucketDecoderConfig &config,
      FST *fst,
      fst::DeterministicOnDemandFst<Arc> *lm_diff_fst):
    LatticeBiglmFasterBucketDecoderTpl<FST, Token>(config, fst, lm_diff_fst) { }


  // Outputs a FST corresponding to the single best path through the lattice.
  // This is efficient because it doesn't get the entire raw lattice and find
  // the best path through it. Instead, it finds the 'best end' and traces it
  // back through the lattice.
  bool GetBestPath(Lattice *ofst, bool use_final_probs = true) const;


  // This function returns an iterator that can be used to trace back the best
  // path.
  // If use_final_probs == true and at least one final state survived till the
  // end, it will use the final-probs in working out the best final Token, and
  // will output the final cost to *final_cost_out (if non-NULL), else it will
  // use only the forward likelihood.
  // Note: if decoding_finalized_ = true, we find the best token on the last
  // token list, which also means the token-level final cost is employed.
  // Otherwise, we find the best bucket on the last but two bucket list (as
  // we process the non-emitting and emitting arcs at the same time so the
  // the last bucket list is not completed) and the bucket-level final cost
  // is included if needed.
  // Required NumFramesDecoded() > 0
  BestPathIterator BestPathEnd(bool use_final_probs,
                               BaseFloat *final_cost_out = NULL) const;


  // This function can be used in conjunction with BestPathEnd() to trace back
  // the best path one link at a time.
  // The return value is the updated iterator. It outputs the ilabel and olabel,
  // and the weight (graph and acoustic) to the arc pointer, while leaving its
  // 'nextstate' variable unchanged.
  BestPathIterator TraceBackBestPath(const BestPathIterator &iter,
                                     LatticeArc *oarc) const;


 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeBiglmFasterOnlineBucketDecoderTpl);
};

typedef LatticeBiglmFasterOnlineBucketDecoderTpl<fst::StdFst>
  LatticeBiglmFasterOnlineBucketDecoder;

}  // end namespacke kaldi.

#endif
