// decoder/lattice-biglm-faster-online-bucket-decoder.cc

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

#include "decoder/lattice-biglm-faster-online-bucket-decoder.h"
#include "lat/lattice-functions.h"

namespace kaldi {

template <typename FST>
bool LatticeBiglmFasterOnlineBucketDecoderTpl<FST>::GetBestPath(
    Lattice *olat, bool use_final_probs) const {
  olat->DeleteStates();
  BaseFloat final_graph_cost;
  BestPathIterator iter = BestPathEnd(use_final_probs, &final_graph_cost);
  if (iter.Done())
    return false;  // would have printed warning.
  StateId state = olat->AddState();
  olat->SetFinal(state, LatticeWeight(final_graph_cost, 0.0));
  while (!iter.Done()) {
    LatticeArc arc;
    iter = TraceBackBestPath(iter, &arc);
    arc.nextstate = state;
    StateId new_state = olat->AddState();
    olat->AddArc(new_state, arc);
    state = new_state;
  }
  olat->SetStart(state);
  return true;
}


template <typename FST>
typename LatticeBiglmFasterOnlineBucketDecoderTpl<FST>::BestPathIterator
LatticeBiglmFasterOnlineBucketDecoderTpl<FST>::BestPathEnd(
    bool use_final_probs, BaseFloat *final_cost_out) const {
  if (this->decoding_finalized_ && !use_final_probs) {
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "BestPathEnd() with use_final_probs == false";
  }
  KALDI_ASSERT(this->NumFramesDecoded() > 0 &&
               "You cannot call BestPathEnd if no frames were decoded.");
  BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();  // for short
  
  if (this->decoding_finalized_) {  // on the last token list
    // All buckets should have been filled. We find the best token on the last
    // token list.
    Token *best_tok = NULL;
    BaseFloat best_cost = infinity, best_final_cost = infinity;

    // redundant but for speed
    if (use_final_probs) {  // any token with final cost is prior to without one
      for (Token *tok = this->active_toks_.back().toks; tok != NULL;
           tok = tok->next) {
        StateId base_state = tok->base_state,
                lm_state = tok->lm_state;
        BaseFloat tok_final_cost = this->fst_->Final(base_state).Value() +
                                   this->lm_diff_fst_->Final(lm_state).Value();
        BaseFloat tok_cost = tok->tot_cost;
        BaseFloat tok_cost_with_final = tok_cost + tok_final_cost;
        if (best_final_cost != infinity) {  // the best token has final cost
          if (tok_cost_with_final < best_cost) {
            best_tok = tok;
            best_cost = tok_cost_with_final;
            best_final_cost = tok_final_cost;
          }
        } else {  // the best token's final cost is infinity
          if (tok_cost_with_final != infinity) {  // the first token with final
            best_tok = tok;
            best_cost = tok_cost;
            best_final_cost = tok_final_cost;
          } else if (tok_cost < best_cost) {
            best_tok = tok;
            best_cost = tok_cost;
          }  // else, tok is worse than the best_tok
        }
      }
    } else {  // only consider the alpha value
      for (Token *tok = this->active_toks_.back().toks; tok != NULL;
           tok = tok->next) {
        BaseFloat tok_cost = tok->tot_cost;
        if (tok_cost < best_cost) {
          best_tok = tok;
          best_cost = tok_cost;
        }
      }
    }
    KALDI_ASSERT(!use_final_probs && best_final_cost == infinity);
    if (best_tok == NULL) {  // this should not happen, and is likely a code
                             // error or caused by infinities in likelihoods.
      KALDI_ERR << "No best token found.";
    }
    if (final_cost_out) *final_cost_out = best_final_cost;
    return BestPathIterator(best_tok, this->NumFramesDecoded() - 1,
                            BestPathIterator::kToken);
  } else {  // on the last but two bucket list
    // Assume we use ProcessForFrame in decoder to process the emitting arcs
    // and non-emitting arcs together. So the last bucket list is half-baked.
    Bucket *best_bucket = NULL;
    BaseFloat best_cost = infinity, best_final_cost = infinity;

    // redundant but for speed
    if (use_final_probs) {  // any bucket with final cost has higher priority
      for (Bucket *bucket =
           this->active_buckets_[this->active_buckets_.size() - 2].buckets;
           bucket != NULL; bucket = bucket->next) {
        StateId base_state = bucket->base_state;
        BaseFloat bucket_final_cost = this->fst_->Final(base_state).Value();
        BaseFloat bucket_cost = bucket->tot_cost;
        BaseFloat bucket_cost_with_final = bucket_cost + bucket_final_cost;
        if (best_final_cost != infinity) {  // the best one has final cost
          if (bucket_cost_with_final < best_cost) {
            best_bucket = bucket;
            best_cost = bucket_cost_with_final;
            best_final_cost = bucket_final_cost;
          }
        } else {  // the best one's final cost is inifinity
          if (bucket_final_cost != infinity) {  // the first one reach final
            best_bucket = bucket;
            best_cost = bucket_cost_with_final;
            best_final_cost = bucket_final_cost;
          } else if (bucket_cost < best_cost) {  // only compare non-final alpha
            best_bucket = bucket;
            best_cost = bucket_cost;
          }
        }
      }
    } else {  // only consider the alpha value
      for (Bucket *bucket =
           this->active_buckets_[this->active_buckets_.size() - 2].buckets;
           bucket != NULL; bucket = bucket->next) {
        BaseFloat bucket_cost = bucket->tot_cost;
        if (bucket_cost < best_cost) {
          best_bucket = bucket;
          best_cost = bucket_cost;
        }
      }
    }
    KALDI_ASSERT(!use_final_probs && best_final_cost == infinity);
    if (best_bucket == NULL) {  // this should not happen, and is likely a code
                                // error or caused by infinites in likelihoods.
      KALDI_ERR << "No best bucket found.";
    }
    if (final_cost_out) *final_cost_out = best_final_cost;
    if (best_bucket->expanded) {
      // This bucket has been expanded. Use the best token.
      Token *best_tok = NULL;
      for (typename std::vector<Token*>::iterator it =
           best_bucket->top_toks.begin(); it != best_bucket->top_toks.end();
           it++) {
        if (best_bucket->tot_cost == (*it)->tot_cost) {
          best_tok = *it;
          break;
        }
      }
      KALDI_ASSERT(best_tok != NULL);
      return BestPathIterator(best_tok, this->NumFramesDecoded() - 2,
                              BestPathIterator::kToken);
    } else {
      return BestPathIterator(best_bucket, this->NumFramesDecoded() - 2,
                              BestPathIterator::kBucket);
    }
  }
  // should not reach
  return BestPathIterator(NULL, -1, BestPathIterator::kUnknown);
}


template <typename FST>
typename LatticeBiglmFasterOnlineBucketDecoderTpl<FST>::BestPathIterator
LatticeBiglmFasterOnlineBucketDecoderTpl<FST>::TraceBackBestPath(
    const BestPathIterator &iter, LatticeArc *oarc) const {
  KALDI_ASSERT(!iter.Done() && oarc != NULL);
  int32 cur_t = iter.frame, ret_t = cur_t;
  if (iter.category == BestPathIterator::kToken) {
    Token *tok = static_cast<Token*>(iter.elem);
    // We use the backpointer to find the preceding token
    if (tok->backpointer != NULL) {
      ForwardLinkT *link;
      for (link = tok->backpointer->links; link != NULL; link = link->next) {
        if (link->next_elem == tok) {  // link to 'tok'
          oarc->ilabel = link->ilabel;
          oarc->olabel = link->olabel;
          BaseFloat graph_cost = link->graph_cost,
                    acoustic_cost = link->acoustic_cost;
          if (oarc->ilabel != 0) {  // add back offset
            KALDI_ASSERT(static_cast<size_t>(cur_t) <
                         this->cost_offsets_.size());
            acoustic_cost -= this->cost_offsets_[cur_t];
            --ret_t;
          }
          oarc->weight = LatticeWeight(graph_cost, acoustic_cost);
          break;
        }
      }
      if (link == NULL) {  // Didn't find correct link
        KALDI_ERR << "Error tracing best-path back";
      }
    } else {  // the start token
      oarc->ilabel = 0;
      oarc->olabel = 0;
      oarc->weight = LatticeWeight::One();  // zero costs
    }
    return BestPathIterator(tok->backpointer, ret_t, BestPathIterator::kToken);
  } else if (iter.category == BestPathIterator::kBucket) {
    Bucket *bucket = static_cast<Bucket*>(iter.elem);
    Bucket *pre_bucket = NULL;
    // Iterate the backwardlinks of the bucket to find the preceding bucket
    BackwardBucketLinkT *blink = NULL;
    for (blink = bucket->bucket_backward_links; blink != NULL;
         blink = blink->next) {
      pre_bucket = blink->prev_elem;
      if (ApproxEqual(pre_bucket->tot_cost + blink->acoustic_cost +
                      blink->graph_cost, bucket->tot_cost)) {
        oarc->ilabel = blink->ilabel;
        oarc->olabel = blink->olabel;
        BaseFloat graph_cost = blink->graph_cost,
                  acoustic_cost = blink->acoustic_cost;
        if (oarc->ilabel != 0) {
          KALDI_ASSERT(static_cast<size_t>(cur_t) < this->cost_offsets_.size());
          acoustic_cost -= this->cost_offsets_[cur_t];
          --ret_t;
        }
        oarc->weight = LatticeWeight(graph_cost, acoustic_cost);
        break;
      }
    }

    if (pre_bucket->expanded) {  // This bucket has been expanded. Use the token
      Token *pre_tok = NULL;
      for (typename std::vector<Token*>::iterator it =
           pre_bucket->top_toks.begin(); it != pre_bucket->top_toks.end();
           it++) {
        if (pre_bucket->tot_cost == (*it)->tot_cost) {
          pre_tok = *it;
          break;
        }
      }
      KALDI_ASSERT(pre_tok != NULL);
      return BestPathIterator(pre_tok, ret_t, BestPathIterator::kToken);
    } else {
      KALDI_ASSERT(pre_bucket != NULL);
      return BestPathIterator(pre_bucket, ret_t, BestPathIterator::kBucket);
    }
  } else {
    KALDI_ASSERT(iter.category == BestPathIterator::kUnknown);
    KALDI_ERR << "An unknown best path iterator is reached.";
  }
  return BestPathIterator(NULL, -1, BestPathIterator::kUnknown);
}

// Instantiate the template for the FST types that we'll need.
template class LatticeBiglmFasterOnlineBucketDecoderTpl<fst::Fst<fst::StdArc> >;
template class LatticeBiglmFasterOnlineBucketDecoderTpl<
  fst::VectorFst<fst::StdArc> >;
template class LatticeBiglmFasterOnlineBucketDecoderTpl<
  fst::ConstFst<fst::StdArc> >;
}  // end namespace kaldi
