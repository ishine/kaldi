// decoder/lattice-biglm-faster-bucket-decoder.cc

// Copyright 2013-2019  Johns Hopkins University (Author: Daniel Povey)
//                2019  Hang Lyu               

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

#include "decoder/lattice-biglm-faster-bucket-decoder.h"
#include "lat/lattice-functions.h"

namespace kaldi {

template<typename Element>
BucketQueue<Element>::BucketQueue(BaseFloat cost_scale) :
    cost_scale_(cost_scale) {
  // NOTE: we reserve plenty of elements to avoid expensive reallocations
  // later on. Normally, the size is a little bigger than (adaptive_beam +
  // 15) * cost_scale.
  int32 bucket_size = (15 + 20) * cost_scale_;
  buckets_.resize(bucket_size);
  bucket_offset_ = 15 * cost_scale_;
  first_nonempty_bucket_index_ = bucket_size - 1;
  first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
  bucket_size_tolerance_ = 1.2 * bucket_size;
}

template<typename Element>
void BucketQueue<Element>::Push(Element *elem) {
  size_t bucket_index = std::floor(elem->tot_cost * cost_scale_) +
                        bucket_offset_;
  if (bucket_index >= buckets_.size()) {
    int32 margin = 10;  // a margin which is used to reduce re-allocate
                        // space frequently
    if (static_cast<int32>(bucket_index) > 0) {
      buckets_.resize(bucket_index + margin);
    } else {  // less than 0
      int32 increase_size = - static_cast<int32>(bucket_index) + margin;
      buckets_.resize(buckets_.size() + increase_size);
      // translation
      for (size_t i = buckets_.size() - 1; i >= increase_size; i--) {
        buckets_[i].swap(buckets_[i - increase_size]);
      }
      bucket_offset_ = bucket_offset_ + increase_size;
      bucket_index += increase_size;
      first_nonempty_bucket_index_ = bucket_index;
    }
    first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
  }
  elem->in_queue = true;
  buckets_[bucket_index].push_back(elem);
  if (bucket_index < first_nonempty_bucket_index_) {
    first_nonempty_bucket_index_ = bucket_index;
    first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
  }
}

template<typename Element>
Element* BucketQueue<Element>::Pop() {
  while (true) {
    if (!first_nonempty_bucket_->empty()) {
      Element *ans = first_nonempty_bucket_->back();
      first_nonempty_bucket_->pop_back();
      if (ans->in_queue) {  // If ans->in_queue is false, this means it is a
                            // duplicate instance of this Token that was left
                            // over when a Token's best_cost changed, and the
                            // Token has already been processed(so conceptually,
                            // it is not in the queue).
        ans->in_queue = false;
        return ans;
      }
    }
    if (first_nonempty_bucket_->empty()) {
      for (; first_nonempty_bucket_index_ + 1 < buckets_.size();
           first_nonempty_bucket_index_++) {
        if (!buckets_[first_nonempty_bucket_index_].empty()) break;
      }
      first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
      if (first_nonempty_bucket_->empty()) return NULL;
    }
  }
}

template<typename Element>
void BucketQueue<Element>::Clear() {
  for (size_t i = first_nonempty_bucket_index_; i < buckets_.size(); i++) {
    buckets_[i].clear();
  }
  if (buckets_.size() > bucket_size_tolerance_) {
    buckets_.resize(bucket_size_tolerance_);
    bucket_offset_ = 15 * cost_scale_;
  }
  first_nonempty_bucket_index_ = buckets_.size() - 1;
  first_nonempty_bucket_ = &buckets_[first_nonempty_bucket_index_];
}

// instantiate this class once for each thing you have to decode.
template <typename FST, typename Token>
LatticeBiglmFasterBucketDecoderTpl<FST, Token>::LatticeBiglmFasterBucketDecoderTpl(
    const FST &fst,
    const LatticeBiglmFasterBucketDecoderConfig &config,
    fst::DeterministicOnDemandFst<Arc> *lm_diff_fst):
    fst_(&fst), delete_fst_(false), lm_diff_fst_(lm_diff_fst), config_(config),
    num_toks_(0), cur_queue_(config_.cost_scale) {
  config_.Check();
  KALDI_ASSERT(fst_->Start() != fst::kNoStateId &&
               lm_diff_fst_->Start() != fst::kNoStateId);
}


template <typename FST, typename Token>
LatticeBiglmFasterBucketDecoderTpl<FST, Token>::LatticeBiglmFasterBucketDecoderTpl(
    const LatticeBiglmFasterBucketDecoderConfig &config, FST *fst,
    fst::DeterministicOnDemandFst<Arc> *lm_diff_fst):
    fst_(fst), delete_fst_(true), lm_diff_fst_(lm_diff_fst), config_(config),
    num_toks_(0), cur_queue_(config_.cost_scale) {
  config_.Check();
  KALDI_ASSERT(fst_->Start() != fst::kNoStateId &&
               lm_diff_fst_->Start() != fst::kNoStateId);
}


template <typename FST, typename Token>
LatticeBiglmFasterBucketDecoderTpl<FST, Token>::~LatticeBiglmFasterBucketDecoderTpl() {
  ClearActiveTokens();
  if (delete_fst_) delete fst_;
}

template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::InitDecoding() {
  // clean up from last time:
  cost_offsets_.clear();
  cutoffs_.clear();
  ClearActiveTokens();  // num_toks_ is set to 0

  warned_ = false;
  warned_noarc_ = false;
  decoding_finalized_ = false;

  final_costs_.clear();
  adaptive_beam_ = config_.beam;

  // initialize
  // Maybe move this to constructor can abvoid allocate a new map for each
  // utterance. Perhaps it can speed up a little bit.
  cur_buckets_ = new StateIdToBucketMap();
  next_buckets_ = new StateIdToBucketMap();

  StateId base_start_state = fst_->Start();
  StateId lm_start_state = lm_diff_fst_->Start();
  PairId start_state = ConstructPair(base_start_state, lm_start_state);
  Token *start_tok = new Token(0.0, std::numeric_limits<BaseFloat>::infinity(),
      base_start_state, lm_start_state, NULL, NULL, NULL);
  // Initialize the first bucket and push the start_tok into it.
  Bucket *start_bucket = new Bucket(true, base_start_state,
                                    config_.bucket_length, NULL);

  active_toks_.resize(1);
  active_toks_[0].toks = start_tok;
  active_buckets_.resize(1);
  active_buckets_[0].buckets = start_bucket;


  (*cur_buckets_)[base_start_state] = start_bucket;  // initialize current
                                                     // buckets map

  cost_offsets_.resize(1);
  cost_offsets_[0] = 0.0;
  
  cutoffs_.resize(2);
  cutoffs_[0] = std::numeric_limits<BaseFloat>::infinity();
  cutoffs_[1] = std::numeric_limits<BaseFloat>::infinity();
}

// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
template <typename FST, typename Token>
bool LatticeBiglmFasterBucketDecoderTpl<FST, Token>::Decode(
    DecodableInterface *decodable) {
  InitDecoding();

  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.
  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveBuckets(config_.lattice_beam * config_.prune_scale);
    }
    ProcessForFrame(decodable);
  }
  // A complete token list of the last frame will be generated in FinalizeDecoding()
  FinalizeDecoding();
  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}


// Outputs an FST corresponding to the single best path through the lattice.
template <typename FST, typename Token>
bool LatticeBiglmFasterBucketDecoderTpl<FST, Token>::GetBestPath(
    Lattice *olat,
    bool use_final_probs) {
  Lattice raw_lat;
  GetRawLattice(&raw_lat, use_final_probs);
  ShortestPath(raw_lat, olat);
  return (olat->NumStates() != 0);
}


// Outputs an FST corresponding to the raw, state-level lattice
template <typename FST, typename Token>
bool LatticeBiglmFasterBucketDecoderTpl<FST, Token>::GetRawLattice(
    Lattice *ofst,
    bool use_final_probs) {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;
  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  if (!decoding_finalized_ && use_final_probs) {
    // Process the non-emitting arcs for the unfinished last frame.
    ProcessNonemitting();
  }


  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  const int32 bucket_count = num_toks_/2 + 3;
  unordered_map<Token*, StateId> tok_map(bucket_count);
  // First create all states.
  std::vector<Token*> token_list;
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    TopSortTokens(active_toks_[f].toks, &token_list);
    for (size_t i = 0; i < token_list.size(); i++)
      if (token_list[i] != NULL)
        tok_map[token_list[i]] = ofst->AddState();
  }
  // The next statement sets the start state of the output FST.  Because we
  // topologically sorted the tokens, state zero must be the start-state.
  ofst->SetStart(0);

  // Now create all arcs.
  for (int32 f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      for (ForwardLinkT *l = tok->links;
           l != NULL;
           l = l->next) {
        typename unordered_map<Token*, StateId>::const_iterator
            iter = tok_map.find(l->next_elem);
        KALDI_ASSERT(iter != tok_map.end());
        StateId nextstate = iter->second;
        BaseFloat cost_offset = 0.0;
        if (l->ilabel != 0) {  // emitting..
          KALDI_ASSERT(f >= 0 && f < cost_offsets_.size());
          cost_offset = cost_offsets_[f];
        }
        Arc arc(l->ilabel, l->olabel,
                Weight(l->graph_cost, l->acoustic_cost - cost_offset),
                cur_state);
        ofst->AddArc(nextstate, arc);
      }
      if (f == num_frames) {
        if (use_final_probs && !final_costs.empty()) {
          typename unordered_map<Token*, BaseFloat>::const_iterator
              iter = final_costs.find(tok);
          if (iter != final_costs.end())
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
        } else {
          ofst->SetFinal(cur_state, LatticeWeight::One());
        }
      }
    }
  }
  
  return (ofst->NumStates() > 0);
}

// This function is now deprecated, since now we do determinization from outside
// the LatticeFasterDecoder class.  Outputs an FST corresponding to the
// lattice-determinized lattice (one path per word sequence).
template <typename FST, typename Token>
bool LatticeBiglmFasterBucketDecoderTpl<FST, Token>::GetLattice(
    CompactLattice *ofst,
    bool use_final_probs) {
  Lattice raw_fst;
  GetRawLattice(&raw_fst, use_final_probs);
  Invert(&raw_fst);  // make it so word labels are on the input.
  // (in phase where we get backward-costs).
  fst::ILabelCompare<LatticeArc> ilabel_comp;
  ArcSort(&raw_fst, ilabel_comp);  // sort on ilabel; makes
  // lattice-determinization more efficient.

  fst::DeterminizeLatticePrunedOptions lat_opts;
  lat_opts.max_mem = config_.det_opts.max_mem;

  DeterminizeLatticePruned(raw_fst, config_.lattice_beam, ofst, lat_opts);
  raw_fst.DeleteStates();  // Free memory-- raw_fst no longer needed.
  Connect(ofst);  // Remove unreachable states... there might be
  // a small number of these, in some cases.
  // Note: if something went wrong and the raw lattice was empty,
  // we should still get to this point in the code without warnings or failures.
  return (ofst->NumStates() != 0);
}


// FindOrAddBucket either locates a bucket in hash map "bucket_map", or if
// necessary inserts a new, empty bucket (i.e. with no backward links) for the
// current frame.
//
// If the destnation bucket is expanded or the olabel isn't 0, we will call
// ExpandBucket() to process the 'real' tokens recursively so that
// 'source_bucket' can be expanded.
//
// [note: it's inserted if necessary into "bucket_map", and also into the
// singly linked list of tokens active on this frame (whose head
// is at active_toks_[frame]).  The token_list_index argument is used to index
// into the active_toks_ array.
template <typename FST, typename Token>
typename LatticeBiglmFasterBucketDecoderTpl<FST, Token>::Bucket*
  LatticeBiglmFasterBucketDecoderTpl<FST, Token>::FindOrAddBucket(
    const Arc &arc, int32 token_list_index,
    BaseFloat tot_cost, BaseFloat ac_cost,
    Bucket *source_bucket, StateIdToBucketMap *bucket_map, bool *changed) {
  // Returns the Bucket pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  KALDI_ASSERT(token_list_index < active_toks_.size());
  Bucket* &buckets = active_buckets_[token_list_index].buckets;

  // Find the target bucket from map
  StateId state = arc.nextstate;
  typename StateIdToBucketMap::iterator e_found = bucket_map->find(state);

  Bucket *new_bucket = NULL;
  if (e_found == bucket_map->end()) {  // no such TokenBucket presently.
    new_bucket = new Bucket(arc.olabel == 0 ? false : true,
                            state, config_.bucket_length,
                            buckets);
    // The new_bucket doesn't have any links or costs.
    buckets = new_bucket;  // update the head of active_buckets_[index]
    
    // insert into the map
    (*bucket_map)[state] = new_bucket;
    if (changed) *changed = true;
  } else {
    Bucket *new_bucket = e_found->second;  // There is an existing
                                           // TokenBucket for this state.
    if (new_bucket->tot_cost > tot_cost) {  // replace old token
      new_bucket->tot_cost = tot_cost;
      // we don't allocate a new bucket, the old stays linked in active_toks_
      // we only replace the tot_cost
      // in the current frame, there are no forward links (and no back_cost)
      // only in ProcessForFrame we have to delete forward links
      // in case we visit a state for the second time
      // those forward links, that lead to this replaced token before:
      // they remain and will hopefully be pruned later (PruneForwardLinks...)
      if (changed) *changed = true;
    } else {
      if (changed) *changed = false;
    }
  }
  // According to the 'expanded' flag, call ExpandBucket() goes along the
  // backwardlinks of the 'source_bucket' to generate the 'real' tokens
  // recursively
  if (new_bucket->expanded) {
    // Expand the source bucket
    int32 frame = arc.ilabel == 0 ? token_list_index : token_list_index - 1;
    ExpandBucket(frame, source_bucket);

    // For here, the 'real' tokens in source bucket have been prepared.
    // build the new_bucket with the tokens from source bucket
    
    // build an unordered_map with target tokens to process token recombination.
    std::unordered_map<PairId, Token*> tmp_map;
    for (typename std::vector<Token*>::iterator iter =
         new_bucket->tokens.begin(); iter != new_bucket->tokens.end();
         iter++) {
      tmp_map[ConstructPair((*iter)->base_state, (*iter)->lm_state)] = *iter;
    }

    BaseFloat &cutoff = cutoffs_[token_list_index];
    // Use the information of source_bucket->tokens and arc to generate tokens
    for (typename std::vector<Token*>::iterator iter =
         source_bucket->tokens.begin(); iter != source_bucket->tokens.end();
         iter++) {
      StateId lm_state = (*iter)->lm_state;
      Arc arc_new(arc);
      StateId next_lm_state = PropagateLm(lm_state, &arc_new);
      PairId next_pair = ConstructPair(state, lm_state);
      
      BaseFloat tok_cur_cost = (*iter)->tot_cost;
      BaseFloat tok_graph_cost = arc.weight.Value();
      BaseFloat tok_ac_cost = arc.ilabel == 0 ? 0 : ac_cost;
      BaseFloat tok_tot_cost = tok_cur_cost + tok_graph_cost + tok_ac_cost;

      // TODO: Store adaptive beam in ProcessForFrame into a vector. Use it here
      if (tok_tot_cost > cutoff)
        continue;
      else if (tok_tot_cost + config_.beam < cutoff)
        cutoff = tok_tot_cost + config_.beam;
      // Add to target bucket
      Token *new_tok = FindOrAddToken(next_pair, token_list_index, tok_tot_cost,
                                      &tmp_map, new_bucket, *iter);
      // If the new_tok is NULL, it means we didn't insert a token into the
      // target bucket.
      if (new_tok != NULL) {  // add a forwardlink
        (*iter)->links = new ForwardLinkT(new_tok, arc.ilabel, arc.olabel,
                                          tok_graph_cost, tok_ac_cost,
                                          (*iter)->links);
      }
    }
  }
  return new_bucket;
}


// Insert a token into a bucket, and update the 'map' if it is needed.
// When the token is inserted, return it. Otherwise, return NULL.
//
// If the target token has been generated, update the tot_cost.
// If the target token is a new token, update the tot_cost and map. When the
// new token is not be inserted successfully, delete it directly.
// Note: the user should keep the 'map' and the 'bucket' is synchronous
template <typename FST, typename Token>
Token* LatticeBiglmFasterBucketDecoderTpl<FST, Token>::FindOrAddToken(
    PairId state_pair, int32 frame, BaseFloat tot_cost,
    PairIdToTokenMap *map, Bucket *bucket,
    Token *backpointer) {
  KALDI_ASSERT(frame < active_toks_.size());
  Token* &toks = active_toks_[frame].toks;
  
  Token *new_tok = NULL;
  typename PairIdToTokenMap::iterator found = map->find(state_pair);
  if (found != map->end()) {  // An existing token. But it maybe has been
                              // sifted out.
    new_tok = found->second;
    new_tok->tot_cost = tot_cost;
    bool ok = bucket->Insert(new_tok);
    if (ok) {
      new_tok->SetBackpointer(backpointer);
    } else {
      new_tok = NULL;  // the found token may has some links, so we didn't
                       // delete it here.
    }
  } else {
    new_tok = new Token(tot_cost,
                        std::numeric_limits<BaseFloat>::infinity(),
                        PairToBaseState(state_pair),
                        PairToLmState(state_pair),
                        NULL, toks, backpointer);
    bool ok = bucket->Insert(new_tok);
    if (ok) {
      // when the new token is inserted into bucket, add it to map.
      (*map)[state_pair] = new_tok;
      // update the head of the list
      toks = new_tok;
    } else {
      delete new_tok;
      new_tok = NULL;
    }
  }
  return new_tok;
}


// fill the 'real' tokens into the bucket recursively
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::ExpandBucket(
    int32 frame, Bucket* bucket) {
  if (bucket->expanded) return;

  // build an unordered_map with bucket's tokens to do token recombination.
  std::unordered_map<PairId, Token*> token_map;
  for (typename std::vector<Token*>::iterator iter = bucket->tokens.begin();
       iter != bucket->tokens.end(); iter++) {
    token_map[ConstructPair((*iter)->base_state, (*iter)->lm_state)] = *iter;
  }
  // Prepare the cutoff value
  BaseFloat &cutoff = cutoffs_[frame];

  // Prepare the source buckets
  for (BackwardBucketLinkT *link = bucket->bucket_backward_links;
       link != NULL; link = link->next) {
    Bucket *prev_bucket = link->prev_elem;
    int32 prev_frame = link->ilabel == 0 ? frame : frame - 1;
    ExpandBucket(prev_frame, prev_bucket);
  }
  // For Here all the source buckets of the 'bucket' have been prepared
  
  // BucketMerging: generate the 'real' tokens for bucket according to
  // each BackwardBucketLink.
  for (BackwardBucketLinkT *link = bucket->bucket_backward_links;
       link != NULL; link = link->next) {
    Bucket *prev_bucket = link->prev_elem;
    for (typename std::vector<Token*>::iterator iter =
         prev_bucket->tokens.begin(); iter != prev_bucket->tokens.end();
         ++iter) {
      Label ilabel = link->ilabel, olabel = link->olabel;
      // make a fake arc which is used to do PropagateLm. Only olabel and weight
      // are useful
      Label fake_ilabel = 0;
      StateId fake_state = 0;
      Arc fake_arc(fake_ilabel, olabel, link->graph_cost, fake_state);

      StateId next_base_state = bucket->base_state,
              next_lm_state = PropagateLm((*iter)->lm_state, &fake_arc);
      PairId next_pair = ConstructPair(next_base_state, next_lm_state);

      BaseFloat cur_cost = (*iter)->tot_cost,
                graph_cost = fake_arc.weight.Value(),
                ac_cost = link->acoustic_cost;
      BaseFloat tot_cost = cur_cost + graph_cost + ac_cost;
      
      // beam prune. TODO: store adaptive_beam into a vector. Use adaptive_beam 
      if (tot_cost > cutoff) continue;
      else if (tot_cost + config_.beam < cutoff)
        cutoff = tot_cost + config_.beam;

      // Add to target bucket
      Token *new_tok = FindOrAddToken(next_pair, frame, tot_cost,
                                      &token_map, bucket, *iter);
      // If the new_tok is NULL, it means we didn't insert a token into the
      // target bucket.
      if (new_tok != NULL) {  // add a forwardlink
        (*iter)->links = new ForwardLinkT(new_tok, ilabel, olabel,
                                          graph_cost, ac_cost,
                                          (*iter)->links);
      }
    }
  }
  // Set the bucket expanded flag
  bucket->expanded = true;
}


// prunes outgoing links for all tokens in active_buckets_[frame]
// it's called by PruneActiveBuckets
// all links, that have link_extra_cost > lattice_beam are pruned
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::PruneBucketForwardLinks(
    int32 frame_plus_one, bool *back_costs_changed,
    bool *links_pruned, BaseFloat delta) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  // toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *back_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_buckets_.size());
  if (active_buckets_[frame_plus_one].buckets == NULL) {  // empty list; should
                                                          // not happen.
    if (!warned_) {
      KALDI_WARN << "No buckets alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }

  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true;  // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    for (Bucket *bucket = active_buckets_[frame_plus_one].buckets;
         bucket != NULL; bucket = bucket->next) {
      ForwardBucketLinkT *link, *prev_link = NULL;
      BaseFloat bucket_back_cost = std::numeric_limits<BaseFloat>::infinity();
      for (link = bucket->bucket_forward_links; link != NULL; ) {
        // See if we need to excise this link...
        Bucket *next_bucket = link->next_elem;
        BaseFloat link_back_cost = link->acoustic_cost + link->graph_cost +
          next_bucket->back_cost;
        BaseFloat link_cost = link_back_cost + bucket->tot_cost;
        // link_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_cost == link_cost);  // check for NaN
        if (link_cost > config_.lattice_beam) {  // excise link
          ForwardBucketLinkT *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else bucket->bucket_forward_links = next_link;
          delete link;
          link = next_link;  // advance link but leave prev_link the same.
          *links_pruned = true;
        } else {   // keep the link
          if (link_cost < 0) {  // this is just a precaution.
            if (link_cost <  - 0.01)
              KALDI_WARN << "Negative alpha-beta cost " << link_cost;
            link_back_cost = - bucket->tot_cost;
          }
          if (link_back_cost < bucket_back_cost)
            bucket_back_cost = link_back_cost;
          prev_link = link;  // move to next link
          link = link->next;
        }
      }  // for all outgoing links
      if (fabs(bucket_back_cost - bucket->back_cost) > 0)
        changed = true;   // difference new minus old is bigger than delta
      bucket->back_cost = bucket_back_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }  // for all Bucket on active_buckets_[frame]
    if (changed) *back_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}


template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::PruneForwardLinks(
    int32 frame_plus_one, bool *back_costs_changed,
    bool *links_pruned, BaseFloat delta) {
  // delta is the amount by which the back_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // back_costs_changed is set to true if back_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *back_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  if (active_toks_[frame_plus_one].toks == NULL) {  // empty list; should not happen.
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }

  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true;  // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // will recompute tok_back_cost for tok.
      BaseFloat tok_back_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_back_cost is the best (min) of link_back_cost of outgoing links
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_elem;
        BaseFloat link_back_cost = link->acoustic_cost + link->graph_cost +
          next_tok->back_cost;
        BaseFloat link_cost = link_back_cost + tok->tot_cost;
        // link_cost is the difference in score between the best paths
        // through link source state and through link destination state
        KALDI_ASSERT(link_cost == link_cost);  // check for NaN
        if (link_cost > config_.lattice_beam) {  // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link;  // advance link but leave prev_link the same.
          *links_pruned = true;
        } else {   // keep the link and update the tok_cost if needed.
          if (link_cost < 0.0) {  // this is just a precaution.
            if (link_cost < -0.01)
              KALDI_WARN << "Negative alpha-beta cost: " << link_cost;
            link_back_cost = - tok->tot_cost;
          }
          if (link_back_cost < tok_back_cost)
            tok_back_cost = link_back_cost;
          prev_link = link;  // move to next link
          link = link->next;
        }
      }  // for all outgoing links
      if (fabs(tok_back_cost - tok->back_cost) > delta)
        changed = true;   // difference new minus old is bigger than delta
      tok->back_cost = tok_back_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }  // for all Token on active_toks_[frame]
    if (changed) *back_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}


// PruneForwardLinksFinal is a version of PruneForwardLinks that we call
// on the final frame.  If there are final tokens active, it uses
// the final-probs for pruning, otherwise it treats all tokens as final.
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame_plus_one = active_toks_.size() - 1;

  if (active_toks_[frame_plus_one].toks == NULL)  // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file";

  typedef typename unordered_map<Token*, BaseFloat>::const_iterator IterType;
  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost_);
  decoding_finalized_ = true;

  // Initialize the backward_costs of the tokens on the final frame.
  // We will recompute tok_backward_cost.  It has a term in it that corresponds
  // to the "final-prob", so instead of initializing tok_backward_cost to -alpha
  // below we set it to the difference between the (score+final_prob) of this
  // token and the best (score+final_prob).
  BaseFloat best_cost = std::numeric_limits<BaseFloat>::infinity();
  for (Token *tok = active_toks_[frame_plus_one].toks; tok != NULL;
       tok = tok->next) {
    BaseFloat final_cost;
    if (final_costs_.empty()) {
      final_cost = 0.0;
    } else {
      IterType iter = final_costs_.find(tok);
      if (iter != final_costs_.end())
        final_cost = iter->second;
      else
        final_cost = std::numeric_limits<BaseFloat>::infinity();
    }
    tok->back_cost = final_cost - final_best_cost_;
    best_cost = std::min(best_cost, tok->tot_cost + tok->back_cost);
  }
  // Now go through tokens on this frame, pruning backward links...  may have to
  // iterate a few times until there is no more change, because the list is not
  // in topological order.  This is a modified version of the code in
  // PruneBackwardLinks, but here we also take account of the final-probs.
  bool changed = true;
  BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLinkT *link, *prev_link = NULL;
      // tok_back_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      BaseFloat final_cost;
      if (final_costs_.empty()) {
        final_cost = 0.0;
      } else {
        IterType iter = final_costs_.find(tok);
        if (iter != final_costs_.end())
          final_cost = iter->second;
        else
          final_cost = std::numeric_limits<BaseFloat>::infinity();
      }
      BaseFloat tok_back_cost = final_cost - final_best_cost_;

      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_elem;
        BaseFloat link_back_cost = next_tok->back_cost + link->acoustic_cost +
                                   link->graph_cost;
        BaseFloat link_cost = link_back_cost + tok->tot_cost;
        if (link_cost > config_.lattice_beam) {  // excise link
          ForwardLinkT *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_cost < 0.0) { // this is just a precaution.
            if (link_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_cost;
            link_back_cost = -next_tok->tot_cost;
          }
          if (link_back_cost < tok_back_cost) {  // min
            tok_back_cost = link_back_cost;
          }
          prev_link = link;
          link = link->next;
        }
      }
      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in the non-final case because then, this case
      // showed up as having no forward links.  Here, the tok_extra_cost has
      // an extra component relating to the final-prob.
      if (tok_back_cost + tok->tot_cost > config_.lattice_beam)
        tok_back_cost = std::numeric_limits<BaseFloat>::infinity();
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok->back_cost, tok_back_cost, delta))
        changed = true;
      tok->back_cost = tok_back_cost; // will be +infinity or <= lattice_beam_.
    }
  } // while changed
}


template <typename FST, typename Token>
BaseFloat LatticeBiglmFasterBucketDecoderTpl<FST, Token>::FinalRelativeCost() const {
  if (!decoding_finalized_) {
    BaseFloat relative_cost;
    ComputeFinalCosts(NULL, &relative_cost, NULL);
    return relative_cost;
  } else {
    // we're not allowed to call that function if FinalizeDecoding() has
    // been called; return a cached value.
    return final_relative_cost_;
  }
}


// Prune away any tokens on this frame that have no forward links.
// [we don't do this in PruneBackwardLinks because it would give us
// a problem with dangling pointers].
// It's called by PruneActiveTokens if any forward links have been pruned
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::PruneTokensForFrame(
    int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  if (toks == NULL)
    KALDI_WARN << "PruneTokensForFrame: No tokens alive [doing pruning]";
  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (tok->back_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL) prev_tok->next = tok->next;
      else toks = tok->next;
      delete tok;
    } else {  // fetch next Token
      prev_tok = tok;
    }
  }
}


// Prune away any buckets on this frame that have no forward links.
// It's called by PruneActiveBuckets if any forward links have been pruned
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::PruneBucketsForFrame(
    int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_buckets_.size());
  Bucket *&buckets = active_buckets_[frame_plus_one].buckets;
  if (buckets == NULL)
    KALDI_WARN << "PruneBucketsForFrame: No buckets alive [doing pruning]";
  Bucket *bucket, *next_bucket, *prev_bucket = NULL;
  for (bucket = buckets; bucket != NULL; bucket = next_bucket) {
    next_bucket = bucket->next;
    // When we do PruneBucketForwardLinks(), we have been set the
    // bucket->back_cost to infinity.
    if (bucket->back_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_bucket != NULL) prev_bucket->next = bucket->next;
      else buckets = bucket->next;
      // Delete the backwardlinks of the bucket
      for (BackwardBucketLinkT *link = bucket->bucket_backward_links;
           link != NULL;) {
        link = link->next;
        delete link;
      }
      delete bucket;
    } else {  // fetch next Bucket
      prev_bucket = bucket;
    }
  }
}


// Go backwards through still-alive tokens, pruning them, starting not from
// the current frame (where we want to keep all tokens) but from the frame before
// that.  We go backwards through the frames and stop when we reach a point
// where the delta-costs are not changing (and the delta controls when we consider
// a cost to have "not changed").
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::PruneActiveTokens(
    BaseFloat delta) {
  int32 cur_frame_plus_one = NumFramesDecoded();
  
  // The index "f" below represents a "frame plus one", i.e. you'd have to subtract
  // one to get the corresponding index for the decodable object.
  for (int32 f = cur_frame_plus_one - 1; f >= 0; f--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next f,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[f].must_prune_forward_links) {
      bool forward_costs_changed = false, links_pruned = false;
      PruneForwardLinks(f, &forward_costs_changed, &links_pruned, delta);
      if (forward_costs_changed && f > 0) // any token has changed extra_cost
        active_toks_[f-1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[f].must_prune_tokens = true;
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    if (f+1 < cur_frame_plus_one &&      // except for last f (no forward links)
        active_toks_[f+1].must_prune_tokens) {
      PruneTokensForFrame(f+1);
      active_toks_[f+1].must_prune_tokens = false;
    }
  }
}


// It will be conduct each config_.prune_interval for save memory
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::PruneActiveBuckets(
    BaseFloat delta) {
  int32 cur_frame_plus_one = NumFramesDecoded();

  // Initalize the 'back_cost' of the buckets on current frame.
  InitBucketBeta(cur_frame_plus_one);

  // The index "f" below represents a "frame plus one", i.e. you'd have to subtract
  // one to get the corresponding index for the decodable object.
  for (int32 f = cur_frame_plus_one - 1; f >= 0; f--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next f,
    // after any of those tokens have changed their extra_cost.
    if (active_buckets_[f].must_prune_forward_links) {
      bool back_costs_changed = false, links_pruned = false;
      PruneBucketForwardLinks(f, &back_costs_changed, &links_pruned, delta);
      if (back_costs_changed && f > 0) // any token has changed extra_cost
        active_buckets_[f-1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_buckets_[f].must_prune_buckets = true;
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    if (f+1 < cur_frame_plus_one &&      // except for last f (no forward links)
        active_buckets_[f+1].must_prune_buckets) {
      PruneBucketsForFrame(f+1);
      active_buckets_[f+1].must_prune_buckets = false;
    }
  }
}


template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::ComputeFinalCosts(
    unordered_map<Token*, BaseFloat> *final_costs,
    BaseFloat *final_relative_cost,
    BaseFloat *final_best_cost) const {
  KALDI_ASSERT(!decoding_finalized_);
  if (final_costs != NULL)
    final_costs->clear();

  BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost = infinity,
      best_cost_with_final = infinity;

  // The final tokens are recorded in active_toks_[last_frame]
  for (Token *tok = active_toks_[active_toks_.size() - 1].toks; tok != NULL;
       tok = tok->next) {
    StateId state = tok->base_state,
            lm_state = tok->lm_state;
    BaseFloat final_cost = fst_->Final(state).Value() + 
                           lm_diff_fst_->Final(lm_state).Value();
    BaseFloat cost = tok->tot_cost,
        cost_with_final = cost + final_cost;
    best_cost = std::min(cost, best_cost);
    best_cost_with_final = std::min(cost_with_final, best_cost_with_final);
    if (final_costs != NULL && final_cost != infinity)
      (*final_costs)[tok] = final_cost;
  }

  if (final_relative_cost != NULL) {
    if (best_cost == infinity && best_cost_with_final == infinity) {
      // Likely this will only happen if there are no tokens surviving.
      // This seems the least bad way to handle it.
      *final_relative_cost = infinity;
    } else {
      *final_relative_cost = best_cost_with_final - best_cost;
    }
  }
  if (final_best_cost != NULL) {
    if (best_cost_with_final != infinity) { // final-state exists.
      *final_best_cost = best_cost_with_final;
    } else { // no final-state exists.
      *final_best_cost = best_cost;
    }
  }
}

template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::AdvanceDecoding(
    DecodableInterface *decodable,
    int32 max_num_frames) {
  if (std::is_same<FST, fst::Fst<fst::StdArc> >::value) {
    // if the type 'FST' is the FST base-class, then see if the FST type of fst_
    // is actually VectorFst or ConstFst.  If so, call the AdvanceDecoding()
    // function after casting *this to the more specific type.
    if (fst_->Type() == "const") {
      LatticeBiglmFasterBucketDecoderTpl<fst::ConstFst<fst::StdArc>, Token>
        *this_cast = reinterpret_cast<LatticeBiglmFasterBucketDecoderTpl<
        fst::ConstFst<fst::StdArc>, Token>* >(this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    } else if (fst_->Type() == "vector") {
      LatticeBiglmFasterBucketDecoderTpl<fst::VectorFst<fst::StdArc>, Token>
        *this_cast = reinterpret_cast<LatticeBiglmFasterBucketDecoderTpl<
        fst::VectorFst<fst::StdArc>, Token>* >(this);
      this_cast->AdvanceDecoding(decodable, max_num_frames);
      return;
    }
  }


  KALDI_ASSERT(!active_toks_.empty() && !decoding_finalized_ &&
               "You must call InitDecoding() before AdvanceDecoding");
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= NumFramesDecoded());
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     NumFramesDecoded() + max_num_frames);
  while (NumFramesDecoded() < target_frames_decoded) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveBuckets(config_.lattice_beam * config_.prune_scale);
    }
    ProcessForFrame(decodable);
  }
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::FinalizeDecoding() {
  // Process the espilon arcs for the last frame
  ProcessNonemitting();
  InitBeta(NumFramesDecoded());

  // At this point, we don't need 'Bucket' any more. We delete them for clearly.
  // TODO: maybe it can merge into another function to speedup
  CleanBucket();

  // Keep in mind, the 'real' token only have forward links.
  int32 final_frame_plus_one = NumFramesDecoded();
  // PruneForwardLinksFinal() prunes final frame (with final-probs), and
  // sets decoding_finalized_.
  PruneForwardLinksFinal();
  for (int32 f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2; // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(f, &b1, &b2, dontcare);
    PruneTokensForFrame(f + 1);
  }
  PruneTokensForFrame(0);
}


// Clean the bucket struct in FinalizeDecoding, as we will only process the
// tokens in the following steps.
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::CleanBucket() {
  KALDI_ASSERT(active_buckets_.size() > 0);

  // Clean the Bucket struct
  for (int32 frame = active_buckets_.size() - 1; frame >=0; frame--) {
    Bucket *bucket = NULL, *next_bucket =NULL;
    for(bucket = active_buckets_[frame].buckets; bucket != NULL;
        bucket = next_bucket) {
      next_bucket = bucket->next;

      // Delete forwardlinks
      ForwardBucketLinkT *flink = NULL, *next_flink = NULL;
      for(flink = bucket->bucket_forward_links; flink != NULL;
          flink = next_flink) {
        next_flink = flink->next;
        delete flink;
      }
      
      // Delete backwardlinks
      BackwardBucketLinkT *blink = NULL, *next_blink = NULL;
      for(blink = bucket->bucket_backward_links; blink != NULL;
          blink = next_blink) {
        next_blink = blink->next;
        delete blink;
      }

      // Delete the bucket
      delete bucket;
    }
  }
}


template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::ProcessForFrame(
    DecodableInterface *decodable) {
  KALDI_ASSERT(active_toks_.size() > 0);
  int32 cur_frame = active_toks_.size() - 1, // frame is the frame-index (zero-
                                             // based) used to get likelihoods
                                             // from the decodable object.
        next_frame = cur_frame + 1;
  active_toks_.resize(active_toks_.size() + 1);
  active_buckets_.resize(active_toks_.size());  // equal to active_toks_

  // swapping prev_buckets_ / cur_buckets_
  cur_buckets_->clear();
  StateIdToBucketMap *tmp_buckets = cur_buckets_;
  cur_buckets_ = next_buckets_;
  next_buckets_ = tmp_buckets;
 
  if (cur_buckets_->empty()) {
    if (!warned_) {
      KALDI_WARN << "Error, no surviving tokens/buckets on frame " << cur_frame;
      warned_ = true;
    }
  }

  cur_queue_.Clear();
  // Add buckets to queue
  for (typename StateIdToBucketMap::const_iterator iter = cur_buckets_->begin();
       iter != cur_buckets_->end(); iter++) {
    cur_queue_.Push(iter->second);
  }

  // Declare a local variable so the compiler can put it in a register, since
  // C++ assumes other threads could be modifying class members.
  BaseFloat adaptive_beam = adaptive_beam_;
  // "cur_cutoff" will be kept to the best-seen-so-far token on this frame
  // + adaptive_beam
  BaseFloat &cur_cutoff = cutoffs_[cur_frame];
  // "next_cutoff" is used to limit a new token in next frame should be handle
  // or not. It will be updated along with the further processing.
  // this will be kept updated to the best-seen-so-far token "on next frame"
  // + adaptive_beam
  BaseFloat &next_cutoff = cutoffs_[next_frame];
  // "cost_offset" contains the acoustic log-likelihoods on current frame in 
  // order to keep everything in a nice dynamic range. Reduce roundoff errors.
  BaseFloat cost_offset = cost_offsets_[cur_frame];

  // Iterator the "cur_queue_" to process non-emittion and emittion arcs in fst.
  Bucket *bucket = NULL;
  // TODO: maybe we use num_bucket_processed and num_toks_processed seperately,
  // I guess the bucket pruning has already got rid of a lot of tokens.
  // num_toks_processed will increase 1 if bucket is un-expanded or length if
  // it is expanded.
  int32 num_processed = 0;
  int32 max_active = config_.max_active;
  for (; num_processed < max_active && (bucket = cur_queue_.Pop()) != NULL;
       num_processed++) {
    BaseFloat cur_cost = bucket->tot_cost;
    StateId base_state = bucket->base_state;

    if (cur_cost > cur_cutoff &&
        num_processed > config_.min_active) { // Don't bother processing
                                                     // successors.
      break;  // This is a priority queue. The following tokens will be worse
    } else if (cur_cost + adaptive_beam < cur_cutoff) {
      cur_cutoff = cur_cost + adaptive_beam; // a tighter boundary
    }
    // If "bucket" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder).
    // Note: we don't delete the backward links, as the backward links for
    // the specific state are generated from the target state to the state.
    // We have to iterate the backwardlinks of target token to remove it, but
    // it will raise the complexity from O(n) to O(n^2). If we use
    // 'std::shared_ptr', there will be some problem when any backward link is
    // the head of 'bucket_backward_links'.
    // Fortunately, the 'bucket_backward_links' is just used to traceback to
    // expand those 'un-expanded bucket' in lazy evaluation. It will not affact
    // too much.
    // TODO: optimize
    DeleteForwardBucketLinks(bucket);  // necessary when re-visiting
    for (fst::ArcIterator<FST> aiter(*fst_, base_state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc_ref = aiter.Value();
      bool changed;

      if (arc_ref.ilabel == 0) {  // propagate nonemitting
        BaseFloat graph_cost = arc_ref.weight.Value(),
                  ac_cost = 0;
        BaseFloat tot_cost = cur_cost + graph_cost;
        if (tot_cost < cur_cutoff) {
          Bucket *new_bucket = FindOrAddBucket(arc_ref, cur_frame,
                                               tot_cost, ac_cost,
                                               bucket, cur_buckets_,
                                               &changed);

          // Add BackwardBcuketLink from new_bucket to bucket. Put it on the
          // head of new_bucket's backwardlink list
          new_bucket->bucket_backward_links =
            new BackwardBucketLinkT(bucket, 0, arc_ref.olabel,
                                    graph_cost, 0,
                                    new_bucket->bucket_backward_links);
          // Add ForwardBucketLink from the bucket to new_bucket. Put it on the
          // head of current bucket's forwardlink list.
          bucket->bucket_forward_links =
            new ForwardBucketLinkT(new_bucket, 0, arc_ref.olabel,
                                   graph_cost, 0,
                                   bucket->bucket_forward_links);
          
          // "changed" tells us whether the new token has a different
          // cost from before, or is new.
          if (changed) {
            cur_queue_.Push(new_bucket);
          }
        }
      } else {  // propagate emitting
        BaseFloat graph_cost = arc_ref.weight.Value(),
                  ac_cost = cost_offset - decodable->LogLikelihood(
                                                     cur_frame, arc_ref.ilabel),
                  tot_cost = cur_cost + ac_cost + graph_cost;
        if (tot_cost > next_cutoff) continue;
        else if (tot_cost + adaptive_beam < next_cutoff) {
          next_cutoff = tot_cost + adaptive_beam;  // a tighter boundary for
                                                   // emitting
        }
        
        // no change flag is needed
        Bucket *new_bucket = FindOrAddBucket(arc_ref, next_frame,
                                             tot_cost, ac_cost,
                                             bucket, next_buckets_, NULL);

        // Add BackwardBcuketLink from new_bucket to bucket. Put it on the
        // head of new_bucket's backwardlink list
        new_bucket->bucket_backward_links =
            new BackwardBucketLinkT(bucket, arc_ref.ilabel, arc_ref.olabel,
                                    graph_cost, ac_cost,
                                    new_bucket->bucket_backward_links);
        // Add ForwardBucketLink from the bucket to new_bucket. Put it on the
        // head of current bucket's forwardlink list.
        bucket->bucket_forward_links =
            new ForwardBucketLinkT(new_bucket, arc_ref.ilabel, arc_ref.olabel,
                                   graph_cost, ac_cost,
                                   bucket->bucket_forward_links);
      }
    }  // for all arcs
  }  // end of while loop

  // Store the offset on the acoustic likelihoods that we're applying.
  // Could just do cost_offsets_.push_back(cost_offset), but we
  // do it this way as it's more robust to future code changes.
  // Set the cost_offset_ for next frame, it equals "- best_cost_on_next_frame".
  cost_offsets_.resize(cur_frame + 2, 0.0);
  cost_offsets_[next_frame] = adaptive_beam - next_cutoff;

  {  // This block updates adaptive_beam_
    BaseFloat beam_used_this_frame = adaptive_beam;
    Bucket *bucket = cur_queue_.Pop();
    if (bucket != NULL) {
      // We hit the max-active contraint, meaning we effectively pruned to a
      // beam tighter than 'beam'. Work out what this was, it will be used to
      // update 'adaptive_beam'.
      BaseFloat best_cost_this_frame = cur_cutoff - adaptive_beam;
      beam_used_this_frame = bucket->tot_cost - best_cost_this_frame;
    }
    if (num_processed <= config_.min_active) {
      // num-toks active is dangerously low, increase the beam even if it
      // already exceeds the user-specified beam.
      adaptive_beam_ = std::max<BaseFloat>(
          config_.beam, beam_used_this_frame + 2.0 * config_.beam_delta);
    } else {
      // have adaptive_beam_ approach beam_ in intervals of config_.beam_delta
      BaseFloat diff_from_beam = beam_used_this_frame - config_.beam;
      if (std::abs(diff_from_beam) < config_.beam_delta) {
        adaptive_beam_ = config_.beam;
      } else {
        // make it close to beam_
        adaptive_beam_ = beam_used_this_frame -
          config_.beam_delta * (diff_from_beam > 0 ? 1 : -1);
      }
    }
  }
}


template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::ProcessNonemitting() {
  int32 cur_frame = active_toks_.size() - 1;
  // swapping prev_toks_ / cur_toks_
  cur_toks_->clear();
  PairIdToTokenMap *tmp_toks_ = cur_toks_;
  cur_toks_ = next_toks_;
  next_toks_ = tmp_toks_;
  
  cur_queue_.Clear();
  // Add buckets to queue
  for (typename StateIdToBucketMap::const_iterator iter = cur_buckets_->begin();
       iter != cur_buckets_->end(); iter++) {
    cur_queue_.Push(iter->second);
  }

  // Declare a local variable so the compiler can put it in a register, since
  // C++ assumes other threads could be modifying class members.
  BaseFloat adaptive_beam = adaptive_beam_;
  // "cur_cutoff" will be kept to the best-seen-so-far token on this frame
  // + adaptive_beam
  BaseFloat cur_cutoff = cutoffs_[cur_frame];

  Bucket *bucket = NULL;
  int32 num_processed = 0;
  int32 max_active = config_.max_active;

  for (; num_processed < max_active && (bucket = cur_queue_.Pop()) != NULL;
       num_processed++) {
    BaseFloat cur_cost = bucket->tot_cost;
    StateId base_state = bucket->base_state;

    if (cur_cost > cur_cutoff &&
        num_processed > config_.min_active) { // Don't bother processing
                                                   // successors.
      break;  // This is a priority queue. The following tokens will be worse
    } else if (cur_cost + adaptive_beam < cur_cutoff) {
      cur_cutoff = cur_cost + adaptive_beam; // a tighter boundary
    }
    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder)
    //
    // See ProcessForFrame to get more information about consideration.
    // TODO: Optimize
    DeleteForwardBucketLinks(bucket);  // necessary when re-visiting
    for (fst::ArcIterator<FST> aiter(*fst_, base_state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc_ref = aiter.Value();
      bool changed;

      if (arc_ref.ilabel == 0) {  // propagate nonemitting
        BaseFloat graph_cost = arc_ref.weight.Value();
        BaseFloat tot_cost = cur_cost + graph_cost;
        if (tot_cost < cur_cutoff) {
          Bucket *new_bucket = FindOrAddBucket(arc_ref, cur_frame, tot_cost, 0,
                                               bucket, cur_buckets_,
                                               &changed);

          // Add BackwardBcuketLink from new_bucket to bucket. Put it on the
          // head of new_bucket's backwardlink list
          new_bucket->bucket_backward_links =
            new BackwardBucketLinkT(bucket, 0, arc_ref.olabel,
                                    graph_cost, 0,
                                    new_bucket->bucket_backward_links);
          // Add ForwardBucketLink from the bucket to new_bucket. Put it on the
          // head of current bucket's forwardlink list.
          bucket->bucket_forward_links =
            new ForwardBucketLinkT(new_bucket, 0, arc_ref.olabel,
                                   graph_cost, 0,
                                   bucket->bucket_forward_links);
          
          // "changed" tells us whether the new token has a different
          // cost from before, or is new.
          if (changed) {
            cur_queue_.Push(new_bucket);
          }
        }
      }
    }  // end of iterate all arcs
  }  // end of for loop
  if (!decoding_finalized_) {
    // Update cost_offsets_, it equals "- best_cost".
    cost_offsets_[cur_frame] = adaptive_beam - cur_cutoff;
    // Needn't to update adaptive_beam_, since we still process this frame in
    // ProcessForFrame.
  }
}


template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::DeleteForwardBucketLinks(
    Bucket *bucket) {
  ForwardBucketLinkT *l = bucket->bucket_forward_links, *m;
  while (l != NULL) {
    m = l->next;
    delete l;
    l = m;
  }
  bucket->bucket_forward_links = NULL;
}


// static inline
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::DeleteForwardLinks(
    Token *tok) {
  ForwardLinkT *l = tok->links, *m;
  while (l != NULL) {
    m = l->next;
    delete l;
    l = m;
  }
  tok->links = NULL;
}


template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::ClearActiveTokens() {
  // a cleanup routine, at utt end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    // Delete all tokens alive on this frame, and any forward
    // links they may have.
    for (Token *tok = active_toks_[i].toks; tok != NULL; ) {
      DeleteForwardLinks(tok);
      Token *next_tok = tok->next;
      delete tok;
      tok = next_tok;
    }
  }
  active_toks_.clear();
  KALDI_ASSERT(num_toks_ == 0);
}

// static
template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::TopSortTokens(
    Token *tok_list, std::vector<Token*> *topsorted_list) {
  unordered_map<Token*, int32> token2pos;
  typedef typename unordered_map<Token*, int32>::iterator IterType;
  int32 num_toks = 0;
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    num_toks++;
  int32 cur_pos = 0;
  // We assign the tokens numbers num_toks - 1, ... , 2, 1, 0.
  // This is likely to be in closer to topological order than
  // if we had given them ascending order, because of the way
  // new tokens are put at the front of the list.
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    token2pos[tok] = num_toks - ++cur_pos;

  unordered_set<Token*> reprocess;

  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter) {
    Token *tok = iter->first;
    int32 pos = iter->second;
    for (ForwardLinkT *link = tok->links; link != NULL; link = link->next) {
      if (link->ilabel == 0) {
        // We only need to consider epsilon links, since non-epsilon links
        // transition between frames and this function only needs to sort a list
        // of tokens from a single frame.
        IterType following_iter = token2pos.find(link->next_elem);
        if (following_iter != token2pos.end()) { // another token on this frame,
                                                 // so must consider it.
          int32 next_pos = following_iter->second;
          if (next_pos < pos) { // reassign the position of the Token.
            following_iter->second = cur_pos++;
            reprocess.insert(link->next_elem);
          }
        }
      }
    }
    // In case we had previously assigned this token to be reprocessed, we can
    // erase it from that set because it's "happy now" (we just processed it).
    reprocess.erase(tok);
  }

  size_t max_loop = 1000000, loop_count; // max_loop is to detect epsilon cycles.
  for (loop_count = 0;
       !reprocess.empty() && loop_count < max_loop; ++loop_count) {
    std::vector<Token*> reprocess_vec;
    for (typename unordered_set<Token*>::iterator iter = reprocess.begin();
         iter != reprocess.end(); ++iter)
      reprocess_vec.push_back(*iter);
    reprocess.clear();
    for (typename std::vector<Token*>::iterator iter = reprocess_vec.begin();
         iter != reprocess_vec.end(); ++iter) {
      Token *tok = *iter;
      int32 pos = token2pos[tok];
      // Repeat the processing we did above (for comments, see above).
      for (ForwardLinkT *link = tok->links; link != NULL; link = link->next) {
        if (link->ilabel == 0) {
          IterType following_iter = token2pos.find(link->next_elem);
          if (following_iter != token2pos.end()) {
            int32 next_pos = following_iter->second;
            if (next_pos < pos) {
              following_iter->second = cur_pos++;
              reprocess.insert(link->next_elem);
            }
          }
        }
      }
    }
  }
  KALDI_ASSERT(loop_count < max_loop && "Epsilon loops exist in your decoding "
               "graph (this is not allowed!)");

  topsorted_list->clear();
  topsorted_list->resize(cur_pos, NULL);  // create a list with NULLs in between.
  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter)
    (*topsorted_list)[iter->second] = iter->first;
}


template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::InitBeta(
    int32 frame, BaseFloat scale) {
  for (Token* tok = active_toks_[frame].toks; tok != NULL; tok = tok->next) {
    tok->back_cost = -tok->tot_cost * scale;
  }
}

template <typename FST, typename Token>
void LatticeBiglmFasterBucketDecoderTpl<FST, Token>::InitBucketBeta(
    int32 frame, BaseFloat scale) {
  for (Bucket* bucket = active_buckets_[frame].buckets; 
       bucket != NULL; bucket = bucket->next) {
    bucket->back_cost = -bucket->tot_cost * scale;
  }
}


// Instantiate the template for the combination of token types and FST types
// that we'll need.

template class LatticeBiglmFasterBucketDecoderTpl<fst::Fst<fst::StdArc>,
         biglmbucketdecoder::StdToken<fst::Fst<fst::StdArc> > >;
template class LatticeBiglmFasterBucketDecoderTpl<fst::VectorFst<fst::StdArc>,
         biglmbucketdecoder::StdToken<fst::VectorFst<fst::StdArc> > >;
template class LatticeBiglmFasterBucketDecoderTpl<fst::ConstFst<fst::StdArc>,
         biglmbucketdecoder::StdToken<fst::ConstFst<fst::StdArc> > >;
template class LatticeBiglmFasterBucketDecoderTpl<fst::GrammarFst,
         biglmbucketdecoder::StdToken<fst::GrammarFst> >;

template class LatticeBiglmFasterBucketDecoderTpl<fst::Fst<fst::StdArc> ,
         biglmbucketdecoder::BackpointerToken<fst::Fst<fst::StdArc> > >;
template class LatticeBiglmFasterBucketDecoderTpl<fst::VectorFst<fst::StdArc>,
         biglmbucketdecoder::BackpointerToken<fst::VectorFst<fst::StdArc> > >;
template class LatticeBiglmFasterBucketDecoderTpl<fst::ConstFst<fst::StdArc>,
         biglmbucketdecoder::BackpointerToken<fst::ConstFst<fst::StdArc> > >;
template class LatticeBiglmFasterBucketDecoderTpl<fst::GrammarFst,
         biglmbucketdecoder::BackpointerToken<fst::GrammarFst> >;

} // end namespace kaldi.
