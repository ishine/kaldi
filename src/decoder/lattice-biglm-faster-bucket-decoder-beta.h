// decoder/lattice-biglm-faster-bucket-decoder.h

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

#ifndef KALDI_DECODER_LATTICE_BIGLM_FASTER_BUCKET_DECODER_H_
#define KALDI_DECODER_LATTICE_BIGLM_FASTER_BUCKET_DECODER_H_


#include "util/stl-utils.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "decoder/grammar-fst.h"
#include "decoder/lattice-faster-decoder.h"
#include <algorithm>

namespace kaldi {

struct LatticeBiglmFasterBucketDecoderConfig {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat lattice_beam;
  int32 prune_interval;
  bool determinize_lattice; // not inspected by this class... used in
                            // command-line program.
  BaseFloat beam_delta; // tune the adaptive_beam_ for each frame
  BaseFloat cost_scale;
  BaseFloat prune_scale;
  // Note: we don't make this configurable on the command line,
  // it's not a very important parameter.  It affects the
  // algorithm that prunes the tokens as we go.
  // Most of the options inside det_opts are not actually queried by the
  // LatticeFasterDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeFaster.
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  BaseFloat proportion;
  int32 beta_interval;
  int32 bucket_length; // the capacity of each bucket

  LatticeBiglmFasterBucketDecoderConfig():
    beam(16.0),
    max_active(std::numeric_limits<int32>::max()),
    min_active(200),
    lattice_beam(10.0),
    prune_interval(25),
    determinize_lattice(true),
    beam_delta(0.5),
    cost_scale(1.0),
    prune_scale(0.1),
    proportion(0.5),
    beta_interval(5),
    bucket_length(5) { }

  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more "
                   "accurate.");
    opts->Register("max-active", &max_active, "Decoder max active states. "
                   "Larger->slower; more accurate");
    opts->Register("min-active", &min_active, "Decoder minimum #active states.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam. "
                   "Larger->slower, and deeper lattices");
    opts->Register("prune-interval", &prune_interval, "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register("determinize-lattice", &determinize_lattice, "If true, "
                   "determinize the lattice (lattice-determinization, keeping "
                   "only best pdf-sequence for each word-sequence).");
    opts->Register("beam-delta", &beam_delta, "Increment used in decoding-- this "
                   "parameter is obscure and relates to a speedup in the way the "
                   "max-active constraint is applied.  Larger is more accurate.");
    opts->Register("cost-scale", &cost_scale, "A scale that we multiply the "
                   "token costs by before intergerizing; a larger value means "
                   "more buckets and precise.");
    opts->Register("proportion", &proportion, "Use it to select a value between "
                   "beam and lattice-beam. The value will be used to decide a "
                   "token expand or not when we trace-back to expand bucket."
                   "1->beam, 0->lattice-beam");
    opts->Register("beta-interval", &beta_interval, "Interval (in frames) over "
                   "which to update the beta value of the bucket. If the value "
                   "is less than zero, the beta of the bucket will always be "
                   "up-to-date.");
    opts->Register("bucket-length", &bucket_length, "The capacity of each "
                   "bucket.");
  }
  void Check() const {
    KALDI_ASSERT(beam > 0.0 && max_active > 1 && lattice_beam > 0.0
                 && min_active <= max_active
                 && prune_interval > 0 && beam_delta > 0.0
                 && prune_scale > 0.0 && prune_scale < 1.0
                 && proportion >= 0.0 && proportion <= 1.0
                 && bucket_length >= 1);
  }
};


namespace biglmbucketdecoder {
// We will template the decoder on the token type as well as the FST type; this
// is a mechanism so that we can use the same underlying decoder code for
// versions of the decoder that support quickly getting the best path
// (LatticeFasterOnlineDecoder, see lattice-faster-online-decoder.h) and also
// those that do not (LatticeFasterDecoder).


// BackwardLinks are the links from an element to another element on the
// preceding frame or sometimes on the current frame (for input-epsilon links).
// Note: We call it 'Element' rather than 'Token' since it will be used for
// both 'TokenBucket' or 'Token'.
template <typename Element>
struct BackwardLink {
  using Label = fst::StdArc::Label;

  Element *prev_elem;  // the previous element
  Label ilabel;  // ilabel on arc
  Label olabel;  // olabel on arc
  BaseFloat graph_cost;  // graph cost of traversing arc (contains LM, etc.)
                         // For 'TokenBucket', it stores the graphcost of
                         // base HCLG graph.
                         // For 'Token', it stores the traversing graph cost.
  BaseFloat acoustic_cost;  // acoustic cost (pre-scaled) of traversing arc
  BackwardLink *next;  // next in singly-linked list of backward arcs (arcs
                       // in the state-level lattice) from an element.
  inline BackwardLink(Element *prev_elem, Label ilabel, Label olabel,
                      BaseFloat graph_cost, BaseFloat acoustic_cost,
                      BackwardLink *next):
      prev_elem(prev_elem), ilabel(ilabel), olabel(olabel),
      graph_cost(graph_cost), acoustic_cost(acoustic_cost),
      next(next) { } 
};


// ForwardLinks are the links from an element to another element on the next
// frame or sometimes on the current frame (for input-epsilon links).
// Note: We call it 'Element' rather than 'Token' since it will be used for
// both 'TokenBucket' or 'Token'.
template <typename Element>
struct ForwardLink {
  using Label = fst::StdArc::Label;

  Element *next_elem;  // the next element
  Label ilabel;  // ilabel on arc
  Label olabel;  // olabel on arc
  BaseFloat graph_cost;  // graph cost of traversing arc (contains LM, etc.)
                         // See 'BackwardLink' for more information.
  BaseFloat acoustic_cost;  // acoustic cost (pre-scaled) of traversing arc
  ForwardLink *next;  // next in singly-linked list of forward arcs (arcs
                      // in the state-level lattice) from an element.
  inline ForwardLink(Element *next_elem, Label ilabel, Label olabel,
                     BaseFloat graph_cost, BaseFloat acoustic_cost,
                     ForwardLink *next):
      next_elem(next_elem), ilabel(ilabel), olabel(olabel),
      graph_cost(graph_cost), acoustic_cost(acoustic_cost),
      next(next) { }
  
  void Print() {
    std::cout << ilabel << " : " << olabel << " /(" << graph_cost << ","
              << acoustic_cost << ").";
  }
};


// For 'Token' (either 'StdToken' or 'BackpointerToken'), we use 'Forwardlink'
// to connect one token to another. It will be easy for us to compute the
// back_cost of each token and do pruning.
template <typename Fst>
struct StdToken {
  using ForwardLinkT = ForwardLink<StdToken>;
  using Token = StdToken;
  using StateId = typename Fst::Arc::StateId;

  // Standard token type for LatticeFasterDecoder.  Each active HCLG
  // (decoding-graph) state on each frame has one token.
  
  // In the decoder, we use alpha-beta pruning to replace the extra_cost.
  // Actually, the performance of both of them are same.

  // tot_cost is the total (LM + acoustic) cost from the beginning of the
  // utterance up to this point.  (but see cost_offset_, which is subtracted
  // to keep it in a good numerical range).
  BaseFloat tot_cost;

  // tot_cost + back_cost >=0. When calling PruneTokenForwardLinks, it will be
  // updated. (tot_cost + back_cost) can describe the minimum difference
  // between the best path including this state and the best overall path,
  // under the assumption that any of the currently active states at the
  // decoding front may eventually succeed.
  // When these quantities are unchanged for a particular frame, the pruning
  // algorithm can stop going backward.
  BaseFloat back_cost;

  // Record the state id of the token
  StateId base_state;  // the state in base graph (the HCLG)
  StateId lm_state;  // the state in LM-diff FST

  // 'links' is the head of singly-linked list of ForwardLinks, which is what
  // we use for lattice generation.
  ForwardLinkT *links;

  //'next' is the next in the singly-linked list of tokens for this frame.
  Token *next;

  // identitfy the token is the top N tokens in a TokenBucket or not
  bool in_heap;

  // This function does nothing and should be optimized out; it's needed
  // so we can share the regular LatticeFasterDecoderTpl code and the code
  // for LatticeFasterOnlineDecoder that supports fast traceback.
  inline void SetBackpointer (Token *backpointer) { }

  // This constructor just ignores the 'backpointer' argument.  That argument is
  // needed so that we can use the same decoder code for LatticeFasterDecoderTpl
  // and LatticeFasterOnlineDecoderTpl (which needs backpointers to support a
  // fast way to obtain the best path).
  inline StdToken(BaseFloat tot_cost, BaseFloat back_cost,
                  StateId base_state, StateId lm_state,
                  ForwardLinkT *links, Token *next, Token *backpointer):
    tot_cost(tot_cost), back_cost(back_cost),
    base_state(base_state), lm_state(lm_state),
    links(links), next(next), in_heap(false) { }
  
  // The smaller, the better
  inline bool operator < (const Token &other) const {
    if ((tot_cost + back_cost) ==
        (other.tot_cost + other.back_cost)) {
      return lm_state < other.lm_state;
    } else {
      return (tot_cost + back_cost) <
             (other.tot_cost + other.back_cost);
    }
  }
  inline bool operator > (const Token &other) const { return other < (*this); }

  inline void PrintInfo() {
    std::cout << "The (" << base_state << "," << lm_state << ") Token cost "
              << "is " << tot_cost << " + " << back_cost << " = "
              << tot_cost + back_cost << " and the heap status is "
              << (in_heap ? " true" : " false") << std::endl;
  }
};


template <typename Fst>
struct BackpointerToken {
  using ForwardLinkT = ForwardLink<BackpointerToken>;
  using Token = BackpointerToken;
  using StateId = typename Fst::Arc::StateId;

  // BackpointerToken is like Token but also
  // Standard token type for LatticeFasterDecoder.  Each active HCLG
  // (decoding-graph) state on each frame has one token.

  // tot_cost is the total (LM + acoustic) cost from the beginning of the
  // utterance up to this point.  (but see cost_offset_, which is subtracted
  // to keep it in a good numerical range).
  BaseFloat tot_cost;

  // tot_cost + back_cost >=0. When calling PruneBackwardLinks, it will be
  // updated. (tot_cost + back_cost) can describe the minimum difference
  // between the best path including this state and the best overall path,
  // under the assumption that any of the currently active states at the
  // decoding front may eventually succeed.
  // When these quantities are unchanged for a particular frame, the pruning
  // algorithm can stop going backward.
  BaseFloat back_cost;

  // Record the state id of the token
  StateId base_state;  // the state in base graph (the HCLG)
  StateId lm_state;  // the state in LM-diff FST

  // 'links' is the head of singly-linked list of BackwardLinks, which is what
  // we use for lattice generation.
  ForwardLinkT *links;

  //'next' is the next in the singly-linked list of tokens for this frame.
  Token *next;

  // identitfy the token is the top N tokens in a TokenBucket or not
  bool in_heap;

  // Best preceding BackpointerToken (could be a on this frame, connected to
  // this via an epsilon transition, or on a previous frame).  This is only
  // required for an efficient GetBestPath function in
  // LatticeFasterOnlineDecoderTpl; it plays no part in the lattice generation
  // (the "links" list is what stores the forward links, for that).
  Token *backpointer;


  inline void SetBackpointer (Token *backpointer) {
    this->backpointer = backpointer;
  }

  inline BackpointerToken(BaseFloat tot_cost, BaseFloat back_cost,
                          StateId base_state, StateId lm_state,
                          ForwardLinkT *links, Token *next, Token *backpointer):
                          tot_cost(tot_cost), back_cost(back_cost),
                          base_state(base_state), lm_state(lm_state),
                          links(links), next(next), 
                          in_heap(false),
                          backpointer(backpointer) { }

  // The smaller, the better
  inline bool operator < (const Token &other) const {
    if ((tot_cost + back_cost) ==
        (other.tot_cost + other.back_cost)) {
      return lm_state < other.lm_state;
    } else {
      return (tot_cost + back_cost) <
             (other.tot_cost + other.back_cost);
    }
  }
  inline bool operator > (const Token &other) const { return other < (*this); }

  inline void PrintInfo() {
    std::cout << "The (" << base_state << "," << lm_state << ") Token cost "
              << "is " << tot_cost << " + " << back_cost << " = "
              << tot_cost + back_cost << " and the heap status is "
              << (in_heap ? " true" : " false") << std::endl;

  }
};


// This comparison function will be used in forward passing. So we compare the
// tot_cost only. It will be used to generate a minimum heap.
template <typename Token>
struct cmp {
  bool operator() (Token *a, Token *b) {
    return a->tot_cost < b->tot_cost;
  }
};

template <typename FST, typename Token>
struct TokenBucket {
  using BackwardBucketLinkT = BackwardLink<TokenBucket>;
  using ForwardBucketLinkT = ForwardLink<TokenBucket>;
  using StateId = typename FST::Arc::StateId;

  bool expanded;  // indicate the bucket is expanded or non-expanded

  BackwardBucketLinkT *bucket_backward_links;  // link current bucket to
                                              // preceding buckets
  ForwardBucketLinkT *bucket_forward_links;  // link current bucket to
                                            // next buckets
  BaseFloat tot_cost;  // the best alpha of the bucket. compatible with
                       // BucketQueue code.
  BaseFloat back_cost;  // the beta of the bucket. It will be used when
                        // prunning
  StateId base_state;  // the tokens which are collected in a bucket will have
                       // the same "base_state" and different "lm_state"s
  const int32 length;  // only keep top N best tokens for a bucket. When the
                       // size of "tokens" beyonds "length", remove the worst
                       // one from the priority_queue
  TokenBucket *next;  // 'next' is the next in the singly-linked list of tokens
                      // for this frame.
  bool in_queue;  // identitfy the 'TokenBucket' is in current queue or not
                  // to prevent duplication in function ProcessForFrame().
  int32 update_time;  // records the specific time-step on which the back_cost
                      // is updated. It helps us to decide we update or not.
 
  // We store the top N 'real' tokens into a vector and orginize it as a
  // maximum heap. When we insert a token into the heap, we compare the
  // "tot_cost" which is the smaller the better. So this is a max-heap.
  std::vector<Token*> top_toks;
  // when we insert an element into max-heap, it will call the compare
  // function automatically to compare the new element with the elements from
  // the leaf to root in order. If the compare function return false, the new
  // element will go on. Othewise, it will stop.
  
  // Store all the 'real' tokens into the vector
  std::vector<Token*> all_toks;


  // Bear in mind, the type of StateId is unsigned int.
  inline TokenBucket(bool expanded, StateId base_state, size_t bucket_size,
                     TokenBucket *next):
    expanded(expanded), bucket_backward_links(NULL), bucket_forward_links(NULL),
    tot_cost(std::numeric_limits<BaseFloat>::infinity()),
    back_cost(std::numeric_limits<BaseFloat>::infinity()),
    base_state(base_state), length(bucket_size), next(next),
    in_queue(false), update_time(-1) {
    
    top_toks.resize(0);
    top_toks.reserve(length + 1);  // reserve "length" + 1 positions to
                                     // prevent memory re-allocate.
    // The top_toks is a max-heap which is used to manage the top N tokens
    std::make_heap(top_toks.begin(), top_toks.end(), cmp<Token>());
    all_toks.resize(0);
  }

  // Insert a token into "top_toks" max-heap. When the size of "top_toks"
  // beyonds "length", remove the worst one.
  void Insert(Token *tok, bool fresh = true) {
    if (top_toks.empty()) {
      tot_cost = tok->tot_cost;  // The first one
    }
    if (!tok->in_heap) {  // the token is not in max-heap
                          // (it maybe in all_toks or it is a new token)
                          // check if we need to insert it into the heap
      if (top_toks.size() < length) {  // the heap is not full
                                       // insert it directly
        top_toks.push_back(tok);
        std::push_heap(top_toks.begin(), top_toks.end(), cmp<Token>());
        tok->in_heap = true;

        if (tok->tot_cost < tot_cost) {
          tot_cost = tok->tot_cost;
        }
      } else {  // the heap is full. Compare with the worst one.
        Token* worst = top_toks.front();

        // If the token is worse than current worst token. Don't insert.
        if (tok->tot_cost > worst->tot_cost) {
          // the token is an invalid token
          tok->in_heap = false;
        } else {
          // Check if we need to update the best tot_cost of the bucket
          if (tok->tot_cost < tot_cost) {
            tot_cost = tok->tot_cost;
          }

          // Pop the worst token.
          worst->in_heap = false;
          // Note: pop_heap function only put the worst element to the end of
          // the container and then make the rest elements to a heap.
          std::pop_heap(top_toks.begin(), top_toks.end(), cmp<Token>());
          top_toks.pop_back();

          // Insert the 'real' token
          tok->in_heap = true;
          top_toks.push_back(tok);
          std::push_heap(top_toks.begin(), top_toks.end(), cmp<Token>());
        }
      }
    } else {  // the token is in heap now.
              // If this part is reached, the token must achieve a better
              // alpha value. Note: it is controlled by the outer logic
      std::make_heap(top_toks.begin(), top_toks.end(), cmp<Token>());
      // Check if we need to update the bucket cost as a better alpha value is
      // achieved.
      if (tok->tot_cost < tot_cost) {
        tot_cost = tok->tot_cost;
      }
    }

    // if this is a new token, put it into the 'all_toks'
    if (fresh) all_toks.push_back(tok); 
  }

 
  void BucketInfo(TokenBucket *bucket,
                  std::set<TokenBucket*> *preceding_buckets) {
     std::cout << "The (" << bucket->base_state << ") bucket's status is "
              << (bucket->expanded ? "expanded." : "non-expanded.")
              << " The best alpha is " << bucket->tot_cost << ", it has "
              << bucket->all_toks.size() << " real tokens, and it has "
              << preceding_buckets->size() << " preceding buckets :";
     for (typename std::set<TokenBucket*>::iterator it =
          preceding_buckets->begin(); it != preceding_buckets->end(); it++) {
       std::cout << " " << (*it)->base_state;
     }
     std::cout << "." << std::endl;
  }

  void Info(TokenBucket *bucket) {
    std::cout << "The (" << bucket->base_state << ") bucket's status is "
              << (bucket->expanded ? "expanded." : "non-expanded.")
              << " The best alpha is " << bucket->tot_cost << ", "
              << " The beta is " << bucket->back_cost << " and time is "
              << bucket->update_time
              << " , it has "
              << bucket->all_toks.size() << " real tokens." << std::endl;
    // Print the 'real' tokens of the cur_bucket.
    if (bucket->expanded && bucket->all_toks.size() != 0) {
      for(typename std::vector<Token*>::iterator it_token =
          bucket->all_toks.begin(); it_token != bucket->all_toks.end();
          it_token++) {
        (*it_token)->PrintInfo();
      }
    }
  }
  
  /*
  void Info(TokenBucket* bucket) {
    // Use two queue. One is used to process the buckets on this depth, the
    // other is used to store the sub-level buckets which will be process in
    // the next turn.
    std::set<TokenBucket* > cur_set;
    std::set<TokenBucket* > next_set;

    cur_set.insert(bucket);
    while (!cur_set.empty()) {
      for (typename std::set<TokenBucket*>::iterator it = cur_set.begin();
           it != cur_set.end(); it++) {
        TokenBucket* cur_bucket = (*it);
        // Get the preceding buckets of the 'cur_bucket'
        std::set<TokenBucket* > tmp_set;
        BackwardBucketLinkT *bl = cur_bucket->bucket_backward_links;
        while (bl != NULL) {
          TokenBucket *preceding_bucket = bl->prev_elem;
          tmp_set.insert(preceding_bucket);
          bl = bl->next;
        }

        // Print the information of the 'cur_bucket'
        BucketInfo(cur_bucket, &tmp_set);
      
        // Print the 'real' tokens of the cur_bucket.
        if (cur_bucket->expanded && cur_bucket->all_toks.size() != 0) {
          for(typename std::vector<Token*>::iterator it_token =
              cur_bucket->all_toks.begin();
              it_token != cur_bucket->all_toks.end(); it_token++) {
            (*it_token)->PrintInfo();
          }
        }

        // Union the next_set and tmp_set
        for (typename std::set<TokenBucket*>::iterator it_set = tmp_set.begin();
             it_set != tmp_set.end(); it_set++) {
          next_set.insert(*it_set);
        }
      }
      cur_set.clear();
      cur_set.swap(next_set);
    }
  }
  */

  void PrintInfo() {
    /*
    std::cout << "The (" << base_state << ") bucket's status is "
              << (expanded ? "expanded." : "non-expanded.")
              << " The best alpha is " << tot_cost << " and It has "
              << all_toks.size() << " real tokens." << std::endl;
    if (expanded && all_toks.size() != 0) {
      for(typename std::vector<Token*>::iterator it = all_toks.begin();
          it != all_toks.end(); it++) {
        (*it)->PrintInfo();
      }
    }
    */
    std::cout << "-----print bucket info ------" << std::endl;
    Info(this);
    //for (BackwardBucketLinkT *bl = bucket_backward_links; bl != NULL;
    //     bl = bl->next) {
    //  
    //  std::cout << bl->prev_elem->base_state 
    //            << " (" << bl->prev_elem->tot_cost << ") <--- "
    //            << bl->ilabel << " : " << bl->olabel << " / ("
    //            << bl->graph_cost << "," << bl->acoustic_cost << ")"
    //            << std::endl;
      /*
      if (bl->prev_elem->expanded && bl->prev_elem->all_toks.size() != 0) {
        for(typename std::vector<Token*>::iterator it =
            bl->prev_elem->all_toks.begin();
            it != bl->prev_elem->all_toks.end(); it++) {
          (*it)->PrintInfo();
        }
      }
      */
    //  Info(bl->prev_elem);
    //  std::cout << "----" << std::endl;
    //}
    std::cout << "----------finish a bucket -------" << std::endl;
  }

  void PrintForwardLinks() {
    ForwardBucketLinkT *tmp = bucket_forward_links;
    while (tmp != NULL) {
      tmp->Print();
      std::cout << " --> " << tmp->next_elem->base_state << std::endl;
      tmp = tmp->next;
    }
  }
};
}  // namespace biglmbucketdecoder


template<typename Element>
class BucketQueue {
 public:
  // Constructor. 'cost_scale' is a scale that we multiply the costs by
  // before intergerizing; a larger value means more buckets.
  // 'bucket_offset_' is initialized to "15 * cost_scale_". It is an empirical
  // value in case we trigger the re-allocation in normal case, since we do in
  // fact normalize costs to be not far from zero on each frame. 
  BucketQueue(BaseFloat cost_scale = 1.0);

  // Adds element to the queue; sets the field tok->in_queue to true (it is not
  // an error if it was already true).
  // If an element was already in the queue but its cost improves, you should
  // just Push it again. It will be added to (possibly) a different bucket, but
  // the old entry will remain. We use "element->in_queue" to decide
  // an entry is nonexistent or not. When pop an element off, the field
  // 'element->in_queue' is set to false. So the old entry in the queue will be
  // considered as nonexistent when we try to pop it.
  void Push(Element *elem);

  // Removes and returns the next element 'elem' in the queue, or NULL if there
  // were no elements left. Sets element->in_queue to false for the returned
  // Element.
  Element* Pop();

  // Clears all the individual buckets. Sets 'first_nonempty_bucket_index_' to
  // the end of buckets_.
  void Clear();

 private:
  // Configuration value that is multiplied by elements' costs before integerizing
  // them to determine the bucket index
  BaseFloat cost_scale_;

  // buckets_ is a list of Elements 'elem' for each bucket.
  // If elem->in_queue is false, then the item is considered as not
  // existing (this is to avoid having to explicitly remove Tokens when their
  // costs change). The index into buckets_ is determined as follows:
  // bucket_index = std::floor(elem->cost * cost_scale_);
  // vec_index = bucket_index - bucket_storage_begin_;
  // then access buckets_[vec_index].
  std::vector<std::vector<Element*> > buckets_;

  // An offset that determines how we index into the buckets_ vector;
  // In the constructor this will be initialized to something like
  // "15 * cost_scale_" which will make it unlikely that we have to change this
  // value in future if we get a much better Token (this is expensive because it
  // involves reallocating 'buckets_').
  int32 bucket_offset_;

  // first_nonempty_bucket_index_ is an integer in the range [0,
  // buckets_.size() - 1] which is not larger than the index of the first
  // nonempty element of buckets_.
  int32 first_nonempty_bucket_index_;

  // Synchronizes with first_nonempty_bucket_index_.
  std::vector<Element*> *first_nonempty_bucket_;

  // If the size of the BucketQueue is larger than "bucket_size_tolerance_", we
  // will resize it to "bucket_size_tolerance_" in Clear. A weird long
  // BucketQueue might be caused when the min-active was activated and an
  // unusually large loglikelihood range was encountered.
  size_t bucket_size_tolerance_;
};

/** This is the "normal" lattice-generating decoder.
    See \ref lattices_generation \ref decoders_faster and \ref decoders_simple
     for more information.

   The decoder is templated on the FST type and the token type.  The token type
   will normally be StdToken, but also may be BackpointerToken which is to support
   quick lookup of the current best path (see lattice-faster-online-decoder.h)

   The FST you invoke this decoder with is expected to equal
   Fst::Fst<fst::StdArc>, a.k.a. StdFst, or GrammarFst.  If you invoke it with
   FST == StdFst and it notices that the actual FST type is
   fst::VectorFst<fst::StdArc> or fst::ConstFst<fst::StdArc>, the decoder object
   will internally cast itself to one that is templated on those more specific
   types; this is an optimization for speed.
 */
template <typename FST, typename Token = biglmbucketdecoder::StdToken<FST> >
class LatticeBiglmFasterBucketDecoderTpl {
 public:
  using Arc = typename FST::Arc;
  using Label = typename Arc::Label;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  using PairId = uint64;  // (StateId in fst) + (StateId in lm_diff_fst) << 32

  using Bucket =
    typename kaldi::biglmbucketdecoder::TokenBucket<FST, Token>;
  using BucketQueue = typename kaldi::BucketQueue<Bucket>;

  using BackwardLinkT =
    typename kaldi::biglmbucketdecoder::BackwardLink<Token>;
  using BackwardBucketLinkT =
    typename kaldi::biglmbucketdecoder::BackwardLink<Bucket>;

  using ForwardLinkT =
    typename kaldi::biglmbucketdecoder::ForwardLink<Token>;
  using ForwardBucketLinkT =
    typename kaldi::biglmbucketdecoder::ForwardLink<Bucket>;

  using StateIdToBucketMap = typename std::unordered_map<StateId, Bucket*>;
  //using StateIdToTokenMap = typename std::unordered_map<StateId, TokenBucket*,
  //      std::hash<StateId>, std::equal_to<StateId>,
  //      fst::PoolAllocator<std::pair<const StateId, TokenBucket*> > >;

  using PairIdToTokenMap = typename std::unordered_map<PairId, Token*>;
  //using PairIdToTokenMap = typename std::unordered_map<PairId, Token*,
  //      std::hash<PairId>, std::equal_to<PairId>,
  //      fst::PoolAllocator<std::pair<const PairId, Token*> > >;

  // Instantiate this class once for each thing you have to decode.
  // This version of the constructor does not take ownership of
  // 'fst'.
  LatticeBiglmFasterBucketDecoderTpl(const FST &fst,
      const LatticeBiglmFasterBucketDecoderConfig &config,
      fst::DeterministicOnDemandFst<Arc> *lm_diff_fst);

  // This version of the constructor takes ownership of the fst, and will delete
  // it when this object is destroyed.
  LatticeBiglmFasterBucketDecoderTpl(
      const LatticeBiglmFasterBucketDecoderConfig &config, FST *fst,
      fst::DeterministicOnDemandFst<Arc> *lm_diff_fst);

  void SetOptions(const LatticeBiglmFasterBucketDecoderConfig &config) {
    config_ = config;
  }

  const LatticeBiglmFasterBucketDecoderConfig &GetOptions() const {
    return config_;
  }

  ~LatticeBiglmFasterBucketDecoderTpl();

  /// Decodes until there are no more frames left in the "decodable" object..
  /// note, this may block waiting for input if the "decodable" object blocks.
  /// Returns true if any kind of traceback is available (not necessarily from a
  /// final state).
  bool Decode(DecodableInterface *decodable);


  /// says whether a final-state was active on the last frame.  If it was not, the
  /// lattice (or traceback) will end with states that are not final-states.
  bool ReachedFinal() const {
    return FinalRelativeCost() != std::numeric_limits<BaseFloat>::infinity();
  }

  /// Outputs an FST corresponding to the single best path through the lattice.
  /// Returns true if result is nonempty (using the return status is deprecated,
  /// it will become void).  If "use_final_probs" is true AND we reached the
  /// final-state of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.  Note: this just calls GetRawLattice()
  /// and figures out the shortest path.
  bool GetBestPath(Lattice *ofst,
                   bool use_final_probs = true);

  /// Outputs an FST corresponding to the raw, state-level
  /// tracebacks.  Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state
  /// of the graph then it will include those as final-probs, else
  /// it will treat all final-probs as one.
  /// The raw lattice will be topologically sorted.
  /// The function can be called during decoding, it will process non-emitting
  /// arcs from "next_toks_" map to get tokens from both non-emitting and 
  /// emitting arcs for getting raw lattice.
  ///
  /// See also GetRawLatticePruned in lattice-faster-online-decoder.h,
  /// which also supports a pruning beam, in case for some reason
  /// you want it pruned tighter than the regular lattice beam.
  /// We could put that here in future needed.
  bool GetRawLattice(Lattice *ofst, bool use_final_probs = true);



  /// [Deprecated, users should now use GetRawLattice and determinize it
  /// themselves, e.g. using DeterminizeLatticePhonePrunedWrapper].
  /// Outputs an FST corresponding to the lattice-determinized
  /// lattice (one path per word sequence).   Returns true if result is nonempty.
  /// If "use_final_probs" is true AND we reached the final-state of the graph
  /// then it will include those as final-probs, else it will treat all
  /// final-probs as one.
  bool GetLattice(CompactLattice *ofst,
                  bool use_final_probs = true);

  /// InitDecoding initializes the decoding, and should only be used if you
  /// intend to call AdvanceDecoding().  If you call Decode(), you don't need to
  /// call this.  You can also call InitDecoding if you have already decoded an
  /// utterance and want to start with a new utterance.
  void InitDecoding();

  /// This will decode until there are no more frames ready in the decodable
  /// object.  You can keep calling it each time more frames become available.
  /// If max_num_frames is specified, it specifies the maximum number of frames
  /// the function will decode before returning.
  void AdvanceDecoding(DecodableInterface *decodable,
                       int32 max_num_frames = -1);

  /// This function may be optionally called after AdvanceDecoding(), when you
  /// do not plan to decode any further.  It does an extra pruning step that
  /// will help to prune the lattices output by GetLattice and (particularly)
  /// GetRawLattice more accurately, particularly toward the end of the
  /// utterance.  It does this by using the final-probs in pruning (if any
  /// final-state survived); it also does a final pruning step that visits all
  /// states (the pruning that is done during decoding may fail to prune states
  /// that are within kPruningScale = 0.1 outside of the beam).  If you call
  /// this, you cannot call AdvanceDecoding again (it will fail), and you
  /// cannot call GetLattice() and related functions with use_final_probs =
  /// false.
  /// Used to be called PruneActiveTokensFinal().
  void FinalizeDecoding();

  /// FinalRelativeCost() serves the same purpose as ReachedFinal(), but gives
  /// more information.  It returns the difference between the best (final-cost
  /// plus cost) of any token on the final frame, and the best cost of any token
  /// on the final frame.  If it is infinity it means no final-states were
  /// present on the final frame.  It will usually be nonnegative.  If it not
  /// too positive (e.g. < 5 is my first guess, but this is not tested) you can
  /// take it as a good indication that we reached the final-state with
  /// reasonable likelihood.
  BaseFloat FinalRelativeCost() const;

  // This function is used to clean the structure about bucket. It heralds that
  // we will process the tokens only in the following.
  // It will be called by FinalizeDecoding().
  void CleanBucket();


  // Returns the number of frames decoded so far.  The value returned changes
  // whenever we call ProcessForFrame().
  inline int32 NumFramesDecoded() const { return active_toks_.size() - 1; }

 protected:
  // we make things protected instead of private, as code in
  // LatticeFasterOnlineDecoderTpl, which inherits from this, also uses the
  // internals.

  // head of per-frame list of Tokens (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;
    TokenList(): toks(NULL), must_prune_forward_links(true),
                 must_prune_tokens(true) { }
  };
  // head of per-frame list of TokenBuckets (list is in topological order),
  // and something saying whether we ever pruned it using PruneForwardLinks.
  struct BucketList {
    Bucket *buckets;
    bool must_prune_forward_links;
    bool must_prune_buckets;
    BucketList(): buckets(NULL), must_prune_forward_links(true),
                  must_prune_buckets(true) { }
  };


  // Deletes the elements of the singly linked list of
  // bucket->bucket_forward_links and the corresponding elements in 'destnation
  // bucket' bucket_backward_links.
  // This function will be called when a state is re-visited on a particular
  // frame. This case will be happend when we process non-emiting arcs.
  // Note: find the corresponding backward link from the singly linked list of
  // destination bucket is painful. Need to be optimize.
  inline static void DeleteBucketLinks(Bucket *bucket);

  // Deletes the elements of the singly linked list tok->links
  inline static void DeleteForwardLinks(Token *tok);

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
  // is at active_buckets_[frame]).  The token_list_index argument is used to
  // index into the active_buckets_ array.
  //
  // Returns the Bucket pointer.  Sets "changed" (if non-NULL) to true if the
  // token was newly created or the cost changed.
  // If Token == StdToken, the 'backpointer' will be set to source_bucket with
  // no purpose (and will hopefully be optimized out).
  inline Bucket* FindOrAddBucket(const Arc &arc, int32 token_list_index,
                                 BaseFloat tot_cost, BaseFloat ac_cost,
                                 BaseFloat graph_cost,
                                 Bucket *source_bucket,
                                 StateIdToBucketMap *bucket_map,
                                 bool *changed);

  // Insert a token into a bucket, and update the 'map' if it is needed.
  // When the token is inserted, return it. Otherwise, return NULL.
  //
  // If the target token has been generated, update the tot_cost.
  // If the target token is a new token, update the tot_cost and map. When the
  // new token is not be inserted successfully, delete it directly.
  // Note: the user should keep the 'map' and the 'bucket' is synchronous
  inline Token* FindOrAddToken(PairId state_pair, int32 frame,
                               BaseFloat tot_cost,
                               PairIdToTokenMap *map, Bucket *bucket,
                               Token *backpointer);

  // Fill the 'real' tokens into the bucket according to the 'Backwardlinks'
  // of the bucket recursively.
  // The frame index is used to find the specific token list
  void ExpandBucket(int32 frame, Bucket* bucket);


  // prunes outgoing links for all buckets in active_buckets_[frame]
  // it's called by PruneActiveBuckets
  // all links, that have link_tot_cost + link_back_cost > lattice_beam are
  // pruned.
  // delta is the amount by which the back_costs must change before we set
  // *back_costs_changed = ture.
  // If delta is larger, we'll tend to go back less far 'back_costs_changed'
  // is set to true if back_cost was changed for any bucket
  // link_pruned is set to true if any link in any bucket was pruned.
  void PruneBucketForwardLinks(int32 frame_plus_one,
                               bool *back_costs_changed,
                               bool *links_pruned, BaseFloat delta);

  // prunes outgoing links for all tokens in active_toks_[frame]
  // it's called by PruneActiveTokens
  // all links, that have link_extra_cost > lattice_beam are pruned
  // delta is the amount by which the extra_costs must change
  // before we set *extra_costs_changed = true.
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned
  void PruneForwardLinks(int32 frame_plus_one, bool *back_costs_changed,
                         bool *links_pruned,
                         BaseFloat delta);

  // This function computes the final-costs for tokens active on the final
  // frame.  It outputs to final-costs, if non-NULL, a map from the Token*
  // pointer to the final-prob of the corresponding state, for all Tokens
  // that correspond to states that have final-probs.  This map will be
  // empty if there were no final-probs.  It outputs to
  // final_relative_cost, if non-NULL, the difference between the best
  // forward-cost including the final-prob cost, and the best forward-cost
  // without including the final-prob cost (this will usually be positive), or
  // infinity if there were no final-probs.  [c.f. FinalRelativeCost(), which
  // outputs this quanitity].  It outputs to final_best_cost, if
  // non-NULL, the lowest for any token t active on the final frame, of
  // forward-cost[t] + final-cost[t], where final-cost[t] is the final-cost in
  // the graph of the state corresponding to token t, or the best of
  // forward-cost[t] if there were no final-probs active on the final frame.
  // You cannot call this after FinalizeDecoding() has been called; in that
  // case you should get the answer from class-member variables.
  void ComputeFinalCosts(unordered_map<Token*, BaseFloat> *final_costs,
                         BaseFloat *final_relative_cost,
                         BaseFloat *final_best_cost) const;

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses
  // the final-probs for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal();

  // Prune away any tokens on this frame that have no forward links.
  // [we don't do this in PruneForwardLinks because it would give us
  // a problem with dangling pointers].
  // It's called by PruneActiveTokens if any forward links have been pruned
  void PruneTokensForFrame(int32 frame_plus_one);

  // Prune away any buckets on this frame that have no forward links.
  // Bear in mind, different from the PruneTokensForFrame, when we delete the
  // bucket, we have to remove the 'backwardlinks' on the bucket.
  void PruneBucketsForFrame(int32 frame_plus_one);

  // Go backwards through still-alive tokens, pruning them if the
  // forward+backward cost is more than lat_beam away from the best path.  It's
  // possible to prove that this is "correct" in the sense that we won't lose
  // anything outside of lat_beam, regardless of what happens in the future.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.  larger delta -> will recurse
  // less far.
  void PruneActiveTokens(BaseFloat delta);

  // Go backwards through still-alive tokens, pruning them if the
  // forward+backward cost is more than lat_beam away from the best path.
  // It will be conduct each config_.prune_interval for save memory.
  void PruneActiveBuckets(BaseFloat delat);

  /// Processes non-emitting (epsilon) arcs and emitting arcs for one frame
  /// together. It takes the emittion tokens in "cur_toks_" from last frame.
  /// Generates non-emitting tokens for previous frame and emitting tokens for
  /// next frame.
  /// Notice: The emitting tokens for the current frame means the token take
  /// acoustic scores of the current frame. (i.e. the destnations of emitting
  /// arcs.)
  void ProcessForFrame(DecodableInterface *decodable);

  /// Processes nonemitting (epsilon) arcs for one frame.
  /// This function is called from FinalizeDecoding(), and also from
  /// GetRawLattice() if GetRawLattice() is called before FinalizeDecoding() is
  /// called.
  void ProcessNonemitting();

  void ClearActiveTokens();
  // This function takes a singly linked list of tokens for a single frame, and
  // outputs a list of them in topological order (it will crash if no such order
  // can be found, which will typically be due to decoding graphs with epsilon
  // cycles, which are not allowed).  Note: the output list may contain NULLs,
  // which the caller should pass over; it just happens to be more efficient for
  // the algorithm to output a list that contains NULLs.
  static void TopSortTokens(Token *tok_list,
                            std::vector<Token*> *topsorted_list);

  // In PruneActiveBuckets, we call this function to set up the 'back_cost' of
  // each bucket on current frame. Then we can go backward to prune the useless
  // buckets.
  // The 'scale' is use to scale the 'back_cost'. Most of time, it is in the
  // range of (0, 1].
  void InitBucketBeta(int32 frame, BaseFloat scale = 1.0);
  void InitBeta(int32 frame, BaseFloat scale = 1.0);

  // We initialize the beta of a bucket when it is first used rather than it
  // is created. After that, if the time stamp of a bucket is too far behind
  // the current frame when we use it again, we will update it.
  void InitOrUpdateBucketBeta(Bucket* bucket, bool restrict = false);


  // Update the graph cost according to lm_state and olabel
  // Return new LM State
  inline StateId PropagateLm(StateId lm_state, Arc *arc) {
    if (arc->olabel == 0) {
      return lm_state;
    } else {
      Arc lm_arc;
      bool ans = lm_diff_fst_->GetArc(lm_state, arc->olabel, &lm_arc);
      /*
      if (debug_) {
        std::cout << "The lm state is " << lm_state;
        std::cout << ". The lm arc is " << lm_arc.ilabel << " : "
                  << lm_arc.olabel << " with cost " << lm_arc.weight
                  << " and the destination is " << lm_arc.nextstate
                  << std::endl;
      }
      */
      if (!ans) {  // this case is unexpected for statistical LMs
        if (!warned_noarc_) {
          warned_noarc_ = true;
          KALDI_WARN << "No arc available in LM (unlikely to be correct "
            "if a statistical language model); Will not warn again";
        }
        arc->weight = Weight::Zero();
        return lm_state;  // doesn't really matter what we return here; will be
                          // pruned.
      } else {
        arc->weight = Times(arc->weight, lm_arc.weight);
        arc->olabel = lm_arc.olabel;  // probably will be the same.
        return lm_arc.nextstate;  // return the new LM state.
      }
    }
  }

  inline PairId ConstructPair(StateId base_state, StateId lm_state) {
    return static_cast<PairId>(base_state) + 
      (static_cast<PairId>(lm_state) << 32);
  }

  static inline StateId PairToBaseState(PairId state) {
    return static_cast<StateId>(static_cast<uint32>(state));
  }

  static inline StateId PairToLmState(PairId state) {
    return static_cast<StateId>(static_cast<uint32>(state >> 32));
  }

  KALDI_DISALLOW_COPY_AND_ASSIGN(LatticeBiglmFasterBucketDecoderTpl);

  /// Gets the weight cutoff.
  /// Notice: In traiditional version, the histogram prunning method is applied
  /// on a complete token list on one frame. But, in this version, it is used
  /// on a token list which only contains the emittion part. So the max_active
  /// and min_active values might be narrowed.

  // Map (base_state, lm_state) to token
  PairIdToTokenMap *cur_toks_, *next_toks_;
  StateIdToBucketMap *cur_buckets_, *next_buckets_;

  std::vector<TokenList> active_toks_; // Lists of tokens, indexed by
  // frame (members of TokenList are toks, must_prune_forward_links,
  // must_prune_tokens).
  std::vector<BucketList> active_buckets_;

  // fst_ is a pointer to the FST we are decoding from.
  const FST *fst_;
  // delete_fst_ is true if the pointer fst_ needs to be deleted when this
  // object is destroyed.
  bool delete_fst_;
  fst::DeterministicOnDemandFst<Arc> *lm_diff_fst_;
  LatticeBiglmFasterBucketDecoderConfig config_;

  std::vector<BaseFloat> cost_offsets_;  // This contains, for each
  // frame, an offset that was added to the acoustic log-likelihoods on that
  // frame in order to keep everything in a nice dynamic range i.e.  close to
  // zero, to reduce roundoff errors.
  // Notice: It will only be added to emitting arcs (i.e. cost_offsets_[t] is
  // added to arcs from "frame t" to "frame t+1").


  int32 num_toks_;  // current total #toks allocated...
  bool warned_;
  bool warned_noarc_;  // Use in PropagateLm to indicate the unusual phenomenon.
                       // Prevent duplicate warnings.

  /// decoding_finalized_ is true if someone called FinalizeDecoding().  [note,
  /// calling this is optional].  If true, it's forbidden to decode more.  Also,
  /// if this is set, then the output of ComputeFinalCosts() is in the next
  /// three variables.  The reason we need to do this is that after
  /// FinalizeDecoding() calls PruneTokensForFrame() for the final frame, some
  /// of the tokens on the last frame are freed, so we free the list from toks_
  /// to avoid having dangling pointers hanging around.
  bool decoding_finalized_;

  /// For the meaning of the next 3 variables, see the comment for
  /// decoding_finalized_ above., and ComputeFinalCosts().
  unordered_map<Token*, BaseFloat> final_costs_;
  BaseFloat final_relative_cost_;
  BaseFloat final_best_cost_;

  BaseFloat adaptive_beam_;  // will be set to beam_ when we start
  BucketQueue cur_queue_;  // temp variable used in 
                           // ProcessForFrame/ProcessNonemitting

  BaseFloat tb_thresh_;  // A value between beam and lattice_beam
  //bool debug_ = false;
  // Record the total number of operations which processes an arc entering
  // a bucket
  int32 stats_bucket_operation_forward = 0;
  int32 stats_bucket_operation_backward = 0;
  int32 stats_bucket_operation_final_forward = 0;
  int32 stats_bucket_operation_final_backward = 0;
  // Record the total number of operations on expanded bucket which creates
  // an arc between tokens
  int32 stats_token_operation = 0;
  int32 stats_token_operation_final = 0;
  // sign we are in the final process
  bool final_ = false;
};

typedef LatticeBiglmFasterBucketDecoderTpl<fst::StdFst,
        biglmbucketdecoder::StdToken<fst::StdFst> >
        LatticeBiglmFasterBucketDecoder;

} // end namespace kaldi.

#endif
