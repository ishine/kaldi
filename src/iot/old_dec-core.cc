#include "iot/dec-core.h"

namespace kaldi {
namespace iot {


DecCore::DecCore(Wfst *wfst,
                 fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst,
                 const DecCoreOptions &opts) :
    fst_(wfst),
    lm_diff_fst_(lm_diff_fst),
    config_(opts),
    num_frames_decoded_(-1)
{
  KALDI_ASSERT(config_.hash_ratio >= 1.0);
  KALDI_ASSERT(config_.max_active > 1);
  KALDI_ASSERT(config_.min_active >= 0 && config_.min_active < config_.max_active);
  cur_toks_.SetSize(1000);
  token_pool_ = new MemoryPool(sizeof(Token), config_.token_pool_realloc);
}


DecCore::~DecCore() {
  ClearToks(cur_toks_.Clear());
  DELETE(token_pool_);
}


void DecCore::InitDecoding() {
  ClearToks(cur_toks_.Clear());

  PairId init_pair = ConstructPair(fst_->Start(), (lm_diff_fst_ == NULL) ? 0 : lm_diff_fst_->Start());
  WfstArc init_arc(kWfstEpsilon, kWfstEpsilon, kWfstDefaultWeight, fst_->Start());
  cur_toks_.Insert(init_pair, NewToken(NULL, init_arc, 0.0f));

  ProcessNonemitting(std::numeric_limits<float>::max());
  num_frames_decoded_ = 0;
}


void DecCore::Decode(DecodableInterface *decodable) {
  InitDecoding();
  while (!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    double weight_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(weight_cutoff);
  }
}


void DecCore::AdvanceDecoding(DecodableInterface *decodable, int32 max_num_frames) {
  KALDI_ASSERT(num_frames_decoded_ >= 0 && "You must call InitDecoding() before AdvanceDecoding()");

  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= num_frames_decoded_);
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     num_frames_decoded_ + max_num_frames);
  while (num_frames_decoded_ < target_frames_decoded) {
    double weight_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(weight_cutoff); 
  }    
}


void DecCore::FinalizeDecoding() {
  ;
}


/*
bool DecCore::GetBestPath(fst::MutableFst<LatticeArc> *fst_out,
                                bool use_final_probs) {
  // GetBestPath gets the decoding output.  If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into
  // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
  // nothing was available.  It returns true if it got output (thus, fst_out
  // will be nonempty).
  fst_out->DeleteStates();
  Token *best_tok = NULL;
  bool is_final = ReachedFinal();
  if (!is_final) {
    for (const Elem *e = cur_toks_.GetList(); e != NULL; e = e->tail)
      if (best_tok == NULL || *best_tok < *(e->val) )
        best_tok = e->val;
  } else {
    double infinity =  std::numeric_limits<double>::infinity(),
        best_cost = infinity;
    for (const Elem *e = cur_toks_.GetList(); e != NULL; e = e->tail) {
      double this_cost = e->val->cost_ + fst_.Final(e->key).Value();
      if (this_cost < best_cost && this_cost != infinity) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }
  if (best_tok == NULL) return false;  // No output.

  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.

  for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
    BaseFloat tot_cost = tok->cost_ -
        (tok->prev_ ? tok->prev_->cost_ : 0.0),
        graph_cost = tok->arc_.weight.Value(),
        ac_cost = tot_cost - graph_cost;
    LatticeArc l_arc(tok->arc_.ilabel,
                     tok->arc_.olabel,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->arc_.nextstate);
    arcs_reverse.push_back(l_arc);
  }
  KALDI_ASSERT(arcs_reverse.back().nextstate == fst_.Start());
  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final && use_final_probs) {
    Weight final_weight = fst_.Final(best_tok->arc_.nextstate);
    fst_out->SetFinal(cur_state, LatticeWeight(final_weight.Value(), 0.0));
  } else {
    fst_out->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(fst_out);
  return true;
}
*/


double DecCore::GetCutoff(Elem *list_head, size_t *tok_count, BaseFloat *adaptive_beam, Elem **best_elem) {
  double best_cost = std::numeric_limits<double>::infinity();
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      double w = e->val->cost_;
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
    return best_cost + config_.beam;
  } else {
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      double w = e->val->cost_;
      tmp_array_.push_back(w);
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    double beam_cutoff = best_cost + config_.beam,
        min_active_cutoff = std::numeric_limits<double>::infinity(),
        max_active_cutoff = std::numeric_limits<double>::infinity();
    
    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_cost + config_.beam_delta;
      return max_active_cutoff;
    }    
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0) min_active_cutoff = best_cost;
      else {
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active) ?
                         tmp_array_.begin() + config_.max_active :
                         tmp_array_.end());
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam)
        *adaptive_beam = min_active_cutoff - best_cost + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}


void DecCore::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                      * config_.hash_ratio);
  if (new_sz > cur_toks_.Size()) {
    cur_toks_.SetSize(new_sz);
  }
}


WfstStateId DecCore::PropagateLm(WfstStateId lm_state, WfstArc *arc) { // returns new LM state.
  if (arc->olabel == kWfstEpsilon) {
    return lm_state; // no change in LM state if no word crossed.
  } else { // Propagate in the LM-diff FST.
    Arc lm_arc;
    bool ans = lm_diff_fst_->GetArc(lm_state, arc->olabel, &lm_arc);
    if (!ans) { // this case is unexpected for statistical LMs.
      KALDI_LOG << "No arc available in LM (unlikely to be correct "
                   "if a statistical language model);";
      exit(0);
    } else {
      arc->weight = arc->weight + lm_arc.weight.Value();
      arc->olabel = lm_arc.olabel; // probably will be the same.
      return lm_arc.nextstate; // return the new LM state.
    }
  }
}


// ProcessEmitting returns the likelihood cutoff used.
double DecCore::ProcessEmitting(DecodableInterface *decodable) {
  int32 frame = num_frames_decoded_;
  Elem *prev_toks = cur_toks_.Clear();
  size_t tok_cnt;
  BaseFloat adaptive_beam;
  Elem *best_elem = NULL;
  double weight_cutoff = GetCutoff(prev_toks, &tok_cnt, &adaptive_beam, &best_elem);
  KALDI_VLOG(3) << tok_cnt << " tokens active.";
  PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.
    
  // This is the cutoff we use after adding in the log-likes (i.e.
  // for the next frame).  This is a bound on the cutoff we will use
  // on the next frame.
  double next_weight_cutoff = std::numeric_limits<double>::infinity();
  
  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.
  if (best_elem) {
    PairId state_pair = best_elem->key;
    WfstStateId state    = PairToState(state_pair);
    WfstStateId lm_state = PairToLmState(state_pair);

    Token *tok = best_elem->val;

    const WfstState *s = fst_->State(state);
    const WfstArc *a = fst_->Arc(s->arc_base);

    for (int32 j = 0; j < s->num_arcs; j++, a++) {
      WfstArc arc = *a;
      if (arc.ilabel != kWfstEpsilon) {
        if (lm_diff_fst_ != NULL) {
          PropagateLm(lm_state, &arc);
        }
        BaseFloat ac_cost = - decodable->LogLikelihood(frame, arc.ilabel);
        double new_weight = tok->cost_ + arc.weight + ac_cost;
        if (new_weight + adaptive_beam < next_weight_cutoff)
          next_weight_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  for (Elem *e = prev_toks, *e_tail; e != NULL; e = e_tail) {
    PairId state_pair = e->key;
    WfstStateId state    = PairToState(state_pair);
    WfstStateId lm_state = PairToLmState(state_pair);

    Token *tok = e->val;
    
    if (tok->cost_ < weight_cutoff) {  // not pruned.
      const WfstState *s = fst_->State(state);
      const WfstArc *a = fst_->Arc(s->arc_base);
      for (int32 j = 0; j < s->num_arcs; j++, a++) {
        if (a->ilabel != kWfstEpsilon) {  // emitting
          WfstArc arc = *a;
          WfstStateId next_lm_state = (lm_diff_fst_ == NULL) ? 0 : PropagateLm(lm_state, &arc);
          BaseFloat ac_cost =  - decodable->LogLikelihood(frame, arc.ilabel);
          double new_weight = tok->cost_ + arc.weight + ac_cost;
          if (new_weight < next_weight_cutoff) {  // not pruned..
            if (new_weight + adaptive_beam < next_weight_cutoff) { // update cutoff
              next_weight_cutoff = new_weight + adaptive_beam;
            }

            PairId next_pair = ConstructPair(arc.dst, next_lm_state);
            Token *new_tok = NewToken(tok, arc, ac_cost);

            Elem *e_found = cur_toks_.Find(next_pair);
            if (e_found == NULL) {
              cur_toks_.Insert(next_pair, new_tok);
            } else {
              if ( *(e_found->val) < *new_tok ) {
                DelToken(e_found->val);
                e_found->val = new_tok;
              } else {
                DelToken(new_tok);
              }
            }
          }
        }
      }
    }
    e_tail = e->tail;
    DelToken(e->val);
    cur_toks_.Delete(e);
  }
  num_frames_decoded_++;
  return next_weight_cutoff;
}


void DecCore::ProcessNonemitting(double cutoff) {
  // Processes nonemitting arcs for one frame. 
  KALDI_ASSERT(queue_.empty());
  for (const Elem *e = cur_toks_.GetList(); e != NULL;  e = e->tail) {
    queue_.push_back(e->key);
  }
  while (!queue_.empty()) {
    PairId state_pair = queue_.back();
    queue_.pop_back();
    Token *tok = cur_toks_.Find(state_pair)->val;
    if (tok->cost_ > cutoff) { // Don't bother processing successors.
      continue;
    }

    WfstStateId state    = PairToState(state_pair);
    WfstStateId lm_state = PairToLmState(state_pair);

    const WfstState *s = fst_->State(state);
    const WfstArc *a = fst_->Arc(s->arc_base);

    for (int32 j = 0; j < s->num_arcs; j++, a++) {
      if (a->ilabel == kWfstEpsilon) {  // non emitting
        WfstArc arc = *a;
        WfstStateId next_lm_state = (lm_diff_fst_ == NULL) ? 0 : PropagateLm(lm_state, &arc);
        PairId next_pair = ConstructPair(arc.dst, next_lm_state);
        Token *new_tok = NewToken(tok, arc, 0.0f);
        if (new_tok->cost_ > cutoff) {  // prune
          DelToken(new_tok);
        } else {
          Elem *e_found = cur_toks_.Find(next_pair);
          if (e_found == NULL) {
            cur_toks_.Insert(next_pair, new_tok);
            queue_.push_back(next_pair);
          } else {
            if ( *(e_found->val) < *new_tok ) {
              DelToken(e_found->val);
              e_found->val = new_tok;
              queue_.push_back(next_pair);
            } else {
              DelToken(new_tok);
            }
          }
        }
      }
    }
  }
}


void DecCore::ClearToks(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    DelToken(e->val);
    e_tail = e->tail;
    cur_toks_.Delete(e);
  }
}

bool DecCore::GetBestTransitionPath() {
  Token *best_tok = NULL;
  double infinity =  std::numeric_limits<double>::infinity();
  double best_cost = infinity;

  for (const Elem *e = cur_toks_.GetList(); e != NULL; e = e->tail) {
    double this_cost = e->val->cost_ + fst_->State(e->key)->weight;
    if (this_cost < best_cost && this_cost != infinity) {
      best_cost = this_cost;
      best_tok  = e->val;
    }
  }

  if (best_tok == NULL) {
    return false;
  }

  for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
    if (tok->arc_.olabel != kWfstEpsilon) {
      KALDI_LOG << fst_->OutputTable()->Index2Symbol(tok->arc_.olabel); 
    }
  }

  return true;
}


} // end namesace iot
} // end namesace kaldi.
