#ifndef KALDI_IOT_LANGUAGE_MODEL_H
#define KALDI_IOT_LANGUAGE_MODEL_H

#include "common.h"

#include <fst/fstlib.h>
#include <fst/fst-decl.h>

namespace kaldi {
namespace iot {

// virtual interface for generic language model in FST representation
template<class Arc>
class LmFst {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;

  virtual StateId Start() = 0;

  virtual Weight Final(StateId s) = 0;

  virtual bool GetArc(StateId s, Label ilabel, Arc *oarc) = 0;

  virtual ~LmFst() { }
};


template<class Arc>
class NgramLmFst: public LmFst<Arc> {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;

  explicit NgramLmFst(const fst::Fst<Arc> *fst) : fst_(fst) {
#ifdef KALDI_PARANOID
    KALDI_ASSERT(fst_.Properties(kILabelSorted|kIDeterministic, true) ==
                (kILabelSorted|kIDeterministic) &&
                "Input FST is not i-label sorted and deterministic.");
#endif
  }

  virtual ~NgramLmFst() { }

  StateId Start() { return fst_->Start(); }

  bool GetArc(StateId s, Label ilabel, Arc *oarc) {
    KALDI_ASSERT(ilabel != 0); //  We don't allow GetArc for epsilon.

    fst::SortedMatcher<fst::Fst<Arc> > sm(*fst_, fst::MATCH_INPUT, 1);
    sm.SetState(s);
    if (sm.Find(ilabel)) {
      const Arc &arc = sm.Value();
      *oarc = arc;
      return true;
    } else {
      Weight backoff_w;
      StateId backoff_state = GetBackoffState(s, &backoff_w);
      if (backoff_state == fst::kNoStateId) return false;
      if (!this->GetArc(backoff_state, ilabel, oarc)) return false;
      oarc->weight = Times(oarc->weight, backoff_w);
      return true;
    }
  }

  Weight Final(StateId state) {
    Weight w = fst_->Final(state);
    if (w != Weight::Zero()) return w;
    Weight backoff_w;
    StateId backoff_state = GetBackoffState(state, &backoff_w);
    if (backoff_state == fst::kNoStateId) return Weight::Zero();
    else return Times(backoff_w, this->Final(backoff_state));
  }

 private:
  inline StateId GetBackoffState(StateId s, Weight *w) {
    fst::ArcIterator<fst::Fst<Arc> > aiter(*fst_, s);
    if (aiter.Done()) // no arcs.
      return fst::kNoStateId;
    const Arc &arc = aiter.Value();
    if (arc.ilabel == 0) {
      *w = arc.weight;
      return arc.nextstate;
    } else {
      return fst::kNoStateId;
    }
  }

  const fst::Fst<Arc> *fst_;
};


template<class Arc>
class ScaleCacheLmFst: public LmFst<Arc> {
 public:
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::Label Label;

  /// We don't take ownership of this pointer.  The argument is "really" const.
  ScaleCacheLmFst(LmFst<Arc> *fst, float scale, 
                  StateId num_cached_arcs = 100000)
   : fst_(fst),
     scale_(scale),
     num_cached_arcs_(num_cached_arcs),
     cached_arcs_(num_cached_arcs)
  {
    KALDI_ASSERT(num_cached_arcs > 0);
    for (StateId i = 0; i < num_cached_arcs; i++)
      cached_arcs_[i].first = fst::kNoStateId; // Invalidate all elements of the cache.
  }

  virtual StateId Start() { return fst_->Start(); }

  /// We don't bother caching the final-probs, just the arcs.
  virtual Weight Final(StateId s) {
    // Note: Weight is indirectly a typedef to TropicalWeight.
    Weight final = fst_->Final(s);
    if (final == Weight::Zero()) {
      return Weight::Zero();
    } else {
      return fst::TropicalWeight(final.Value() * scale_);
    }
  }

  virtual bool GetArc(StateId s, Label ilabel, Arc *oarc) {
    // Note: we don't cache anything in case a requested arc does not exist.
    // In the uses that we imagine this will be put to, essentially all the
    // requested arcs will exist.  This only affects efficiency.
    KALDI_ASSERT(s >= 0 && ilabel != 0);
    size_t index = this->GetIndex(s, ilabel);
    if (cached_arcs_[index].first == s &&
        cached_arcs_[index].second.ilabel == ilabel) {
      *oarc = cached_arcs_[index].second;
      oarc->weight = fst::TropicalWeight(oarc->weight.Value() * scale_);
      return true;
    } else {
      Arc arc;
      if (fst_->GetArc(s, ilabel, &arc)) {
        cached_arcs_[index].first = s;
        cached_arcs_[index].second = arc;
        *oarc = arc;
        oarc->weight = fst::TropicalWeight(oarc->weight.Value() * scale_);
        return true;
      } else {
        return false;
      }
    }
  }

 private:
  // Get index for cached arc.
  inline size_t GetIndex(StateId src_state, Label ilabel) {
    const StateId p1 = 26597, p2 = 50329; // these are two
    // values that I drew at random from a table of primes.
    // note: num_cached_arcs_ > 0.

    // We cast to size_t before the modulus, to ensure the
    // result is positive.
    return static_cast<size_t>(src_state * p1 + ilabel * p2) %
        static_cast<size_t>(num_cached_arcs_);
  }

  LmFst<Arc> *fst_;
  float scale_;
  StateId num_cached_arcs_;
  std::vector<std::pair<StateId, Arc> > cached_arcs_;
};

} // namespace iot
} // namespace kaldi

#endif