// Rework of Kaldi's BigLm-Faster-Decoder
// Author: Jiayu DU 2016

#ifndef KALDI_IOT_DEC_CORE_H_
#define KALDI_IOT_DEC_CORE_H_

#include "itf/options-itf.h"
#include "itf/decodable-itf.h"
#include "fst/fstlib.h"
#include "fstext/deterministic-fst.h"

#include "iot/memory-pool.h"
#include "iot/hash-list.h"
#include "iot/wfst.h"

namespace kaldi {
namespace iot {

struct DecCoreOptions {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat beam_delta;
  BaseFloat hash_ratio;
  int32 token_pool_realloc;

  DecCoreOptions() : 
    beam(16.0),
    max_active(std::numeric_limits<int32>::max()),
    min_active(20),
    beam_delta(0.5),
    hash_ratio(2.0),
    token_pool_realloc(2048)
  { }

  void Register(OptionsItf *opts, bool full) {  /// if "full", use obscure
    /// options too.
    /// Depends on program.
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active,
                   "Decoder min active states (don't prune if #active less than this).");
    if (full) {
      opts->Register("beam-delta", &beam_delta,
                     "Increment used in decoder [obscure setting]");
      opts->Register("hash-ratio", &hash_ratio,
                     "Setting used in decoder to control hash behavior");
    }
    opts->Register("token-pool-realloc", &token_pool_realloc,
                   "number of tokens per alloc in token memory pool");
  }
};



class DecCore {
 public:

  typedef uint64 PairId;
  typedef fst::StdArc Arc;

  DecCore(Wfst *wfst,
          fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst,
          const DecCoreOptions &config);

  ~DecCore();

  void SetOptions(const DecCoreOptions &config) { config_ = config; }

/*
  /// Returns true if a final state was active on the last frame.
  bool ReachedFinal();

  /// GetBestPath gets the decoding traceback. If "use_final_probs" is true
  /// AND we reached a final state, it limits itself to final states;
  /// otherwise it gets the most likely token not taking into account
  /// final-probs. Returns true if the output best path was not the empty
  /// FST (will only return false in unusual circumstances where
  /// no tokens survived).
  
  bool GetBestPath(fst::MutableFst<LatticeArc> *fst_out,
                   bool use_final_probs = true);
*/

  void InitDecoding();
  void Decode(DecodableInterface *decodable);
  void AdvanceDecoding(DecodableInterface *decodable, int32 max_num_frames = -1);
  void FinalizeDecoding();

  bool GetBestTransitionPath();

  int32 NumFramesDecoded() const { return num_frames_decoded_; }

 private:

  inline PairId ConstructPair(WfstStateId fst_state, WfstStateId lm_state) {
    return static_cast<PairId>(fst_state) + (static_cast<PairId>(lm_state) << 32);
  }
  static inline WfstStateId PairToState(PairId state_pair) {
    return static_cast<WfstStateId>(static_cast<uint32>(state_pair));
  }
  static inline WfstStateId PairToLmState(PairId state_pair) {
    return static_cast<WfstStateId>(static_cast<uint32>(state_pair >> 32));
  }

  class Token {
   public:
    Token   *prev_;
    WfstArc  arc_;
    double   cost_;
    int32    ref_count_;

    inline Token(Token *prev, const WfstArc &arc, BaseFloat ac_cost):
        prev_(prev), arc_(arc), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        cost_ = prev->cost_ + arc.weight + ac_cost;
      } else {
        cost_ = arc.weight + ac_cost;
      }
    }

    inline bool operator < (const Token &other) {
      return cost_ > other.cost_;
    }

    inline static void GC(Token *tok, MemoryPool *pool) {
      while (--tok->ref_count_ == 0) {
        Token *prev = tok->prev_;
        tok->~Token();
        pool->FreeElem(tok);
        if (prev == NULL) return;
        else tok = prev;
      }
#ifdef KALDI_PARANOID
      KALDI_ASSERT(tok->ref_count_ > 0);
#endif
    }
  };

  inline Token* NewToken(Token *prev, const WfstArc &arc, BaseFloat ac_cost) {
    Token *tok = (Token*)token_pool_->MallocElem();
    new (tok) Token(prev, arc, ac_cost); // placement new
    return tok;
  }

  inline void DelToken(Token *tok) {
    Token::GC(tok, token_pool_);
  }


  typedef HashList<PairId, Token*>::Elem Elem;

  void   ClearToks(Elem *list);
  double GetCutoff(Elem *list_head, size_t *tok_count, BaseFloat *adaptive_beam, Elem **best_elem);
  void   PossiblyResizeHash(size_t num_toks);
  inline WfstStateId PropagateLm(WfstStateId lm_state, WfstArc *arc);
  double ProcessEmitting(DecodableInterface *decodable);
  void   ProcessNonemitting(double cutoff);

 private:
  Wfst *fst_;
  fst::DeterministicOnDemandFst<fst::StdArc> *lm_diff_fst_;

  HashList<PairId, Token*> cur_toks_;
  MemoryPool *token_pool_;

  DecCoreOptions config_;
  std::vector<PairId> queue_;     // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat>  tmp_array_; // used in GetCutoff.

  int32 num_frames_decoded_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecCore);
};

} // end namespace iot
} // end namespace kaldi.

#endif
