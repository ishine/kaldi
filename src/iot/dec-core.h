// iot/dec-core.h

#ifndef KALDI_IOT_DEC_CORE_H_
#define KALDI_IOT_DEC_CORE_H_

#include "util/stl-utils.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "fstext/fstext-lib.h"
#include "lat/determinize-lattice-pruned.h"
#include "lat/kaldi-lattice.h"
#include "base/timer.h"

#include "iot/memory-pool.h"
#include "iot/wfst.h"
#include "iot/language-model.h"

namespace kaldi {
namespace iot {

struct DecCoreConfig {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat lattice_beam;
  BaseFloat lm_token_beam;
  int32 prune_interval;
  bool determinize_lattice; // not inspected by this class... used in
                            // command-line program.
  BaseFloat beam_delta; // has nothing to do with beam_ratio
  BaseFloat hash_ratio;
  BaseFloat prune_scale;   // Note: we don't make this configurable on the command line,
                           // it's not a very important parameter.  It affects the
                           // algorithm that prunes the tokens as we go.
  // Most of the options inside det_opts are not actually queried by the
  // LatticeFasterDecoder class itself, but by the code that calls it, for
  // example in the function DecodeUtteranceLatticeFaster.

  int32 token_pool_realloc;
  int32 link_pool_realloc;
  
  fst::DeterminizeLatticePhonePrunedOptions det_opts;

  bool use_cost_offset;
  bool debug_mode;

  DecCoreConfig() : 
    beam(16.0),
    max_active(std::numeric_limits<int32>::max()),
    min_active(200),
    lattice_beam(10.0),
    lm_token_beam(3.0f),
    prune_interval(25),
    determinize_lattice(true),
    beam_delta(0.5),
    hash_ratio(2.0),
    prune_scale(0.1),
    token_pool_realloc(2048),
    link_pool_realloc(2048),
    use_cost_offset(true),
    debug_mode(false)
  { }

  void Register(OptionsItf *opts) {
    det_opts.Register(opts);
    opts->Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    opts->Register("max-active", &max_active, "Decoder max active states.  Larger->slower; "
                   "more accurate");
    opts->Register("min-active", &min_active, "Decoder minimum #active states.");
    opts->Register("lattice-beam", &lattice_beam, "Lattice generation beam.  Larger->slower, "
                   "and deeper lattices");
    opts->Register("lm-token-beam", &lm_token_beam, "lm token beam.  Larger->slower");
    opts->Register("prune-interval", &prune_interval, "Interval (in frames) at "
                   "which to prune tokens");
    opts->Register("determinize-lattice", &determinize_lattice, "If true, "
                   "determinize the lattice (lattice-determinization, keeping only "
                   "best pdf-sequence for each word-sequence).");
    opts->Register("beam-delta", &beam_delta, "Increment used in decoding-- this "
                   "parameter is obscure and relates to a speedup in the way the "
                   "max-active constraint is applied.  Larger is more accurate.");
    opts->Register("hash-ratio", &hash_ratio, "Setting used in decoder to "
                   "control hash behavior");
    opts->Register("token-pool-realloc", &token_pool_realloc,
                   "number of tokens per alloc in memory pool");
    opts->Register("link-pool-realloc", &link_pool_realloc,
                   "number of forward-links per alloc in memory pool");
    opts->Register("use-cost-offset", &use_cost_offset,
                   "use-cost-offset=true/false, default=true");
    opts->Register("debug-mode", &debug_mode,
                   "debug-mode=true/false, default=false");
  }

  void Check() const {
    KALDI_ASSERT(beam > 0.0 && max_active > 1 && lattice_beam > 0.0
                 && prune_interval > 0 && beam_delta > 0.0 && hash_ratio >= 1.0
                 && prune_scale > 0.0 && prune_scale < 1.0);
  }
};


class DecCore {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  DecCore(Wfst *fst,
          const TransitionModel &trans_model, 
          const DecCoreConfig &config);
  ~DecCore();

  void AddExtLM(LmFst<fst::StdArc>* lm);

  void SetOptions(const DecCoreConfig &config) { config_ = config; }
  const DecCoreConfig &GetOptions() const { return config_; }

  // offline ASR interfaces
  bool Decode(DecodableInterface *decodable);

  // online streaming ASR interfaces
  void StartSession(const char* session_key = NULL);
  void AdvanceDecoding(DecodableInterface *decodable, int32 max_num_frames = -1);
  void StopSession();

  int32 TrailingSilenceFrames() const;

  inline int32 NumFramesDecoded() const { return token_net_.size() - 1; }
  
  BaseFloat FinalRelativeCost() const;
  bool ReachedFinal() const {
    return FinalRelativeCost() != std::numeric_limits<BaseFloat>::infinity();
  }

  struct BestPathIterator {
    void *tok;
    int32 frame;
    // note, "frame" is the frame-index of the frame you'll get the
    // transition-id for next time, if you call TraceBackBestPath on this
    // iterator (assuming it's not an epsilon transition).  Note that this
    // is one less than you might reasonably expect, e.g. it's -1 for
    // the nonemitting transitions before the first frame.
    BestPathIterator(void *t, int32 f): tok(t), frame(f) { }
    bool Done() { return tok == NULL; }
  };

  BestPathIterator BestPathEnd(bool use_final_probs, BaseFloat *final_cost = NULL) const;
  BestPathIterator TraceBackBestPath(BestPathIterator iter, LatticeArc *arc) const;
  bool GetBestPath(Lattice *ofst, bool use_final_probs = true) const;
  bool TestGetBestPath(bool use_final_probs = true) const;

  void PrintAlignmentDetail(Lattice &lat);

  bool GetRawLattice(Lattice *ofst, bool use_final_probs = true) const;
  bool GetRawLatticePruned(Lattice *ofst, bool use_final_probs, BaseFloat beam) const;


 private:

  struct Token;
  struct TokenList;
  struct ForwardLink;

  struct RescoreToken;
  struct RescoreTokenList;

  typedef WfstStateId ViterbiState;
  typedef WfstStateId LmState;

/* ----- For standard on-the-fly (LA o LM) composition implementation -----
  typedef uint64 ViterbiState;

  inline ViterbiState (WfstStateId la_state, WfstStateId lm_state) {
    return static_cast<ViterbiState>(la_state) + (static_cast<ViterbiState>(lm_state) << 32);
  }

  static inline void DecomposeViterbiState(ViterbiState viterbi_state, Token* tok, WfstStateId *la_state, WfstStateId *lm_state) {
    *la_state = static_cast<WfstStateId>(static_cast<uint32>(viterbi_state));
    *lm_state = static_cast<WfstStateId>(static_cast<uint32>(viterbi_state >> 32));
  }

  static inline WfstStateId ExtractLaState(ViterbiState viterbi_state) {
    return static_cast<WfstStateId>(static_cast<uint32>(viterbi_state));
  }
  static inline WfstStateId ExtractLmState(ViterbiState viterbi_state) {
    return static_cast<WfstStateId>(static_cast<uint32>(viterbi_state >> 32));
  }
*/

/*------------------------------ Token ------------------------------*/
  struct Token {
    BaseFloat total_cost;
    BaseFloat extra_cost;
    ForwardLink *links;
    Token *next;
    Token *backpointer;

    RescoreTokenList *rtoks; // rescore token list(co-hypotheses) associate with this token

    inline Token(
        BaseFloat total_cost,
        BaseFloat extra_cost,
        ForwardLink *links,
        Token *next,
        Token *backpointer) :
      total_cost(total_cost),
      extra_cost(extra_cost),
      links(links),
      next(next),
      backpointer(backpointer),
      rtoks(NULL)
    { }

    inline ~Token() { }
  };

  inline Token* NewToken(
      BaseFloat total_cost,
      BaseFloat extra_cost,
      ForwardLink *links,
      Token *next,
      Token *backpointer) {
    Token *tok = (Token*) token_pool_->MallocElem();
    // placement new
    new (tok) Token(total_cost, extra_cost, links, next, backpointer);
    return tok;
  }

  inline void DeleteToken(Token *tok) {
    // Forward links are not owned by token
    if (lm_fst_ != NULL && tok->rtoks != NULL) {
      GcRescoreTokenList(tok);
    }
    tok->~Token();
    token_pool_->FreeElem(tok);
  }

  struct TokenList {
    Token *toks;
    bool must_prune_forward_links;
    bool must_prune_tokens;

    TokenList() :
      toks(NULL),
      must_prune_forward_links(true),
      must_prune_tokens(true)
    { }
  };


  /*------------------------------ ForwardLinks ------------------------------*/
  struct ForwardLink {
    Token *dst_tok; // the next token [or NULL if represents final-state]
    Label ilabel;
    Label olabel;
    BaseFloat graph_cost; // graph cost of traversing link (contains LM, etc.)
    BaseFloat acoustic_cost; // acoustic cost (pre-scaled) of traversing link
    ForwardLink *next;

    inline ForwardLink(
        Token *dst_tok,
        Label ilabel,
        Label olabel,
        BaseFloat graph_cost,
        BaseFloat acoustic_cost,
        ForwardLink *next) :
      dst_tok(dst_tok),
      ilabel(ilabel),
      olabel(olabel),
      graph_cost(graph_cost),
      acoustic_cost(acoustic_cost),
      next(next)
    { }

    inline ~ForwardLink() { }
  };

  inline ForwardLink* NewForwardLink(
      Token *dst_tok,
      Label ilabel,
      Label olabel,
      BaseFloat graph_cost,
      BaseFloat acoustic_cost,
      ForwardLink *next) {
    ForwardLink *link = (ForwardLink*)link_pool_->MallocElem();
    // placement new
    new (link) ForwardLink(dst_tok, ilabel, olabel, graph_cost, acoustic_cost, next);
    return link;
  }

  inline void DeleteForwardLink(ForwardLink *link) {
    link->~ForwardLink();
    link_pool_->FreeElem(link);
  }

  inline void DeleteLinksFromToken(Token *tok) {
    ForwardLink *l = tok->links, *next = NULL;
    while (l != NULL) {
      next = l->next;
      DeleteForwardLink(l);
      l = next;
    }
    tok->links = NULL;
  }


/* ------------------------------ RescoreToken ------------------------------ */
  struct RescoreToken {
    LmState state;
    BaseFloat cost;
    struct RescoreToken *next;

    inline RescoreToken(LmState state, BaseFloat cost, RescoreToken *next)
     : state(state), cost(cost), next(next)
    { }

    inline ~RescoreToken() { }
  };

  inline RescoreToken* NewRescoreToken(LmState lm_state, BaseFloat cost, RescoreToken *next) {
    RescoreToken *tok = (RescoreToken*) rescore_token_pool_->MallocElem();
    new (tok) RescoreToken(lm_state, cost, next);  // placement new
    return tok;
  }

  inline void DeleteRescoreToken(RescoreToken *tok) {
    tok->~RescoreToken();
    rescore_token_pool_->FreeElem(tok);
  }

  struct RescoreTokenList {
    RescoreToken *head;
    RescoreToken *best;
    int32 rc;

    RescoreTokenList() : 
      head(NULL),
      best(NULL),
      rc(0)
    { }
  };

  inline void GcRescoreTokenList(Token* tok) {
    KALDI_ASSERT(tok->rtoks != NULL);
    if (--(tok->rtoks->rc) == 0) {
      DeleteRescoreTokenList(tok);
    }
    tok->rtoks = NULL;
  }

  inline void DeleteRescoreTokenList(Token* tok) {
    KALDI_ASSERT(tok->rtoks != NULL);
    RescoreToken *p = tok->rtoks->head, *next;
    while (p != NULL) {
      next = p->next;
      DeleteRescoreToken(p);
      p = next;
    }
    tok->rtoks->head = NULL;
    tok->rtoks->best = NULL;
    DELETE(tok->rtoks);
  }

  inline void AddRescoreToken(RescoreTokenList *list, LmState state, BaseFloat cost);
  // this should be the only way to setup rescore token list to a token
  inline void HookRescoreTokenList(Token *tok, RescoreTokenList *rtoks);
  inline void MergeRescoreTokenList(Token *from, Token *to);


/* ------------------------------ diagnostic ------------------------------ */
  inline void PreFrame();
  inline void PostFrame();
  void PreSession();
  void PostSession();

  typedef HashList<ViterbiState, Token*>::Elem Elem;

  void PossiblyResizeHash(size_t num_toks);

  inline void PropagateLm(Token *from, WfstArc *arc, Token *to);

  // FindOrAddToken either locates a token in hash of token_hash_, or if necessary
  // inserts a new, empty token (i.e. with no forward links) for the current
  // frame.  [note: it's inserted if necessary into hash token_hash_ and also into the
  // singly linked list of tokens active on this frame (whose head is at
  // token_net_[t]).  The 't' argument is the acoustic frame
  // index plus one, which is used to index into the token_net_ array.
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true if the
  // token was newly created or the cost changed.
  inline Token *TokenViterbi(Token *tok, int32 t, ViterbiState to_state, bool *changed);

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

  // prunes outgoing links for all tokens in token_net_[t]
  // it's called by PruneTokenNet
  // all links, that have link_extra_cost > lattice_beam are pruned
  // delta is the amount by which the extra_costs must change
  // before we set *extra_costs_changed = true.
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned
  void PruneForwardLinks(int32 t, 
                         bool *extra_costs_changed,
                         bool *links_pruned,
                         BaseFloat delta);

  // PruneForwardLinksFinal is a version of PruneForwardLinks that we call
  // on the final frame.  If there are final tokens active, it uses
  // the final-probs for pruning, otherwise it treats all tokens as final.
  void PruneForwardLinksFinal();

  void PruneTokenList(int32 t);

  // Go backwards through still-alive tokens, pruning them if the
  // forward+backward cost is more than lat_beam away from the best path.  It's
  // possible to prove that this is "correct" in the sense that we won't lose
  // anything outside of lat_beam, regardless of what happens in the future.
  // delta controls when it considers a cost to have changed enough to continue
  // going backward and propagating the change.  larger delta -> will recurse
  // less far.
  void PruneTokenNet(BaseFloat delta);

  /// Gets the weight cutoff.  Also counts the active tokens.
  BaseFloat GetCutoff(Elem *list_head, 
                      size_t *tok_count,
                      BaseFloat *adaptive_beam, 
                      Elem **best_elem);

  BaseFloat ProcessEmitting(DecodableInterface *decodable);
  void ProcessNonemitting(BaseFloat cost_cutoff);

  // There are various cleanup tasks... the the token_hash_ structure contains
  // singly linked lists of Token pointers, where Elem is the list type.
  // It also indexes them in a hash, indexed by state (this hash is only
  // maintained for the most recent frame).  token_hash_.Clear()
  // deletes them from the hash and returns the list of Elems.  The
  // function DeleteElems calls token_hash_.Delete(elem) for each elem in
  // the list, which returns ownership of the Elem to the token_hash_ structure
  // for reuse, but does not delete the Token pointer.  The Token pointers
  // are reference-counted and are ultimately deleted in PruneTokenList,
  // but are also linked together on each frame by their own linked-list,
  // using the "next" pointer.  We delete them manually.
  void DeleteElems(Elem *list);

  // This function takes a singly linked list of tokens for a single frame, and
  // outputs a list of them in topological order (it will crash if no such order
  // can be found, which will typically be due to decoding graphs with epsilon
  // cycles, which are not allowed).  Note: the output list may contain NULLs,
  // which the caller should pass over; it just happens to be more efficient for
  // the algorithm to output a list that contains NULLs.
  static void TopSortTokens(Token *tok_list, std::vector<Token*> *topsorted_list);

  void ClearTokenNet();

  const char * session_key_; // no ownership

  MemoryPool *token_pool_;
  MemoryPool *link_pool_;
  MemoryPool *rescore_token_pool_;

  HashList<ViterbiState, Token*> token_hash_;
  std::vector<TokenList> token_net_;

  std::vector<ViterbiState> queue_;
  std::vector<BaseFloat> tmp_array_;

  Wfst *fst_;  // no ownership
  std::vector<LmFst<fst::StdArc>*> ext_lm_;  // no ownership
  LmFst<fst::StdArc> *lm_fst_;

  const TransitionModel &trans_model_;

  std::vector<BaseFloat> cost_offsets_;
  DecCoreConfig config_;
  int32 num_toks_; // current total #toks allocated...
  bool warned_;

  /// decoding_finalized_ is true if someone called FinalizeDecoding().  [note,
  /// calling this is optional].  If true, it's forbidden to decode more.  Also,
  /// if this is set, then the output of ComputeFinalCosts() is in the next
  /// three variables.  The reason we need to do this is that after
  /// FinalizeDecoding() calls PruneTokensForFrame() for the final frame, some
  /// of the tokens on the last frame are freed, so we free the list from token_hash_
  /// to avoid having dangling pointers hanging around.
  bool decoding_finalized_;
  /// For the meaning of the next 3 variables, see the comment for
  /// decoding_finalized_ above., and ComputeFinalCosts().
  unordered_map<Token*, BaseFloat> final_costs_;
  BaseFloat final_relative_cost_;
  BaseFloat final_best_cost_;

  float timer_;
  BaseFloat cutoff_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecCore);
};


} // end namespace iot
} // end namespace kaldi.
#endif
