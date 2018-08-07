#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"
#include "fst/vector-fst.h"

typedef float score_t;

struct lattice_trans_basic_t
{
    int     to_node;
    score_t am_score;
    score_t lm_score;
    int     out_label;
    int    next;
};

struct lattice_trans_t : public lattice_trans_basic_t
{
    int in_label;
};


struct lattice_node_t
{
    int trans_head;
    int frame_id;
};

struct lattice_final_weight_t
{
    int node_id;
    score_t am_score;
    score_t lm_score;
};

class lattice_t
{
public:
    std::vector<lattice_node_t> nodes;
    std::vector<lattice_trans_t> trans;
    std::vector<lattice_final_weight_t> final_weights;

    int init_node;
    int end_node;
    lattice_t(){}
private:
    lattice_t remove_epsilon() const ;
};

using namespace kaldi;
bool GetLattice(Lattice& raw_fst, CompactLattice *ofst) {
  TopSort(&raw_fst);
  Invert(&raw_fst);  // make it so word labels are on the input.
  // (in phase where we get backward-costs).
  fst::ILabelCompare<LatticeArc> ilabel_comp;
  ArcSort(&raw_fst, ilabel_comp);  // sort on ilabel; makes
  // lattice-determinization more efficient.

  fst::DeterminizeLatticePrunedOptions lat_opts;
  lat_opts.max_mem = 123456789;
  auto lattice_beam = 123.45;
  DeterminizeLatticePruned(raw_fst, lattice_beam, ofst, lat_opts);
  raw_fst.DeleteStates();  // Free memory-- raw_fst no longer needed.
  Connect(ofst);  // Remove unreachable states... there might be
  // a small number of these, in some cases.
  // Note: if something went wrong and the raw lattice was empty,
  // we should still get to this point in the code without warnings or failures.
  return (ofst->NumStates() != 0);
}

void sogou_lat_to_fst(const lattice_t& lat, Lattice* fst)
{
  for(const auto _ __attribute__((unused)) : lat.nodes) fst->AddState();
  for(auto i=0u; i<lat.nodes.size(); i++)
    for(int tid=lat.nodes[i].trans_head; tid!=-1; tid = lat.trans[tid].next)
    {
      const auto & t = lat.trans[tid];
      LatticeArc arc(t.in_label, t.out_label, LatticeArc::Weight(t.lm_score,t.am_score), t.to_node);
      fst->AddArc(i, arc);
    }
  for(const auto f : lat.final_weights)
  {
    fst->SetFinal(f.node_id, LatticeArc::Weight(f.lm_score, f.am_score));
  }
  fst->SetStart(lat.init_node);
}

template <typename Arc>
lattice_trans_t to_sogou_trans(Arc arc)
{
  lattice_trans_t lt;
  lt.to_node = arc.nextstate;
  lt.in_label = arc.ilabel;
  lt.out_label = arc.olabel;
  lt.next = -1;
  lt.am_score = arc.weight.Weight().Value2();
  lt.lm_score = arc.weight.Weight().Value1();
  return lt;
}
void fst_to_sogou_lat(const CompactLattice& clat, lattice_t* lat)
{
  lat->nodes.clear();
  lat->trans.clear();
  lat->final_weights.clear();
  lat->init_node = clat.Start();
  int i=0;
  for (fst::StateIterator<CompactLattice> siter(clat); !siter.Done(); siter.Next(),i++) 
  {
    auto state_id = siter.Value();
    assert(state_id == i);
    lat->nodes.push_back(lattice_node_t{-1,0});

    // Get state i's final weight; if == Weight::Zero() => non-final. 
    auto weight = clat.Final(state_id);
    if (weight != decltype(weight)::Zero())
    {
      lat->final_weights.push_back({state_id, weight.Weight().Value1(), weight.Weight().Value2()});
    }

    for (fst::ArcIterator<CompactLattice> aiter(clat, i); !aiter.Done(); aiter.Next())
    {
      auto &arc = aiter.Value();
      auto lt = to_sogou_trans(arc);
      lt.next = lat->nodes.back().trans_head;
      lat->nodes.back().trans_head = lat->trans.size();
      lat->trans.push_back(lt);
    }
  }
  
}

lattice_t lattice_t::remove_epsilon() const
{
  lattice_t ret;
  CompactLattice clat;
  Lattice raw_fst;
  sogou_lat_to_fst(*this, &raw_fst);
  GetLattice(raw_fst, &clat);
  //raw_fst.Write("raw.fst");
  //clat.Write("clat.fst");
  fst_to_sogou_lat(clat, &ret);
  return ret;
}