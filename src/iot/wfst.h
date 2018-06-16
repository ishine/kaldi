#ifndef KALDI_IOT_WFST_H
#define KALDI_IOT_WFST_H

#include "base/kaldi-common.h"
#include "iot/common.h"

namespace kaldi {
namespace iot {

typedef uint32     WfstStateId;
typedef uint32     WfstArcId;
typedef uint32     WfstLabelId;
typedef BaseFloat  WfstWeight;

const WfstLabelId kWfstEpsilon       = 0;
const std::string kWfstEpsilonString = "<eps>";
const WfstWeight  kWfstDefaultWeight = 0.0f;

class WfstSymbolTable {
public:
  WfstSymbolTable() :
    symbols_()
  { }

  ~WfstSymbolTable() {
    symbols_.clear();
  }

  std::string Index2Symbol(WfstLabelId id);
  WfstLabelId Symbol2Index(const std::string &symbol);

  size_t Size() { return symbols_.size(); }

  void Read(std::istream &is, bool binary);
  void Write(std::ostream &os, bool binary);

  int ReadOpenFst(FILE *fp);

private:
  std::vector<std::string> symbols_;
};


struct WfstArc {
  WfstLabelId ilabel;
  WfstLabelId olabel;
  WfstWeight  weight;
  WfstStateId dst;

  WfstArc() :
    ilabel(kWfstEpsilon),
    olabel(kWfstEpsilon),
    weight(kWfstDefaultWeight),
    dst(0)
  { }

  WfstArc(WfstLabelId i, WfstLabelId o, WfstWeight w, WfstStateId d) :
    ilabel(i),
    olabel(o),
    weight(w),
    dst(d)
  { }
};


struct WfstState {
  WfstArcId  arc_base;
  int32      num_arcs;
  WfstWeight weight;

  WfstState() :
    arc_base(0),
    num_arcs(0),
    weight(kWfstDefaultWeight)
  { }
};


class Wfst {
public:
  Wfst();
  ~Wfst();

  WfstStateId Start();

  WfstState* State(WfstStateId i);
  WfstArc*   Arc(WfstArcId j);
  WfstWeight Final(WfstStateId i);
  
  size_t    NumStates();
  size_t    NumArcs();

  WfstSymbolTable* InputTable()  { return itable_; }
  WfstSymbolTable* OutputTable() { return otable_; }

  void      Read(std::istream &is, bool binary);
  void      Write(std::ostream &os, bool binary);

  int       ReadOpenFst(FILE *fp, FILE *itable_fp, FILE *otable_fp);

  int       WriteDot(FILE *fp);

private:

  WfstStateId  start_;

  WfstState   *states_;
  uint64       num_states_;

  WfstArc     *arcs_;
  uint64       num_arcs_;

  WfstSymbolTable *itable_;
  WfstSymbolTable *otable_;
};

} // namespace iot
} // namespace kaldi

#endif
