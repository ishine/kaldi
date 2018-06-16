#include "iot/wfst.h"

namespace kaldi {
namespace iot {
  // symbol table
  std::string WfstSymbolTable::Index2Symbol(WfstLabelId i) {
    return symbols_[i];
  }

  WfstLabelId WfstSymbolTable::Symbol2Index(const std::string &symbol) {
    for (size_t i = 0; i != symbols_.size(); i++) {
      if (symbol == symbols_[i]) {
        return i;
      }
    }
    return -1;
  }

  void WfstSymbolTable::Read(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<WfstSymbolTable>");
    ExpectToken(is, binary, "<TableSize>");
    uint64 table_size = 0;
    ReadBasicType(is, binary, &table_size);
    symbols_.resize(table_size);
    for (size_t i = 0; i != symbols_.size(); i++) {
      ReadToken(is, binary, &symbols_[i]);
    }
  }

  void WfstSymbolTable::Write(std::ostream &os, bool binary) {
    WriteToken(os, binary, "<WfstSymbolTable>");
    WriteToken(os, binary, "<TableSize>");
    WriteBasicType(os, binary, (uint64)symbols_.size());
    for (size_t i = 0; i != symbols_.size(); i++) {
      WriteToken(os, binary, symbols_[i]);
    }
  }

  int WfstSymbolTable::ReadOpenFst(FILE *fp) {
    char line[2048];
    const char *delim = " \t\n";
    std::vector<char*> toks;
    while (fgets(line, sizeof(line), fp) != NULL) {
      if (Tokenize(line, delim, toks) == 2) {
        KALDI_ASSERT(symbols_.size() == atoi(toks[1]));
        symbols_.push_back(std::string(toks[0]));
      }
    }
    return 0;
  }

  // wfst
  Wfst::Wfst() :
    start_(0),
    states_(NULL),
    num_states_(0),
    arcs_(NULL),
    num_arcs_(0),
    itable_(NULL),
    otable_(NULL)
  { }

  Wfst::~Wfst() {
    DELETE_ARRAY(states_);
    DELETE_ARRAY(arcs_);
    DELETE(itable_);
    DELETE(otable_);
  }

  WfstStateId Wfst::Start() {
    return start_;
  }
  
  WfstState* Wfst::State(WfstStateId i) {
    assert(i < num_states_ && "node index out of range in Wfst::State.");
    return &states_[i];
  }

  WfstWeight Wfst::Final(WfstStateId i) {
    assert(i < num_states_ && "node index out of range in Wfst::Final.");
    return states_[i].weight;
  }

  WfstArc* Wfst::Arc(WfstArcId j) {
    assert(j < num_arcs_ && "arc index out of range in Wfst::GetArc.");
    return &arcs_[j];
  }

  size_t Wfst::NumStates() {
    return num_states_;
  }

  size_t Wfst::NumArcs() {
    return num_arcs_;
  }

  void Wfst::Read(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<Wfst>");

    ExpectToken(is, binary, "<Start>");
    ReadBasicType(is, binary, &start_);

    ExpectToken(is, binary, "<NumStates>");
    ReadBasicType(is, binary, &num_states_);

    ExpectToken(is, binary, "<NumArcs>");
    ReadBasicType(is, binary, &num_arcs_);

    states_ = new WfstState[num_states_];
    arcs_   = new WfstArc[num_arcs_];

    ExpectToken(is, binary, "<States>");
    is.read(reinterpret_cast<char*>(states_), num_states_ * sizeof(WfstState));

    ExpectToken(is, binary, "<Arcs>");
    is.read(reinterpret_cast<char*>(arcs_), num_arcs_ * sizeof(WfstArc));

    bool has_itable = false;
    ExpectToken(is, binary, "<InputTable>");
    ReadBasicType(is, binary, &has_itable);
    if (has_itable) {
      itable_ = new WfstSymbolTable;
      itable_->Read(is, binary);
    }

    bool has_otable = false;
    ExpectToken(is, binary, "<OutputTable>");
    ReadBasicType(is, binary, &has_otable);
    if (has_otable) {
      otable_ = new WfstSymbolTable;
      otable_->Read(is, binary);
    }
  }

  void Wfst::Write(std::ostream &os, bool binary) {
    WriteToken(os, binary, "<Wfst>");

    WriteToken(os, binary, "<Start>");
    WriteBasicType(os, binary, start_);

    WriteToken(os, binary, "<NumStates>");
    WriteBasicType(os, binary, num_states_);

    WriteToken(os, binary, "<NumArcs>");
    WriteBasicType(os, binary, num_arcs_);

    WriteToken(os, binary, "<States>");
    os.write(reinterpret_cast<const char*>(states_), num_states_ * sizeof(WfstState));

    WriteToken(os, binary, "<Arcs>");
    os.write(reinterpret_cast<const char*>(arcs_), num_arcs_ * sizeof(WfstArc));

    WriteToken(os, binary, "<InputTable>");
    if (itable_ != NULL) {
      WriteBasicType(os, binary, true);
      itable_->Write(os, binary);
    } else {
      WriteBasicType(os, binary, false);
    }

    WriteToken(os, binary, "<OutputTable>");
    if (otable_ != NULL) {
      WriteBasicType(os, binary, true);
      otable_->Write(os, binary);
    } else {
      WriteBasicType(os, binary, false);
    }
  }

  int Wfst::ReadOpenFst(FILE *fp, FILE *itable_fp, FILE *otable_fp) {
    char line[2048];
    const char *delim = " \t\n";

    if (itable_fp != NULL) {
      itable_ = new WfstSymbolTable;
      itable_->ReadOpenFst(itable_fp);
    }

    if (otable_fp != NULL) {
      otable_ = new WfstSymbolTable;
      otable_->ReadOpenFst(otable_fp);
    }

    std::vector<char*> toks;

    // 1st-sweep
    size_t i = 0, n = 0, maxi = 0;
    while(fgets(line, sizeof(line), fp) != NULL) {
      n = Tokenize(line, delim, toks);
      KALDI_ASSERT(n == 0 || n == 1 || n == 2 || n == 4 || n == 5);
      i = atoi(toks[0]);
      maxi = (i>maxi) ? i : maxi;

      if (n == 4 || n == 5) {
        num_arcs_++;
      }
    }

    start_ = 0;  // openfst assumption

    num_states_ = maxi + 1;
    states_ = new WfstState[num_states_];
    KALDI_ASSERT(states_ != NULL);

    arcs_  = new WfstArc[num_arcs_];
    KALDI_ASSERT(arcs_ != NULL);

    // 2nd-sweep
    // openfst file format:
    //   for arc:
    //     src  dst  ilabel|ilabe_id  olabel|olabel_id  [weight]
    //   for final node:
    //     final_id  [weight]
    rewind(fp);

    WfstStateId src, dst;
    WfstLabelId ilabel, olabel;
    WfstWeight weight;

    WfstStateId si = 0;
    WfstArcId  ai = 0;;
    // loop invariant: states[0, si), arcs[0,ai)
    while (fgets(line, sizeof(line), fp) != NULL) {
      n = Tokenize(line, delim, toks);
      KALDI_ASSERT(n == 0 || n == 1 || n == 2 || n == 4 || n == 5);
      
      if (n == 1 || n == 2) {  // final
        WfstStateId f = atoi(toks[0]);
        states_[f].weight = (n == 1) ? kWfstDefaultWeight : (WfstWeight)atof(toks[1]);

      } else if ( n == 4 || n == 5) {  // arc
        src    = (WfstStateId)atoi(toks[0]);
        dst    = (WfstStateId)atoi(toks[1]);
        ilabel = (WfstLabelId)atoi(toks[2]);
        olabel = (WfstLabelId)atoi(toks[3]);
        weight = (n == 4) ? kWfstDefaultWeight : (WfstWeight)atof(toks[4]);
        
        if (src > si) { // new state
          si = src;
          states_[si].arc_base = ai;
        }

        WfstState *state = &states_[si];
        WfstArc *arc = &arcs_[ai];

        arc->ilabel = ilabel;
        arc->olabel = olabel;
        arc->weight = weight;
        arc->dst    = dst;

        state->num_arcs++;
        ai++;
      }
    }

    return 0;
  }

  int Wfst::WriteDot(FILE *fp) {
    fprintf(fp, "digraph {\n"
                "rankdir = LR;\n"
                "label = \"WFST\";\n"
                "center = 1;\n"
                "ranksep = \"0.4\";\n"
                "nodesep = \"0.25\";\n");

    for (WfstStateId i = 0; i < NumStates(); i++) {
      WfstState *state = State(i);
      fprintf(fp, "%d [label = \"%d\", shape = %s, style = %s, fontsize = 14]\n", i, i,
        ((state->weight != kWfstDefaultWeight) ? "doublecircle" : "circle"), // final
        (i == 0) ? "bold" : "solid");  // isititial

      WfstArc *arc = Arc(state->arc_base);
      for (int32 j = 0; j < state->num_arcs; j++,arc++) {
        fprintf(fp, "\t%d -> %d [label=\"%d:%d:%f\", fontsize = 14];\n",
          i,
          arc->dst,
          arc->ilabel,
          arc->olabel,
          arc->weight);
      }
    }
    fprintf(fp, "}\n");
    return 0;
  }
  
} // namespace iot
} // namespace kaldi
