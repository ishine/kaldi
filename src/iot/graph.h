#ifndef KALDI_IOT_GRAPH_H
#define KALDI_IOT_GRAPH_H

#include "base/kaldi-common.h"

#include "iot/list.h"
#include "iot/common.h"

namespace kaldi {
namespace iot {

struct Node;

struct Arc {
  Link  arcs_link;
  Link  src_link;
  Link  dst_link;

  size_t id;
  std::string name;

  Node *src;
  Node *dst;
  int32 ilabel;
  int32 olabel;
  float32 weight;

  void *aux;

  Arc() :
    arcs_link(), src_link(), dst_link(),
    id(-1), name(""),
    src(NULL), dst(NULL), ilabel(-1), olabel(-1), weight(0.0),
    aux(NULL)
  { }

  ~Arc() { }
};

struct Node {
  Link  nodes_link;

  size_t id;
  std::string name;
  
  List  iarcs;
  List  oarcs;

  void *aux;

  Node() :
    nodes_link(),
    id(-1),
    name(""),
    iarcs(offsetof(Arc, dst_link)),
    oarcs(offsetof(Arc, src_link)),
    aux(NULL)
  { }

  ~Node() { }
};

class Graph {
public:
  Graph();
  ~Graph();

  size_t NumNodes();
  size_t NumArcs();
  Node*  AddNode();
  int    DelNode(Node *node);
  Arc*   AddArc(Node *src, Node *dst);
  int    DelArc(Arc *arc);
  
  int    MarkNodeId();
  int    WriteDot(FILE *fp);

private:
  List nodes_;
  List arcs_;
};

} // namespace iot
} // namespace kaldi
#endif
