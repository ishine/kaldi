#include "iot/graph.h"

namespace kaldi {
namespace iot {

Graph::Graph() :
  nodes_(offsetof(Node, nodes_link)),
  arcs_(offsetof(Arc, arcs_link))
{ }

Graph::~Graph() {
  while(!arcs_.IsEmpty()) {
    delete (Arc*)arcs_.Head();
  }
  while(!nodes_.IsEmpty()) {
    delete (Node*)nodes_.Head();
  }
}

size_t Graph::NumNodes() {
  return nodes_.NumElems();
}

size_t Graph::NumArcs() {
  return arcs_.NumElems();
}
  
Node* Graph::AddNode() {
  Node *node = new Node;
  nodes_.InsertTail(node);
  return node;
}

int Graph::DelNode(Node *node) {
  while(!node->iarcs.IsEmpty()) {
    delete (Arc*)node->iarcs.Head();
  }
  while(!node->oarcs.IsEmpty()) {
    delete (Arc*)node->oarcs.Head();
  }
  delete node;
  return 0;
}

Arc* Graph::AddArc(Node *src, Node *dst) {
  Arc *arc = new Arc;
  arcs_.InsertTail(arc);
  arc->src = src;  src->oarcs.InsertTail(arc);
  arc->dst = dst;  dst->iarcs.InsertTail(arc);
  return arc;
}

int Graph::DelArc(Arc *arc) {
  delete arc;
  return 0;
}


int Graph::MarkNodeId() {
  int32 i = 0;
  for (Node *p = (Node*)nodes_.Head(); p != NULL; p = (Node*)nodes_.Next(p)) {
    p->id = i++;
  }
  return 0;
}

int Graph::WriteDot(FILE *fp) {
  MarkNodeId();

  fprintf(fp, 
    "digraph {\n"
    "rankdir = LR;\n"
    "label = \"Graph\";\n"
    "center = 1;\n"
    "ranksep = \"0.4\";\n"
    "nodesep = \"0.25\";\n");
  for (Node *node = (Node*)nodes_.Head(); node != NULL; node = (Node*)nodes_.Next(node)) {
    fprintf(fp, "%lu [label = \"%s\", fontsize = 14]\n", node->id, node->name.c_str());
    for (Arc *arc = (Arc*)node->oarcs.Head(); arc != NULL; arc = (Arc*)node->oarcs.Next(arc)) {
      fprintf(fp, "\t%lu -> %lu [label=\"%s\", fontsize = 14];\n", node->id, arc->dst->id, arc->name.c_str());
    }
  }
  fprintf(fp, "}\n");

  return 0;
}

} // namespace iot
} // namespace kaldi
