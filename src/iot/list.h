#ifndef KALDI_IOT_LIST_H
#define KALDI_IOT_LIST_H

#include "base/kaldi-common.h"
#include "iot/link.h"

namespace kaldi {
namespace iot {

class List {
public:
  List();
  List(size_t offset);
  ~List();

  void   SetOffset(size_t offset);
  int    IsEmpty();
  size_t NumElems(); // O(n)
  void*  Head();
  void*  Tail();
  void*  Prev(void* node);
  void*  Next(void* node);
  void   InsertHead(void* node);
  void   InsertTail(void* node);
  void   UnlinkNode(void* node);
  void   UnlinkAll();

private:
  Link* GetLinkFromNode(void* node) { return (Link*) ((size_t)node + offset_); }
  void* GetNodeFromLink(Link* link) { return (link == &origin_) ? NULL : (void*) ((size_t)link - offset_); }

  Link   origin_;
  size_t offset_;

  // hide
  KALDI_DISALLOW_COPY_AND_ASSIGN(List);
};

} // namespace iot
} // namespace kaldi
#endif
