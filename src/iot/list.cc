#include "iot/list.h"

namespace kaldi {
namespace iot {

List::List() :
  origin_(),
  offset_(0)
{ }

List::List(size_t offset) :
  origin_(),
  offset_(offset)
{ }

List::~List() {
  UnlinkAll();
}

void List::SetOffset(size_t offset) {
  offset_ = offset;
}

int List::IsEmpty() {
  return (Head() == NULL);
}

size_t List::NumElems() {
  size_t n = 0;
  for (void *e = Head(); e != NULL; e = Next(e)) {
    n++;
  }
  return n;
}

void* List::Head() {
  return GetNodeFromLink(origin_.Next());
}

void* List::Tail() {
  return GetNodeFromLink(origin_.Prev());
}

void* List::Prev(void* node) {
  return GetNodeFromLink(GetLinkFromNode(node)->Prev());
}

void* List::Next(void* node) {
  return GetNodeFromLink(GetLinkFromNode(node)->Next());
}

void List::InsertHead(void* node) {
  UnlinkNode(node);
  GetLinkFromNode(node)->InsertAfter(&origin_);
}

void List::InsertTail(void* node) {
  UnlinkNode(node);
  GetLinkFromNode(node)->InsertBefore(&origin_);
}

void List::UnlinkNode(void* node) {
  Link *link = GetLinkFromNode(node);
  if (link->IsLinked()) {
    link->Unlink();
  }
}

void List::UnlinkAll() {
  while (!IsEmpty()) {
    UnlinkNode(Head());
  }
}

} // namespace iot
} // namespace kaldi
