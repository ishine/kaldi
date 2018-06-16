#ifndef KALDI_IOT_LINK_H
#define KALDI_IOT_LINK_H

#include "base/kaldi-common.h"

namespace kaldi {
namespace iot {

class Link {
public:
  Link();
  ~Link();

  Link* Prev();
  Link* Next();
  void  InsertBefore(Link* ref);
  void  InsertAfter(Link* ref);
  int   IsLinked();
  void  Unlink();

private:
  Link *prev_;
  Link *next_;

  // hide
  KALDI_DISALLOW_COPY_AND_ASSIGN(Link);
};

} // namespace iot
} // namespace kaldi
#endif
