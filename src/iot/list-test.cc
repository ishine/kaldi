#include "iot/list.h"

namespace kaldi {
namespace iot {

void UnitTestList() {
  struct A {
    Link link;
    char* c;
    int i;
  };

  A *a1 = new A;
  A *a2 = new A;

  List *l = new List(offsetof(A, link));
  KALDI_ASSERT(l->IsEmpty());
  
  l->InsertHead(a1);
  KALDI_ASSERT(l->NumElems() == 1);
  KALDI_ASSERT((A*)l->Head() == a1);
  KALDI_ASSERT((A*)l->Tail() == a1);

  l->InsertTail(a2);
  KALDI_ASSERT(l->NumElems() == 2);
  KALDI_ASSERT((A*)l->Tail() == a2);
  KALDI_ASSERT((A*)l->Prev(a2) == a1);
  KALDI_ASSERT((A*)l->Next(a1) == a2);

  delete a1;
  KALDI_ASSERT(l->NumElems() == 1);
  KALDI_ASSERT((A*)l->Head() == (A*)l->Tail());
  KALDI_ASSERT((A*)l->Head() == a2);

  l->UnlinkNode(a2);
  KALDI_ASSERT(l->IsEmpty());
  delete a2;
  delete l;
}

} // namespace iot
} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::iot;
  UnitTestList();
  return 0;
}
