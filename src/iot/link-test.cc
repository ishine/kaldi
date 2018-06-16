#include "iot/link.h"

namespace kaldi {
namespace iot {

void UnitTestLink() {
  Link *l1 = new Link;
  Link *l2 = new Link;
  Link *l3 = new Link;

  KALDI_ASSERT(!l1->IsLinked());
  KALDI_ASSERT(!l2->IsLinked());
  KALDI_ASSERT(!l3->IsLinked());
  
  l1->InsertBefore(l2);
  KALDI_ASSERT(l2->Prev() == l1);

  l3->InsertAfter(l2);
  KALDI_ASSERT(l2->Next() == l3);

  KALDI_ASSERT(l1->IsLinked());
  KALDI_ASSERT(l2->IsLinked());
  KALDI_ASSERT(l3->IsLinked());

  delete l2;  // automatic destructive unlinking
  KALDI_ASSERT(l1->Next() == l3);
  KALDI_ASSERT(l3->Prev() == l1);

  l3->Unlink();
  KALDI_ASSERT(!l3->IsLinked());
  delete l3;

  KALDI_ASSERT(!l1->IsLinked());
  KALDI_ASSERT(l1->Prev() == l1);
  KALDI_ASSERT(l1->Next() == l1);
  delete l1;
}

} // namespace iot
} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::iot;
  UnitTestLink();
  return 0;
}
