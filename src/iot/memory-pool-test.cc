#include "iot/memory-pool.h"

namespace kaldi {
namespace iot {

void UnitTestMemoryPool() {
  struct S {
    char c;
    int i;
    void *p;
  };

  size_t num_elem_per_alloc = 2;
  MemoryPool *pool = new MemoryPool(sizeof(S), num_elem_per_alloc);
  KALDI_ASSERT(pool->NumUsed() == 0);
  KALDI_ASSERT(pool->NumFree() == 0);

  S *s1 = (S*)pool->MallocElem();
  KALDI_ASSERT(pool->NumUsed() == 1);
  KALDI_ASSERT(pool->NumFree() == 1);

  S *s2 = (S*)pool->MallocElem();
  KALDI_ASSERT(pool->NumUsed() == 2);
  KALDI_ASSERT(pool->NumFree() == 0);

  pool->FreeElem(s2);
  KALDI_ASSERT(pool->NumUsed() == 1);
  KALDI_ASSERT(pool->NumFree() == 1);

  S *s3 = (S*)pool->MallocElem();
  S *s4 = (S*)pool->MallocElem(); // trigger another alloc
  KALDI_ASSERT(pool->NumUsed() == 3);
  KALDI_ASSERT(pool->NumFree() == 1);

  pool->FreeElem(s1);
  pool->FreeElem(s3);
  pool->FreeElem(s4);
  delete pool;
}

} // namespace iot
} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::iot;
  UnitTestMemoryPool();
  return 0;
}
