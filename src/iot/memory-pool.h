#ifndef KALDI_IOT_MEMORY_POOL_H
#define KALDI_IOT_MEMORY_POOL_H

#include "base/kaldi-common.h"

namespace kaldi {
namespace iot {

class MemoryPool {
public:
  MemoryPool(size_t elem_bytes, size_t num_elem_per_alloc);
  ~MemoryPool();

  void* MallocElem();
  void  FreeElem(void* p);
  size_t NumUsed();
  size_t NumFree();

private:
  struct Elem {
    Elem *next;

    Elem() { next = NULL; }
    ~Elem() { next = NULL; }
  };

  struct FreeList {
    Elem *head;

    FreeList() :
      head(NULL)
    { }

    bool IsEmpty() {
      return (head == NULL);
    }

    void Push(Elem *p) {
      p->next = head;
      head = p;
    }

    Elem* Pop() {
      Elem *p = head;
      if (!IsEmpty()) {
        head = head->next;
      }
      return p;
    }
  };

  size_t elem_bytes_;
  size_t num_elem_per_alloc_;
  size_t num_used_;
  std::vector<char*> allocs_;
  FreeList free_list_;
};

} // namespace iot
} // namespace kaldi
#endif
