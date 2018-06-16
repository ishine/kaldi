#include "iot/memory-pool.h"

namespace kaldi {
namespace iot {

MemoryPool::MemoryPool(size_t elem_bytes, size_t num_elem_per_alloc) {
  assert(elem_bytes >= sizeof(Elem*) && "element size should be larger than a pointer");
  elem_bytes_ = elem_bytes;
  num_elem_per_alloc_ = num_elem_per_alloc;
  num_used_ = 0;
}

MemoryPool::~MemoryPool() {
  for(int i = 0; i < allocs_.size(); i++) {
    delete[] allocs_[i];
  }
}

void* MemoryPool::MallocElem() {
  if (free_list_.IsEmpty()) {
    char *p = new char[elem_bytes_ * num_elem_per_alloc_];
    assert(p != NULL && "memory pool allocation failed.");
    allocs_.push_back(p);
    for (int i = 0; i < num_elem_per_alloc_; i++) {
      free_list_.Push((Elem*)(p + i * elem_bytes_));
    }
  }
  num_used_++;
  return free_list_.Pop();
}

void MemoryPool::FreeElem(void *p) {
  num_used_--;
  free_list_.Push((Elem*)p);
}

size_t MemoryPool::NumUsed() {
  return num_used_;
}

size_t MemoryPool::NumFree() {
  size_t n = 0;
  for (Elem *p = free_list_.head; p != NULL; p = p->next) {
    n++;
  }
  return n;
}

} // namespace iot
} // namespace kaldi
