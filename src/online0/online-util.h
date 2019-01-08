// online0/online-util.h

// Copyright 2012 Cisco Systems (author: Matthias Paulik)

//   Modifications to the original contribution by Cisco Systems made by:
//   Vassil Panayotov

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_ONLINE0_ONLINE_UTIL_H_
#define KALDI_ONLINE0_ONLINE_UTIL_H_

#include <deque>

#include "base/kaldi-common.h"
#include "fstext/fstext-lib.h"
#include "util/kaldi-mutex.h"
#include "util/kaldi-semaphore.h"

// This file hosts the declarations of various auxiliary functions, used by
// the binaries in "onlinebin" directory. These functions are not part of the
// core online decoding infrastructure, but rather artefacts of the particular
// implementation of the binaries.

namespace kaldi {

// Reads a decoding graph from a file
fst::Fst<fst::StdArc> *ReadDecodeGraph(std::string filename);

// Prints a string corresponding to (a possibly partial) decode result as
// and adds a "new line" character if "line_break" argument is true
void PrintPartialResult(const std::vector<int32>& words,
                        const fst::SymbolTable *word_syms,
                        bool line_break);


/** This struct stores neural net training examples to be used in
    multi-threaded training.  */
class Repository {
 public:
  /// The following function is called by the code that reads in the examples.
  void Accept(void *example);

  /// The following function is called by the code that reads in the examples,
  /// when we're done reading examples; it signals this way to this class
  /// that the stream is now empty
  void Done();

  /// This function is called by the code that does the training.  If there is
  /// an example available it will provide it, or it will sleep till one is
  /// available.  It returns NULL when there are no examples left and
  /// ExamplesDone() has been called.
  void *Provide();

  int Size();

  Repository(int32 buffer_size = 128): buffer_size_(buffer_size),
                                      empty_semaphore_(buffer_size_),
                                      done_(false) { }
 private:
  int32 buffer_size_;
  Semaphore full_semaphore_;
  Semaphore empty_semaphore_;
  Mutex examples_mutex_; // mutex we lock to modify examples_.

  std::deque<void*> examples_;
  bool done_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(Repository);
};

} // namespace kaldi

#endif // KALDI_ONLINE0_ONLINE_UTIL_H_
