#ifndef KALDI_IOT_VOCAB_H
#define KALDI_IOT_VOCAB_H

#include "base/kaldi-common.h"
#include "iot/common.h"
namespace kaldi {
namespace iot {

class Vocab {
public:
  Vocab();
  ~Vocab();

  void LoadFromFile(FILE *stream);

  size_t  NumWords();
  char*   Id2Str(WordId id);
  WordId  Str2Id(char *word_str);

private:
  std::vector<char*> words_;
};

} // namespace iot
} // namespace kaldi

#endif
