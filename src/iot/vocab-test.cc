#include <cstdio>
#include "iot/vocab.h"

namespace kaldi {
namespace iot {

void UnitTestVocab() {
  Vocab *vocab = new Vocab();
  FILE *stream = fopen("test_data/words.txt", "r");
  vocab->LoadFromFile(stream);
  fclose(stream);

  fprintf(stdout, "num_words: %lu\n", vocab->NumWords());

  for (WordId i = 0; i < vocab->NumWords(); i++) {
    char *str = vocab->Id2Str(i);
    WordId k = vocab->Str2Id(str);
    KALDI_ASSERT(k == i);
    fprintf(stderr, "id:%d str:%s\n", i, str);
  }
  DELETE(vocab);
}

} // namespace iot
} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::iot;
  UnitTestVocab();
  return 0;
}
