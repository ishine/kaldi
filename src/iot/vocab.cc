#include "iot/vocab.h"

namespace kaldi {
namespace iot {

Vocab::Vocab() {
  ;
}

Vocab::~Vocab() {
  for (size_t i = 0; i < words_.size(); i++) {
    KALDI_ASSERT(words_[i] != NULL);
    free(words_[i]);
  }
}

// format: "word_str word_id" 
void Vocab::LoadFromFile(FILE *stream) {
  KALDI_ASSERT(stream != NULL);
  char line[256];
  std::vector<char*> toks;
  while (fgets(line, sizeof(line), stream) != NULL) {
    if (Tokenize(line, " \t\n", toks) == 2) {
      KALDI_ASSERT(atoi(toks[1]) == (int)words_.size());
      words_.push_back(strdup(toks[0]));
    }
  }
}

size_t Vocab::NumWords() {
  return words_.size();
}

char* Vocab::Id2Str(WordId i) {
  KALDI_ASSERT(i >= 0 && i < words_.size());
  return words_[i];
}

// TODO: optimization
WordId Vocab::Str2Id(char* word) {
  for (WordId i = 0; i != words_.size(); i++) {
    if (strcmp(word, words_[i]) == 0) {
      return i;
    }
  }
  KALDI_WARN << "Unknown word " << word;
  return kUndefinedWordId;
}

} // namespace iot
} // namespace kaldi
