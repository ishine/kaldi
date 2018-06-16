#include <cstdio>
#include "iot/lexicon.h"

namespace kaldi {
namespace iot {

void UnitTestLexicon() {
  // load phone set
  PhoneSet *phone_set = new PhoneSet();
  FILE *phone_set_stream = fopen("test_data/phones.txt", "r");
  phone_set->LoadFromFile(phone_set_stream);
  fclose(phone_set_stream);

  // load base lexicon
  BaseLexicon *base_lex = new BaseLexicon();
  FILE *lexicon_stream = fopen("test_data/lexicon.txt", "r");
  base_lex->LoadFromFile(lexicon_stream, phone_set);
  fclose(lexicon_stream);

  // load vocabulary
  Vocab *vocab = new Vocab();
  FILE *vocab_stream = fopen("test_data/words.txt", "r");
  vocab->LoadFromFile(vocab_stream);
  fclose(vocab_stream);

  // construct lexicon via (vocab & base_lexicon)
  Lexicon *lex = new Lexicon();
  lex->Extract(base_lex, vocab);

  fprintf(stdout, "phone set: %lu phones\n", phone_set->NumPhones());
  fprintf(stdout, "base lexicon: %lu words\n", base_lex->NumWords());
  fprintf(stdout, "lexicon: %lu words\n", lex->NumWords());

  for (size_t i = 0; i < lex->NumWords(); i++) {
    LexWord *word = lex->Word(i);
    for (size_t j = 0; j < word->NumProns(); j++) {
      LexPron *pron = word->Pron(j);
      fprintf(stdout, "%s\t", word->Str());
      for (size_t k = 0; k < pron->NumPhones(); k++) {
        fprintf(stdout, "%s(%d) ", phone_set->Id2Str(pron->Phone(k)), pron->Phone(k));
      }
      fprintf(stdout, "\n");
    }
  }

  DELETE(lex);
  DELETE(vocab);
  DELETE(base_lex);
  DELETE(phone_set);
}

} // namespace iot
} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::iot;
  UnitTestLexicon();
  return 0;
}
