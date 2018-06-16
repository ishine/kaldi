#include <cstdio>
#include "iot/word-loop.h"

namespace kaldi {
namespace iot {

void UnitTestWordLoop() {
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

  WordLoop *word_loop = new WordLoop();
  word_loop->BuildAsTriphone(vocab, lex, phone_set);

  FILE *fp = fopen("graph.dot", "w+");
  word_loop->GetGraph()->MarkNodeId();
  word_loop->GetGraph()->WriteDot(fp);
  fprintf(stderr, "nodes: %d arcs: %d\n", word_loop->NumNodes(), word_loop->NumArcs());

  fclose(fp);

  DELETE(word_loop);
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
  UnitTestWordLoop();
  return 0;
}
