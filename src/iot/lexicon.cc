#include "iot/lexicon.h"

namespace kaldi {
namespace iot {

BaseLexicon::BaseLexicon() :
  words_(NULL),
  num_words_(0),
  prons_(NULL),
  num_prons_(0),
  phones_(NULL),
  num_phones_(0),
  phone_set_(NULL)
{ }

BaseLexicon::~BaseLexicon() {
  DELETE_ARRAY(words_);
  DELETE_ARRAY(prons_);
  DELETE_ARRAY(phones_);
  phone_set_ = NULL;
}

void BaseLexicon::LoadFromFile(FILE *lexicon_stream, PhoneSet *phone_set) {
  KALDI_ASSERT(lexicon_stream != NULL);
  phone_set_ = phone_set;

  char line[2048];
  std::vector<char*> toks;

  char cur_word[512];
  strncpy(cur_word, "", 512);
  while (fgets(line, sizeof(line), lexicon_stream) != NULL) {
    if (Tokenize(line, " \t\n", toks) != 0) {
      num_prons_++;
      num_phones_ += (toks.size()-1);

      if (strcmp(cur_word, toks[0]) != 0) {
        num_words_++;
        strncpy(cur_word, toks[0], 512);
      }
    }
  }

  words_  = new LexWord[num_words_];
  prons_  = new LexPron[num_prons_];
  phones_ = new LexPhone[num_phones_];

  rewind(lexicon_stream);
  LexWord  *word = &words_[0];
  LexPron  *pron = &prons_[0];
  LexPhone *phone = &phones_[0];

  PeekFileToken(lexicon_stream, " \t\n", cur_word);
  word->str_ = strdup(cur_word);
  word->prons_ = pron;
  word->num_prons_ = 0;
  // invariants: words_[0, word), prons_[0, pron), phones_[0, phone) loaded
  while (fgets(line, sizeof(line), lexicon_stream) != NULL) {
    if (Tokenize(line, " \t\n", toks) != 0) {
      if (strcmp(word->str_, toks[0]) != 0) {
        word++;
        word->str_ = strdup(toks[0]);
        word->prons_ = pron;
        word->num_prons_ = 0;
      }

      pron->phones_ = phone;
      pron->num_phones_ = 0;
      for (size_t i = 1; i != toks.size(); i++) {
        *phone = (LexPhone)phone_set_->Str2Id(toks[i]);
        phone++;
        pron->num_phones_++;
      }
      pron++;
      word->num_prons_++;
    }
  }
}

LexWord* BaseLexicon::Word(WordId i) {
  if (i >= 0 && i < num_words_) {
    return &words_[i];
  } else {
    return NULL;
  }
}

size_t BaseLexicon::NumWords() {
  return num_words_;
}

/* lexicon */
// TODO:
//   bsearch for word look-up  
void Lexicon::Extract(BaseLexicon *base_lexicon, Vocab *vocab) {
  base_lexicon_ = base_lexicon;

  reindex_.resize(vocab->NumWords(), kUndefinedWordId);
  for (WordId i = 0; i != vocab->NumWords(); i++) {
    for (WordId j = 0; j != base_lexicon_->NumWords(); j++) {
      if (strcmp(vocab->Id2Str(i), base_lexicon_->Word(j)->Str()) == 0) {
        reindex_[i] = j;
      }
    }
    if (reindex_[i] == kUndefinedWordId) {
      KALDI_WARN << "No lexicon entry for word: " << vocab->Id2Str(i);
    }
  }
}

} // namespace iot
} // namespace kaldi
