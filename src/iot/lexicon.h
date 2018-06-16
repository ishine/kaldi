#ifndef KALDI_IOT_LEXICON_H
#define KALDI_IOT_LEXICON_H

#include "iot/common.h"
#include "iot/phone-set.h"
#include "iot/vocab.h"

namespace kaldi {
namespace iot {

typedef PhoneId LexPhone;

class LexPron {
 friend class BaseLexicon;
 public:
  LexPron() :
    phones_(NULL),
    num_phones_(0)
  { }

  ~LexPron() { }

  LexPhone  Phone(size_t i) { return phones_[i]; }
  size_t    NumPhones() { return num_phones_; }

 private:
  LexPhone *phones_;
  size_t    num_phones_;
};

class LexWord {
 friend class BaseLexicon;
 public:
  LexWord() :
    str_(NULL),
    prons_(NULL),
    num_prons_(0)
  { }

  ~LexWord() { if (str_ != NULL ) free(str_); }

  char*    Str() { return str_; }
  LexPron* Pron(size_t i) { return &prons_[i]; }
  size_t   NumProns() { return num_prons_; }

 private:
  char    *str_;
  LexPron *prons_;
  size_t   num_prons_;
};

class BaseLexicon {
 public:
  BaseLexicon();
  ~BaseLexicon();

  void LoadFromFile(FILE *lexicon_stream, PhoneSet *phone_set);

  LexWord* Word(WordId i);
  size_t   NumWords();

 private:
  LexWord  *words_;
  size_t    num_words_;

  LexPron  *prons_;
  size_t    num_prons_;

  LexPhone *phones_;
  size_t    num_phones_;

  PhoneSet *phone_set_;
};

class Lexicon {
 public:
  Lexicon()
    : base_lexicon_(NULL) 
  { }

  ~Lexicon() { base_lexicon_ = NULL; }

  void     Extract(BaseLexicon *base_lexicon, Vocab *vocab);
  LexWord* Word(WordId i) { return base_lexicon_->Word(reindex_[i]); }
  size_t   NumWords() { return reindex_.size(); }

 private:
  std::vector<WordId> reindex_;
  BaseLexicon *base_lexicon_;
};

} // namespace iot
} // namespace kaldi

#endif
