#ifndef KALDI_IOT_WORD_LOOP_H
#define KALDI_IOT_WORD_LOOP_H

#include <map>

#include "iot/graph.h"
#include "iot/vocab.h"
#include "iot/lexicon.h"
#include "iot/phone-set.h"

namespace kaldi {
namespace iot {

typedef std::pair<PhoneId, PhoneId> PhonePair;

class WordLoop {
 public:
  WordLoop()
    : vocab_(NULL), lex_(NULL), phone_set_(NULL), graph_(NULL)
  { }

  ~WordLoop() {
    vocab_ = NULL;
    lex_ = NULL;
    phone_set_ = NULL;
    DELETE(graph_);
  }

  Graph* GetGraph() { return graph_; }

  int BuildAsTriphone(Vocab* vocab, Lexicon* lexicon, PhoneSet* phone_set);

  int NumNodes() { return graph_->NumNodes(); }
  int NumArcs() { return graph_->NumArcs(); }

 private:
  int CollectPhoneStats();

  int CollectPhoneSet(int pos, std::vector<PhoneId> &collected);
  int CollectPhonePairSet(int pos1, int pos2, std::vector<PhonePair> &collected);

  int BuildYZ();
  int BuildZA();
  int BuildAB();

  int BuildHeadPhone();
  int BuildTailPhone();
  int BuildMiddlePhones();

 private:
  Vocab   *vocab_;
  Lexicon *lex_;
  PhoneSet *phone_set_;

  Graph   *graph_;

  std::vector<PhoneId> A_set_;
  std::vector<PhoneId> B_set_;
  std::vector<PhoneId> Y_set_;
  std::vector<PhoneId> Z_set_;

  std::vector<PhonePair> AB_set_;
  std::vector<PhonePair> YZ_set_;

  std::map<PhonePair, Node*> YZ_nodes_;
  std::map<PhonePair, Node*> ZA_nodes_;
  std::map<PhonePair, Node*> AB_nodes_;
  
};

} // namespace iot
} // namespace kaldi

#endif
