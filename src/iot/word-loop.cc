#include "iot/word-loop.h"

namespace kaldi {
namespace iot {

int WordLoop::BuildAsTriphone(Vocab* vocab, Lexicon* lex, PhoneSet* phone_set) {
  DELETE(graph_);
  graph_ = new Graph();

  vocab_ = vocab;
  lex_ = lex;
  phone_set_ = phone_set;

  CollectPhoneStats();

  BuildAB();
  BuildYZ();
  BuildZA();

  BuildTailPhone();
  BuildHeadPhone();
  BuildMiddlePhones();

  return 0;
}

int WordLoop::CollectPhoneStats() {
  // TODO: do we need 1-pass lexicon sweep for efficiency?
  CollectPhoneSet(0, A_set_);
  CollectPhoneSet(1, B_set_);
  CollectPhoneSet(-2, Y_set_);
  CollectPhoneSet(-1, Z_set_);
  CollectPhonePairSet(0, 1, AB_set_);
  CollectPhonePairSet(-2, -1, YZ_set_);
  return 0;
}

int WordLoop::CollectPhoneSet(int pos, std::vector<PhoneId> &collected) {
  KALDI_LOG << "Collecting phone set by position " << pos;
  std::vector<bool> marker(phone_set_->NumPhones(), false);
  for (size_t i = 0; i != vocab_->NumWords(); i++) {
    if (i == kEpsilonWordId || i == kSilenceWordId) { // special words
      continue;
    }
    LexWord *word = lex_->Word(i);
    KALDI_ASSERT(word != NULL);
    for (size_t j = 0; j != word->NumProns(); j++) {
      LexPron *pron = word->Pron(j);
      KALDI_ASSERT(pron != NULL);
      KALDI_ASSERT(pron->NumPhones() >= 2); //TODO: support singleton phone pron later
      PhoneId p = ((pos>=0) ? pron->Phone(pos) : pron->Phone(pron->NumPhones() + pos));
      marker[p] = true;
    }
  }

  collected.clear();
  for (PhoneId p = 0; p != phone_set_->NumPhones(); p++) {
    if (marker[p]) {
      collected.push_back(p);
    }
  }

  bool debug = true;
  if (debug) {
    fprintf(stderr, "PHONE-SET BY POS[%d]:", pos);
    for (size_t i = 0; i != collected.size(); i++) {
      fprintf(stderr, " %s(%d)", phone_set_->Id2Str(collected[i]), collected[i]);
    }
    fprintf(stderr, "\n");
  }

  return 0;
}

int WordLoop::CollectPhonePairSet(int pos1, int pos2, std::vector<PhonePair> &collected) {
  KALDI_LOG << "Collecting phone pair set by position " << pos1 << "," << pos2;
  int P = phone_set_->NumPhones();
  std::vector<bool> marker(P * P, false);
  for (size_t i = 0; i != vocab_->NumWords(); i++) {
    if (i == kEpsilonWordId || i == kSilenceWordId) { // special words
      continue;
    }
    LexWord *word = lex_->Word(i);
    KALDI_ASSERT(word != NULL);
    for (size_t j = 0; j != word->NumProns(); j++) {
      LexPron *pron = word->Pron(j);
      KALDI_ASSERT(pron != NULL);
      KALDI_ASSERT(pron->NumPhones() >= 2); //TODO: support singleton phone pron later
      PhoneId p1 = ((pos1>=0) ? pron->Phone(pos1) : pron->Phone(pron->NumPhones() + pos1));
      PhoneId p2 = ((pos2>=0) ? pron->Phone(pos2) : pron->Phone(pron->NumPhones() + pos2));
      marker[p1 * P + p2] = true;
    }
  }

  collected.clear();
  for (PhoneId p1 = 0; p1 != P; p1++) {
    for (PhoneId p2 = 0; p2 != P; p2++) {
      if (marker[p1*P + p2]) {
        collected.push_back(PhonePair(p1, p2));
      }
    }
  }

  bool debug = true;
  if (debug) {
    fprintf(stderr, "PHONE-PAIR-SET BY POS1[%d],POS2[%d]:", pos1, pos2);
    for (size_t i = 0; i != collected.size(); i++) {
      fprintf(stderr, " {%s(%d),%s(%d)}", 
        phone_set_->Id2Str(collected[i].first), collected[i].first,
        phone_set_->Id2Str(collected[i].second), collected[i].second
        );
    }
    fprintf(stderr, "\n");
  }

  return 0;
}

int WordLoop::BuildAB() {
  for (size_t i = 0; i != AB_set_.size(); i++) {
    PhonePair ab = AB_set_[i];
    Node *node = graph_->AddNode();
    AB_nodes_[ab] = node;

    std::string a(phone_set_->Id2Str(ab.first));
    std::string b(phone_set_->Id2Str(ab.second));
    node->name = a + "," + b;
  }
  return 0;
}

int WordLoop::BuildYZ() {
  for (size_t i = 0; i != YZ_set_.size(); i++) {
    PhonePair yz = YZ_set_[i];
    Node *node = graph_->AddNode();
    YZ_nodes_[yz] = node;

    std::string y(phone_set_->Id2Str(yz.first));
    std::string z(phone_set_->Id2Str(yz.second));
    node->name = y + "," + z;
  }
  return 0;
}

int WordLoop::BuildZA() {
  for (size_t i = 0; i != Z_set_.size(); i++) {
    for (size_t j = 0; j != A_set_.size(); j++) {
      PhonePair za(Z_set_[i], A_set_[j]);
      Node *node = graph_->AddNode();
      ZA_nodes_[za] = node;

      std::string z(phone_set_->Id2Str(za.first));
      std::string a(phone_set_->Id2Str(za.second));
      node->name = z + "," + a;
    }
  }
  return 0;
}

int WordLoop::BuildHeadPhone() {
  // Z-A+B
  for (int ai = 0; ai != A_set_.size(); ai++) {
    PhoneId a = A_set_[ai];
    for (int zi = 0; zi != Z_set_.size(); zi++) {
      PhoneId z = Z_set_[zi];
      Node *za = ZA_nodes_[PhonePair(z,a)];
      for (int i = 0; i != AB_set_.size(); i++) {
        PhoneId a_hat = AB_set_[i].first;
        PhoneId b = AB_set_[i].second;
        if (a_hat == a) {
          Node *ab = AB_nodes_[PhonePair(a,b)];
          Arc *arc = graph_->AddArc(za, ab);
          arc->name = std::string(phone_set_->Id2Str(z)) + "-" + 
                      std::string(phone_set_->Id2Str(a)) + "+" +
                      std::string(phone_set_->Id2Str(b));
        }
      }
    }
  }
  return 0;
}

int WordLoop::BuildTailPhone() {
  // Y-Z+A
  for (int zi = 0; zi != Z_set_.size(); zi++) {
    PhoneId z = Z_set_[zi];
    for (int ai = 0; ai != A_set_.size(); ai++) {
      PhoneId a = A_set_[ai];
      Node *za = ZA_nodes_[PhonePair(z,a)];
      for (int i = 0; i != YZ_set_.size(); i++) {
        PhoneId y = YZ_set_[i].first;
        PhoneId z_hat = YZ_set_[i].second;
        if (z_hat == z) {
          Node *yz = YZ_nodes_[PhonePair(y,z)];
          Arc *arc = graph_->AddArc(yz, za);
          arc->name = std::string(phone_set_->Id2Str(y)) + "-" + 
                      std::string(phone_set_->Id2Str(z)) + "+" +
                      std::string(phone_set_->Id2Str(a));         
        }
      }
    }
  }
  return 0;
}

int WordLoop::BuildMiddlePhones() {
  for (size_t i = 0; i != vocab_->NumWords(); i++) {
    if (i == kEpsilonWordId || i == kSilenceWordId) {
      continue;
    }
    LexWord *word = lex_->Word(i);
    KALDI_ASSERT(word != NULL);
    for (size_t j = 0; j != word->NumProns(); j++) {
      LexPron *pron = word->Pron(j);
      KALDI_ASSERT(pron != NULL);
      KALDI_ASSERT(pron->NumPhones() >= 2); //TODO: support singleton phone pron later
      PhoneId a = pron->Phone(0);
      PhoneId b = pron->Phone(1);
      PhoneId y = pron->Phone(pron->NumPhones() - 2);
      PhoneId z = pron->Phone(pron->NumPhones() - 1);
      PhonePair ab(a,b);
      PhonePair yz(y,z);
      Node *src = AB_nodes_[ab];
      Node *dst = NULL;
      for (size_t k = 1; k != pron->NumPhones()-1; k++) {
        dst = graph_->AddNode();
        Arc *arc = graph_->AddArc(src, dst);
        arc->name = std::string(phone_set_->Id2Str(pron->Phone(k-1))) + "-" + 
                    std::string(phone_set_->Id2Str(pron->Phone(k))) + "+" +
                    std::string(phone_set_->Id2Str(pron->Phone(k+1)));
        src = dst;
      }
      dst = YZ_nodes_[yz];
      Arc *we_arc = graph_->AddArc(src, dst);
      we_arc->olabel = (WordId)i;
      we_arc->name = std::string(vocab_->Id2Str(i));
    }
  }
  return 0;
}


} // namespace iot
} // namespace kaldi
