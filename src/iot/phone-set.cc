#include "iot/phone-set.h"

namespace kaldi {
namespace iot {

PhoneSet::PhoneSet() {
  ;
}

PhoneSet::~PhoneSet() {
  for (size_t i = 0; i < phones_.size(); i++) {
    KALDI_ASSERT(phones_[i] != NULL);
    free(phones_[i]);
  }
}

// format: "phone_str phone_id" 
void PhoneSet::LoadFromFile(FILE *stream) {
  KALDI_ASSERT(stream != NULL);
  char line[256];
  std::vector<char*> toks;
  while (fgets(line, sizeof(line), stream) != NULL) {
    if (Tokenize(line, " \t\n", toks) == 2) {
      KALDI_ASSERT(atoi(toks[1]) == (int)phones_.size());
      phones_.push_back(strdup(toks[0]));
    }
  }
}

size_t PhoneSet::NumPhones() {
  return phones_.size();
}

char* PhoneSet::Id2Str(PhoneId i) {
  return phones_[i];
}

// TODO: optimization
PhoneId PhoneSet::Str2Id(char* phone) {
  for (PhoneId i = 0; i != phones_.size(); i++) {
    if (strcmp(phone, phones_[i]) == 0) {
      return i;
    }
  }
  KALDI_WARN << "Unknown phone " << phone;
  return kUndefinedPhoneId;
}

} // namespace iot
} // namespace kaldi
