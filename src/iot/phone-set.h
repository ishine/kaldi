#ifndef KALDI_IOT_PHONE_SET_H
#define KALDI_IOT_PHONE_SET_H

#include "base/kaldi-common.h"
#include "iot/common.h"
namespace kaldi {
namespace iot {

class PhoneSet {
public:
  PhoneSet();
  ~PhoneSet();

  void LoadFromFile(FILE *stream);

  size_t  NumPhones();
  char*   Id2Str(PhoneId id);
  PhoneId Str2Id(char *str);

private:
  std::vector<char*> phones_;
};

} // namespace iot
} // namespace kaldi

#endif
