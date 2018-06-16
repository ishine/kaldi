#include <cstdio>
#include "iot/phone-set.h"

namespace kaldi {
namespace iot {

void UnitTestPhoneSet() {
  PhoneSet *phone_set = new PhoneSet();
  FILE *stream = fopen("test_data/phones.txt", "r");
  phone_set->LoadFromFile(stream);
  fclose(stream);

  fprintf(stdout, "num_phones: %lu\n", phone_set->NumPhones());

  for (PhoneId i = 0; i < phone_set->NumPhones(); i++) {
    char *str = phone_set->Id2Str(i);
    PhoneId k = phone_set->Str2Id(str);
    KALDI_ASSERT(k == i);
    fprintf(stderr, "id:%d str:%s\n", i, str);
  }
  DELETE(phone_set);
}

} // namespace iot
} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::iot;
  UnitTestPhoneSet();
  return 0;
}
