#ifndef KALDI_IOT_COMMON_H
#define KALDI_IOT_COMMON_H

#include <vector>
#include "base/kaldi-common.h"

namespace kaldi {
namespace iot {

typedef int32 WordId;
typedef int32 PhoneId;

#define kUndefinedWordId -1
#define kUndefinedPhoneId -1

#define kEpsilonWordId 0
#define kEpsilonWordString "<EPS>"
#define kEpsilonPhoneId 0
#define kEpsilonPhoneString "<eps>"

#define kSilenceWordId 1
#define kSilenceWordString "<SIL>"
#define kSilencePhoneId 1
#define kSilencePhoneString "<sil>"

template<typename T>
inline void DELETE(T& ptr) {
  if (ptr != NULL) {
    delete ptr;
  }
  ptr = NULL;
}

template <typename T>
inline void DELETE_ARRAY(T& ptr) {
  if (ptr != NULL) {
    delete [] ptr;
  }
  ptr = NULL;
}

size_t Tokenize(char *s, const char *delims, std::vector<char *> &tokens);

// ReadToken reads a token from a file,
// token memory is pre-allocated,
// length of token is returned
size_t ReadFileToken(FILE *fp, char *delims, char *token);

// PeekToken is the same as ReadToken, 
// except it doesn't consume the file stream
size_t PeekFileToken(FILE *fp, char *delims, char *token);

// ExpectToken consume a token from the file stream,
// and abort if it is different from expected token
void ExpectFileToken(FILE *fp, char *delims, char *token);

} // namespace iot
} // namespace kaldi

#endif
