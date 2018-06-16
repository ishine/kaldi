#include "iot/common.h"

namespace kaldi {
namespace iot {

size_t Tokenize(char *s, const char *delims, std::vector<char *> &tokens) {
  tokens.resize(0);
  for (char *tok = strtok(s, delims); tok != NULL; tok = strtok(NULL, delims)) {
    tokens.push_back(tok);
  }
  return (size_t)tokens.size();
}


size_t ReadFileToken(FILE *fp, char *delims, char *token)
{
    size_t len = 0;
    int ch;
    while((ch = fgetc(fp)) != EOF) {
        if (strchr(delims, ch)) {
            if (len != 0) break;
        } else {
            token[len++] = ch;
        }
    }
    token[len] = '\0';
    return len;
}


size_t PeekFileToken(FILE *fp, char *delims, char *token)
{
    size_t len;
    fpos_t pos;
    fgetpos(fp, &pos);
    len = ReadFileToken(fp, delims, token);
    fsetpos(fp, &pos);
    return len;
}


void ExpectFileToken(FILE *fp, char *delims, char *expect)
{
    char tok[256];
    int n = 0;
    n = ReadFileToken(fp, delims, tok);
    assert(n < 256);
    assert(strcmp(tok, expect) == 0);
}

} // namespace iot
} // namespace kaldi
