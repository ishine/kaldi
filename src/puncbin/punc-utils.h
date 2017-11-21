#ifndef KALDI_PUNC_BIN_PUNC_UTILS_H_
#define KALDI_PUNC_BIN_PUNC_UTILS_H_

#include "nnet/nnet-nnet.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"

using namespace std;
using namespace kaldi;
using namespace kaldi::nnet1;
typedef kaldi::int32 int32;

std::map<std::string, size_t> BuildVocab(std::ifstream &vocab_file) {
    std::map<std::string, size_t> word_to_id;
    std::string word;
    int i = 0;
    while (vocab_file >> word) {
        word_to_id[word] = i;
        i++;
    }
    return word_to_id;
}

std::map<size_t, std::string> BuildReverseVocab(std::ifstream &vocab_file) {
    std::map<size_t, std::string> id_to_word;
    std::string word;
    size_t i = 0;
    while (getline(vocab_file, word)) {
        id_to_word[i] = word;
        i++;
    }
    return id_to_word;
}

vector<size_t> Transform(const string &words, const map<string, size_t> &vocab) {
    vector<size_t> ids;
    istringstream stream(words);
    string word;
    size_t id;

    while (stream >> word) {
        auto map_it = vocab.find(word);
        if (map_it == vocab.end()) {
            map_it = vocab.find("<unk>");
            if (map_it == vocab.end()) {
                std::cout << "Your Vocab Should include <unk>" << std::endl;
                exit(0);
            }
        }
        id = map_it->second;
        ids.push_back(id);
    }
    return ids;
}

void PrintVec(const vector<size_t> & ids) {
    for (int i = 0; i < ids.size(); ++i) {
        cout << ids[i] << " ";
    }
    cout << endl;
}

template<typename Real>
void IdsToMatrix(const std::vector<size_t>& ids, Matrix<Real>* m) {
  std::string s("[ ");
  int i;
  for (i = 0; i <ids.size()-1; ++i) {
      s += to_string(ids[i]) + " \n";
  }
  s += to_string(ids[i]) + " ]";
  // std::cout << s << endl;

  std::istringstream is(s + "\n");
  m->Read(is, false);  // false for ascii
}

template<typename Real>
void ProbToId(const Matrix<Real> &m, vector<size_t> &ids) { 
    // m: T x C
    // ids: T
    for (int r = 0; r < m.NumRows(); ++r) {
        Real max = 0.0;
        int32 max_id = -1;
        for (int c = 0; c < m.NumCols(); ++c) {
            if (m(r, c) > max) {
                max = m(r, c);
                max_id = c;
            }
        }
        ids.push_back(max_id);
    }
}

void AddPuncToTxt(const std::string &txt_line, 
                  const std::vector<size_t> &punc_ids, 
                  const std::map<size_t, std::string> &id_to_punc,
                  std::string &txt_line_with_punc) {
  istringstream stream(txt_line);
  std::string word, punc;
  size_t i = 0;
  while (stream >> word) {
    punc = id_to_punc.find(punc_ids[i])->second;
    if (punc == " ") {
      txt_line_with_punc += word + " ";
    } else {
      txt_line_with_punc += punc + " " + word + " ";
    }
    ++i;
  }
  punc = id_to_punc.find(punc_ids[i])->second;
  txt_line_with_punc += punc;
}

#endif  // KALDI_PUNC_BIN_PUNC_UTILS_H_
