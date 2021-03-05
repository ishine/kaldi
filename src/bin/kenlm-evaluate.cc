// bin/kenlm-evaluate.cc

// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include <iomanip>

#if HAVE_KENLM == 1
#include "lm/model.hh"
		typedef lm::ngram::Model KenModel;
		typedef lm::ngram::State KenState;
		typedef lm::ngram::Vocabulary KenVocab;
#endif


void split(std::string &str, const std::string &delim, std::vector<std::string> &strs) {
    size_t start = 0, end = str.find(delim);
    strs.clear();
    while (end != std::string::npos) {
        strs.push_back(str.substr(start, end-start));
        start = end + delim.length();
        end = str.find(delim, start);
    }
    strs.push_back(str.substr(start, end-start));
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Evaluate sentence ppl with kenlm \n"
        "Usage: kenlm-evaluate [options] <kenlm>\n"
        "e.g.: kenlm-evaluate -ppl test.txt lm.arpa.kenbin \n";

    ParseOptions po(usage);
    
    std::string test_filename;
    po.Register("ppl", &test_filename, "text file to compute perplexity from.");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }


    std::string const_arpa_filename = po.GetArg(1);
    std::string line, word, pword, pstr, sos = "<s>", eos = "</s>";

    float word_logp, logp10, logpe, prob, logp, logp_sum;
    int32 num_sen = 0, size, index, num_words = 0;

    KenModel *kenlm_arpa = NULL;
    KenState    state, nstate;

    kenlm_arpa = new KenModel(const_arpa_filename.c_str());
    const KenVocab *kenlm_vocab = &(kenlm_arpa->GetVocabulary());
	std::vector<std::string> strs;

    std::ifstream ifs(test_filename);
    std::ostringstream ostr;
    while(getline(ifs, line)) {
        std::cout << line << std::endl;
        split(line, " ", strs);
        size = strs.size();
        state = kenlm_arpa->BeginSentenceState();
        ostr.clear();
        pword = sos; 
        logp = 0;

        for (int i = 0; i <= size; i++) {
            word = i<size ? strs[i] : eos;
            word.erase(std::remove(word.begin(), word.end(), ' '), word.end());
            index = kenlm_vocab->Index(word);
            word_logp = kenlm_arpa->Score(state, index, nstate);

            prob = std::pow(10, word_logp);
            logp10 = word_logp;
            logpe = word_logp * M_LN10;
            logp += logp10;
            pstr = i==0 ? pword+"   " : pword+" ...";

            pword = word;
            state = nstate;
            if (kaldi::g_kaldi_verbose_level >= 1)
                ostr << "\tp( " << word << " | " << pstr << " )\t" << "= " << prob << " [" << logp10 << "] [" << logpe << "]\n";
        }

        if (kaldi::g_kaldi_verbose_level >= 1) {
            std::cout << ostr.str();
            ostr.str("");
        }
        std::cout << "logprob= " << logp << " ppl= " << std::pow(10, -logp/(size+1)) << " ppl1= " << std::pow(10, -logp/size) << std::endl;
        num_sen++;
        num_words += size;
        logp_sum += logp;
    }

    KALDI_LOG << "file " << test_filename << ": " << num_sen << " sentences, " << num_words << " words";
    KALDI_LOG << "logprob= " << logp_sum << " ppl= " << std::pow(10, -logp_sum/(num_words+num_sen)) << " ppl1= " << std::pow(10, -logp_sum/num_words);
    return (num_sen != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


