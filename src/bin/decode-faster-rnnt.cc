// bin/decode-faster-rnnt.cc

// Copyright 2018-2019   Alibaba Inc (author: Wei Deng)

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
#include "tree/context-dep.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "nnet0/nnet-nnet.h"
#include "lm/kaldi-rnntlm.h"
#include <list>

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef nnet0::Nnet Nnet;
    typedef kaldi::Sequence Sequence;
    typedef kaldi::RNNTUtil RNNTUtil;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Decode, reading log-likelihoods (RNN transducer output)\n"
        "as matrices.  Note: you'll usually want decode-faster-ctc rather than this program.\n"
        "\n"
        "Usage:   decode-faster-rnnt [options] <lstm-language-model> <loglikes-rspecifier> <words-wspecifier>\n";
    ParseOptions po(usage);
    bool binary = true;
    BaseFloat acoustic_scale = 1.0;
    bool allow_partial = true;
    std::string word_syms_filename;
    BaseFloat beam = 5;
    int blank = 0;
    bool use_prefix = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("allow-partial", &allow_partial, "Produce output even when final state was not reached");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("beam", &beam, "Decoding beam.  Larger->slower, more accurate.");
    po.Register("blank", &blank, "RNNT bank id.");
    po.Register("use-prefix", &use_prefix, "Process prefix probability.");

    KaldiRNNTlmWrapperOpts rnntlm_opts;
    rnntlm_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
		po.PrintUsage();
		exit(1);
    }

    std::string  lstmlm_rxfilename = po.GetArg(1),
    			loglikes_rspecifier = po.GetArg(2),
				words_wspecifier = po.GetArg(3);

    Int32VectorWriter words_writer(words_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
    	word_syms = fst::SymbolTable::ReadText(word_syms_filename);
    	if (!word_syms)
    		KALDI_ERR << "Could not read symbol table from file "<<word_syms_filename;
    }

    SequentialBaseFloatMatrixReader loglikes_reader(loglikes_rspecifier);
    // Reads the language model.
	KaldiRNNTlmWrapper rnntlm(rnntlm_opts, word_syms_filename, "", lstmlm_rxfilename);

	std::list<Sequence* > *A = new std::list<Sequence*>;
	std::list<Sequence* > *B = new std::list<Sequence*>;
	Vector<BaseFloat> pred, logprob;
	Sequence *seq, *seqi, *seqj;
    std::vector<int> rd = rnntlm.GetRDim(), cd = rnntlm.GetCDim();
	LstmLmHistroy his(rd, cd, kUndefined);
    LstmLmHistroy sos_h(rd, cd, kSetZero);

    BaseFloat tot_like = 0.0, logp = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    int vocab_size, len;

    Timer timer;
    for (; !loglikes_reader.Done(); loglikes_reader.Next()) {
		std::string key = loglikes_reader.Key();
		const Matrix<BaseFloat> &loglikes (loglikes_reader.Value());

		if (loglikes.NumRows() == 0) {
			KALDI_WARN << "Zero-length utterance: " << key;
			num_fail++;
			continue;
		}

		// initialization
		for (auto &seq : *B) delete seq;
		B->clear();
		seq = new Sequence(sos_h, blank);
		B->push_back(seq);

		// decode one utterance
		int nframe = loglikes.NumRows();
		for (int n = 0; n < nframe; n++) {
			B->sort(RNNTUtil::compare_len_reverse);
			for (auto &seq : *A) delete seq;
			delete A;
			A = B;
			B = new std::list<Sequence*>;

			if (use_prefix) {
				for (auto iterj = A->begin(); iterj != A->end(); iterj++) {
                    auto iteri = iterj; iteri++;
					for (; iteri != A->end(); iteri++) {
						seqi = *iteri; seqj = *iterj;
						if (!RNNTUtil::isprefix(seqi->k, seqj->k))
							continue;

						int leni = seqi->k.size();
						int lenj = seqj->k.size();
						rnntlm.Forward(seqi->k[leni-1], seqi->lmhis, pred, his);
						logprob = pred;
						logprob.AddVec(1.0, loglikes.Row(n));
						logprob.ApplyLogSoftMax();
						BaseFloat curlogp = seqi->logp + logprob(seqj->k[leni]);
						for (int m = leni; m < lenj-1; m++) {
							logprob = seqj->pred[m];
							logprob.AddVec(1.0, loglikes.Row(n));
							logprob.ApplyLogSoftMax();
							curlogp += seqj->k[m+1];
						}
						seqj->logp = LogAdd(seqj->logp, curlogp);
					}
				}
			}

			while (true) {
				// y* = most probable in A
				Sequence *y_hat, *y_b;
				auto it = std::max_element(A->begin(), A->end(), RNNTUtil::compare_logp);
                y_hat = *it;
                A->erase(it);
				//A->remove(y_hat);

				// get rnnt lm current output and hidden state
				len = y_hat->k.size();
				rnntlm.Forward(y_hat->k[len-1], y_hat->lmhis, pred, his);

				// log probability for each rnnt output k
				logprob = pred;
				logprob.AddVec(1.0, loglikes.Row(n));
				logprob.ApplyLogSoftMax();

				vocab_size = logprob.Dim();
				for (int k = 0; k < vocab_size; k++) {
					Sequence *y_k = new Sequence(*y_hat);
					y_k->logp += logprob(k);
					if (k == blank) {
						B->push_back(y_k);
						continue;
					}
					// next t add to A
					y_k->lmhis = his;
					y_k->k.push_back(k);
					if (use_prefix) {
						y_k->pred.push_back(pred);
					}
					A->push_back(y_k);
				}

				y_hat = *std::max_element(A->begin(), A->end(), RNNTUtil::compare_logp);
				y_b = *std::max_element(B->begin(), B->end(), RNNTUtil::compare_logp);
				if (B->size() >= beam && y_b->logp >= y_hat->logp) break;
			}

			// beam width
			B->sort(RNNTUtil::compare_logp_reverse);
			// free memory
			int idx = 0;
			for (auto it = B->begin(); it != B->end(); it++) {
				if (idx >= beam) delete (*it);
				idx++;
			}
			B->resize(beam);
		}
		seq = B->front();

		if (seq != NULL) {
			logp = -seq->logp;
			std::vector<int> words = seq->k;
			words_writer.Write(key, words);
			if (word_syms != NULL) {
				std::cerr << key << ' ';
				for (size_t i = 0; i < words.size(); i++) {
					std::string s = word_syms->Find(words[i]);
					if (s == "")
						KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
					std::cerr << s << ' ';
				}
				std::cerr << '\n';
			}

			num_success++;
			frame_count += loglikes.NumRows();
			tot_like += logp;
			KALDI_LOG << "Log-like per frame for utterance " << key << " is "
					  << (logp / loglikes.NumRows()) << " over "
					  << loglikes.NumRows() << " frames.";
		} else {
			num_fail++;
			KALDI_WARN << "Did not successfully decode utterance " << key
					   << ", len = " << loglikes.NumRows();
		}
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count)
              << " over " << frame_count << " frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


