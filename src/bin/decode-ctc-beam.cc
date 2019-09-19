// bin/decode-ctc-beam.cc

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

#include <list>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "decoder/ctc-decoder.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::SymbolTable;

    const char *usage =
        "Decode, reading log-likelihoods (CTC acoustic model output) as matrices.\n"
        "Note: you'll usually want decode-faster-ctc rather than this program.\n"
        "\n"
        "Usage:   decode-ctc-beam [options] <lstm-language-model> <loglikes-rspecifier> <words-wspecifier>\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string search = "beam";
    float blank_posterior_scale = -1.0;
    float en_penalty = 0.0;
    std::string word_syms_filename;
    std::string const_arpa_filename;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("search", &search, "search function(beam|greedy)");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("blank-posterior-scale", &blank_posterior_scale, "For CTC decoding, "
    		"scale blank acoustic posterior by a constant value(e.g. 0.01), other label posteriors are directly used in decoding.");
    po.Register("en-penalty", &en_penalty, "For CTC decoding, "
    		"scale blank acoustic posterior by a constant value(e.g. 0.01), other label posteriors are directly used in decoding.");
    po.Register("const-arpa", &const_arpa_filename, "Fusion using const ngram arpa language model (optional).");

    KaldiLstmlmWrapperOpts lstmlm_opts;
    CTCDecoderOptions decoder_opts;
    lstmlm_opts.Register(&po);
    decoder_opts.Register(&po);

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
	KaldiLstmlmWrapper lstmlm(lstmlm_opts, word_syms_filename, "", lstmlm_rxfilename);
	// Reads the language model in ConstArpaLm format.
	ConstArpaLm const_arpa;
	if (const_arpa_filename != "" && decoder_opts.rnnlm_scale < 1.0) {
		ReadKaldiObject(const_arpa_filename, &const_arpa);
	}

	// decoder
	CTCDecoder decoder(decoder_opts, lstmlm, const_arpa);

    BaseFloat tot_like = 0.0, logp = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    std::vector<int> words;

    Timer timer;
    for (; !loglikes_reader.Done(); loglikes_reader.Next()) {
		std::string key = loglikes_reader.Key();
		Matrix<BaseFloat> &loglikes (loglikes_reader.Value());
        if (en_penalty > 0)
        loglikes.ColRange(1, 1050).Add(en_penalty);

		if (loglikes.NumRows() == 0) {
			KALDI_WARN << "Zero-length utterance: " << key;
			num_fail++;
			continue;
		}

		// decoding
		if (search == "beam")
			decoder.BeamSearch(loglikes);
		else if (search == "greedy")
			decoder.GreedySearch(loglikes);
		else
			KALDI_ERR << "UnSupported search function: " << search;

		if (decoder.GetBestPath(words, logp)) {
			words_writer.Write(key, words);
			if (word_syms != NULL) {
				std::cerr << key << ' ';
				for (size_t i = 1; i < words.size(); i++) {
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


