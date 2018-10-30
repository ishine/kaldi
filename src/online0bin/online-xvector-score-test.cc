// online0bin/online-xvector-score-test.cc

// Copyright 2018   Alibaba Inc (author: Wei Deng)

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
#include "online0/online-xvector-extractor.h"

int main(int argc, char *argv[])
{
	try {
	    using namespace kaldi;

	    typedef kaldi::int32 int32;

	    const char *usage =
	    		"Computes two xvector(list) score.\n"
	    		"Note: some configuration values and inputs are\n"
	    	"set via config files whose filenames are passed as options\n"
	    	"\n"
	        "Usage: online-xvector-score-test --cfg=conf/xvector.conf "
	        "<train-xvector-rspecifier> <test-xvector-rspecifier> <trials-rxfilename> <scores-wxfilename>\n"
			"\n"
			"e.g.: online-xvector-score-test --cfg=conf/xvector.conf --num-utts=ark:exp/train/num_utts.ark "
			"ark:exp/train/spk_xvectors.ark ark:exp/test/xvectors.ark trials scores\n";

	    ParseOptions po(usage);

	    std::string cfg, num_utts_rspecifier;
	    po.Register("cfg", &cfg, "xvector config file");
	    po.Register("num-utts", &num_utts_rspecifier, "Table to read the number of "
	                    "utterances per speaker, e.g. ark:num_utts.ark\n");

        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
        		po.PrintUsage();
        		exit(1);
        }

        std::string train_xvector_rspecifier = po.GetArg(1),
            test_xvector_rspecifier = po.GetArg(2),
            trials_rxfilename = po.GetArg(3),
            scores_wxfilename = po.GetArg(4);

        OnlineXvectorExtractor extractor(cfg);
        extractor.InitExtractor();

        RandomAccessInt32Reader num_utts_reader(num_utts_rspecifier);
        SequentialBaseFloatVectorReader train_xvector_reader(train_xvector_rspecifier);
        SequentialBaseFloatVectorReader test_xvector_reader(test_xvector_rspecifier);

        typedef unordered_map<std::string, Vector<BaseFloat>*, StringHasher> HashType;
        HashType train_xvectors, test_xvectors;

        int64 num_train_xvectors = 0, num_test_xvectors = 0, num_train_errs = 0;
        int64 num_trials_done = 0, num_trials_err = 0;

        KALDI_LOG << "Reading train xvectors";
        for (; !train_xvector_reader.Done(); train_xvector_reader.Next()) {
			std::string spk = train_xvector_reader.Key();
			if (train_xvectors.count(spk) != 0) {
				KALDI_ERR << "Duplicate training xvector found for speaker " << spk;
			}

			if (!num_utts_rspecifier.empty()) {
				if (!num_utts_reader.HasKey(spk)) {
					KALDI_WARN << "Number of utterances not given for speaker " << spk;
					num_train_errs++;
					continue;
				}
			}
            Vector<BaseFloat> *xvector = new Vector<BaseFloat>();
            *xvector = train_xvector_reader.Value();
			train_xvectors[spk] = xvector;
			num_train_xvectors++;
		}
		KALDI_LOG << "Read " << num_train_xvectors << " training xvectors, "
				<< "errors on " << num_train_errs;

		KALDI_LOG << "Reading test xvectors";
		for (; !test_xvector_reader.Done(); test_xvector_reader.Next()) {
			std::string spk = test_xvector_reader.Key();
			if (test_xvectors.count(spk) != 0) {
				KALDI_ERR << "Duplicate training xvector found for speaker " << spk;
			}
            Vector<BaseFloat> *xvector = new Vector<BaseFloat>();
            *xvector = test_xvector_reader.Value();
			test_xvectors[spk] = xvector;
			num_test_xvectors++;
		}
		KALDI_LOG << "Read " << num_test_xvectors << " training xvectors.";


		Input ki(trials_rxfilename);
		bool binary = false;
		Output ko(scores_wxfilename, binary);

		double sum = 0.0, sumsq = 0.0;
		std::string line;

        Timer timer;
		while (std::getline(ki.Stream(), line)) {
			std::vector<std::string> fields;
			SplitStringToVector(line, " \t\n\r", true, &fields);
			if (fields.size() != 2) {
				KALDI_ERR << "Bad line " << (num_trials_done + num_trials_err)
					  << "in input (expected two fields: key1 key2): " << line;
			}

			std::string key1 = fields[0], key2 = fields[1];
			if (train_xvectors.count(key1) == 0) {
				KALDI_WARN << "Key " << key1 << " not present in training xvectors.";
				num_trials_err++;
				continue;
			}

			if (test_xvectors.count(key2) == 0) {
				KALDI_WARN << "Key " << key2 << " not present in test xvectors.";
				num_trials_err++;
				continue;
			}

			int32 num_train_examples;
			if (!num_utts_rspecifier.empty()) {
				// we already checked that it has this key.
				num_train_examples = num_utts_reader.Value(key1);
			} else {
				num_train_examples = 1;
			}

			BaseFloat score = extractor.GetScore(*train_xvectors[key1], num_train_examples, *test_xvectors[key2]);
			sum += score;
			sumsq += score * score;
			num_trials_done++;
			ko.Stream() << key1 << ' ' << key2 << ' ' << score << std::endl;
		}

        double elapsed = timer.Elapsed();

	    for (HashType::iterator iter = train_xvectors.begin();
	         iter != train_xvectors.end(); ++iter)
	    	delete iter->second;
	    for (HashType::iterator iter = test_xvectors.begin();
	         iter != test_xvectors.end(); ++iter)
	    	delete iter->second;


	    if (num_trials_done != 0) {
			BaseFloat mean = sum / num_trials_done, scatter = sumsq / num_trials_done,
			variance = scatter - mean * mean, stddev = sqrt(variance);
			KALDI_LOG << "Mean score was " << mean << ", standard deviation was " << stddev;
		}
		KALDI_LOG << "Processed " << num_trials_done << " trials, " << num_trials_err << " had errors.";
        KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
                      << "s: real-time process trials "<< (num_trials_done/elapsed) << " per second.";

		return (num_trials_done != 0 ? 0 : 1);
	} catch(const std::exception& e) {
		std::cerr << e.what();
		return -1;
	}
} // main()


