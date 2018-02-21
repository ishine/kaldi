// online0bin/online-ivector-enroll-test.cc

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

#include "base/timer.h"
#include "feat/wave-reader.h"
#include "base/kaldi-common.h"
#include "online0/online-ivector-extractor.h"

int main(int argc, char *argv[])
{
	try {

	    using namespace kaldi;
	    using namespace fst;

	    typedef kaldi::int32 int32;

	    const char *usage =
	    		"Reads in wav file(s) and simulates extract speaker ivector online.\n"
	    		"Note: some configuration values and inputs are"
	    	"set via config files whose filenames are passed as options\n"
	    	"\n"
	        "Usage: online-ivector-enroll-test [config option] <spk2utt-rspecifier> <wav-rspecifier> "
	        "<ivector-wspecifier> [<num-utt-wspecifier>]\n"
	    	"e.g.: \n"
	        "online-ivector-enroll-test --cfg=conf/ivector.conf spk2utt wav.scp ark,t:ivectors.1.ark ark,t:spk_num_utts.ark \n";

	    ParseOptions po(usage);

	    std::string cfg;
	    po.Register("cfg", &cfg, "ivector extractor config file");

        std::string audio_format = "wav";
        po.Register("audio-format", &audio_format, "input audio format(e.g. wav, pcm)");

        po.Read(argc, argv);

        if (po.NumArgs() < 2) {
        		po.PrintUsage();
        		exit(1);
        }

        std::string spk2utt_rspecifier = po.GetArg(1),
        				wav_rspecifier = po.GetArg(2),
                  ivector_wspecifier = po.GetArg(3),
                  num_utts_wspecifier = po.GetOptArg(4);

        SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
        RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
        BaseFloatVectorWriter ivector_writer(ivector_wspecifier);
        Int32Writer num_utts_writer(num_utts_wspecifier);

        Ivector *ivector;

        OnlineIvectorExtractor extractor(cfg);
        extractor.InitExtractor();

        int64 num_spk_done = 0, num_spk_err = 0,
            num_utt_done = 0, num_utt_err = 0;

        for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        		std::string spk = spk2utt_reader.Key();
        		const std::vector<std::string> &uttlist = spk2utt_reader.Value();
        		if (uttlist.empty()) {
        			KALDI_ERR << "Speaker with no utterances.";
			}

        		Vector<BaseFloat> spk_mean;
        		int32 utt_count = 0;
        		std::vector<Vector<BaseFloat> > ivectors(uttlist.size());
        		for (size_t i = 0; i < uttlist.size(); i++) {
        			std::string utt = uttlist[i];
				if (!wav_reader.HasKey(utt)) {
					KALDI_WARN << "No wav present in input for utterance " << utt;
					num_utt_err++;
				} else {
				    const WaveData &audio_data = wav_reader.Value(utt);
				    SubVector<BaseFloat> data(audio_data.Data(), 0);
				    // one utterance
				    	extractor.Reset();
				    extractor.FeedData((void*)data.Data(), data.Dim()*sizeof(float), FEAT_END);
				    ivector = extractor.GetCurrentIvector(0);
				    ivectors[i] = ivector->ivector_;

		            num_utt_done++;
		            utt_count++;
				}
        		}

        		if (utt_count == 0) {
        			KALDI_WARN << "Not producing output for speaker " << spk
						 << " since no utterances had iVectors";
        			num_spk_err++;
			} else {
				extractor.GetEnrollSpeakerIvector(ivectors, spk_mean, 0);
				ivector_writer.Write(spk, spk_mean);
				if (num_utts_wspecifier != "")
					num_utts_writer.Write(spk, utt_count);
				num_spk_done++;
			}

        }

        KALDI_LOG << "Computed mean of " << num_spk_done << " speakers ("
                  << num_spk_err << " with no utterances), consisting of "
                  << num_utt_done << " utterances (" << num_utt_err
                  << " absent from input).";

	    return 0;
	} catch(const std::exception& e) {
		std::cerr << e.what();
		return -1;
	}
} // main()


