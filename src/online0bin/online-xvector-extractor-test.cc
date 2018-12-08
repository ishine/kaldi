// online0bin/online-xvector-extractor-test.cc

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

#include "base/timer.h"
#include "feat/wave-reader.h"
#include "online0/online-xvector-extractor.h"
#include "online0/speex-ogg-dec.h"

int main(int argc, char *argv[])
{
	try {
	    using namespace kaldi;

	    typedef kaldi::int32 int32;

	    const char *usage =
	    		"Reads in wav file(s) and simulates online xvector extractor with neural nets "
	    		"(nnet3 setup), Note: some configuration values and inputs are\n"
	    	"set via config files whose filenames are passed as options\n"
	    	"\n"
	        "Usage: online-xvector-extractor-test [config option]\n"
	    	"e.g.: \n"
	        "	online-xvector-extractor-test --cfg=conf/xvector.conf wav.scp <xvector-wspecifier> \n"
	    	"	online-xvector-extractor-test --audio-format=raw --cfg=conf/xvector.conf "
	    	"<feats-rspecifier> <xvector-wspecifier> \n";

	    ParseOptions po(usage);

	    std::string cfg;
	    po.Register("cfg", &cfg, "xvector config file");

        std::string audio_format = "wav";
        po.Register("audio-format", &audio_format, "input audio format(e.g. wav, pcm, ogg)");

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
        		po.PrintUsage();
        		exit(1);
        }

        std::string wavlist_rspecifier = po.GetArg(1);
        std::string feats_rspecifier = po.GetArg(1);
        std::string xvector_wspecifier = po.GetArg(2);

        OnlineXvectorExtractor extractor(cfg);
        extractor.InitExtractor();

	    BaseFloat  total_frames = 0;
        size_t size;
        Matrix<BaseFloat> audio_data;
        Xvector *xvector;
	    FeatState state;
        char fn[1024];

        BaseFloatVectorWriter xvector_writer(xvector_wspecifier);

		Timer timer;
        if (audio_format != "raw") {
        	std::ifstream wavlist_reader(wavlist_rspecifier);

			while (wavlist_reader.getline(fn, 1024)) {
				if (audio_format == "wav") {
					bool binary;
					Input ki(fn, &binary);
					WaveHolder holder;
					holder.Read(ki.Stream());

					const WaveData &wave_data = holder.Value();
					audio_data = wave_data.Data();
				} else if (audio_format == "pcm") {
					std::ifstream pcm_reader(fn, std::ios::binary);
					// get length of file:
					pcm_reader.seekg(0, std::ios::end);
					int length = pcm_reader.tellg();
					pcm_reader.seekg(0, std::ios::beg);
					size = length/sizeof(short);
					std::vector<short> buffer(size);
					// read data as a block:
					pcm_reader.read((char*)&buffer.front(), length);
					audio_data.Resize(1, size);
					for (int i = 0; i < size; i++)
						audio_data(0, i) = buffer[i];
					pcm_reader.close();
				} else if (audio_format == "ogg") {
					std::ifstream ogg_reader(fn, std::ios::binary);
					// get length of file:
					ogg_reader.seekg(0, std::ios::end);
					int length = ogg_reader.tellg();
					ogg_reader.seekg(0, std::ios::beg);

					std::vector<char> buffer(length);
					ogg_reader.read((char*)(&buffer.front()), length);
                    char *pcm_audio = NULL;
                #ifdef HAVE_SPEEX
                    length = SpeexOggDecoder(&buffer.front(), length, pcm_audio);
                #endif

                    if (length <= 0) {
					    KALDI_WARN << std::string(fn) << " not converted successfully.";
                        continue;
                    }

					size = length/sizeof(short);
					// read data as a block:
					audio_data.Resize(1, size);
					for (int i = 0; i < size; i++)
						audio_data(0, i) = ((short *)pcm_audio)[i];
					ogg_reader.close();

                    if (NULL != pcm_audio)
                        delete pcm_audio;
                }
				else
					KALDI_ERR << "Unsupported input audio format, now only support wav, pcm or ogg.";

				// get the data for channel zero (if the signal is not mono, we only
				// take the first channel).
				SubVector<BaseFloat> data(audio_data, 0);

				// one utterance
				extractor.Reset();

				state = FEAT_END;
				extractor.FeedData((void*)data.Data(), data.Dim()*sizeof(float), state);
				xvector = extractor.GetCurrentXvector(2);

				if (NULL == xvector) {
					KALDI_WARN << std::string(fn) << " extract empty xvector.";
					continue;
				}
				xvector->utt = std::string(fn);
				xvector_writer.Write(xvector->utt, xvector->xvector_);

				total_frames += xvector->num_frames_;
			}
			wavlist_reader.close();
        } else {

        	SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
        	for (; !feats_reader.Done(); feats_reader.Next()) {
        		std::string utt = feats_reader.Key();
        		const Matrix<BaseFloat> &mat = feats_reader.Value();
        		Matrix<BaseFloat> feat(mat.NumRows(), mat.NumCols(), kUndefined, kStrideEqualNumCols);
        		feat.CopyFromMat(mat);

				// one utterance
				extractor.Reset();

				state = FEAT_END;
				extractor.FeedData((void*)feat.Data(), feat.SizeInBytes(), state);
				xvector = extractor.GetCurrentXvector(2);

				if (NULL == xvector) {
					KALDI_WARN << utt << " extract empty xvector.";
					continue;
				}
				xvector->utt = utt;
				xvector_writer.Write(xvector->utt, xvector->xvector_);

				total_frames += xvector->num_frames_;
        	}
        }

	    double elapsed = timer.Elapsed();
		KALDI_LOG << "Time taken [excluding initialization] "<< elapsed
			  << "s: real-time factor assuming 100 frames/sec is "
			  << (elapsed*100.0/total_frames);

	    return 0;
	} catch(const std::exception& e) {
		std::cerr << e.what();
		return -1;
	}
} // main()


