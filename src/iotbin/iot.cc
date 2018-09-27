// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

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

#include "feat/wave-reader.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

#include "iot/decoder.h"

namespace kaldi {
namespace iot {

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);

  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";

  if (word_syms != NULL) {
    std::cerr << utt << ' ';
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      std::cerr << s << ' ';
    }
    std::cerr << std::endl;
  }
}

}
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using namespace kaldi::iot;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Reads in wav file(s) and simulates online decoding with neural nets\n"
        "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
        "optional endpointing.  Note: some configuration values and inputs are\n"
        "set via config files whose filenames are passed as options\n"
        "\n"
        "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
        "<spk2utt-rspecifier> <wav-rspecifier> <lattice-wspecifier>\n"
        "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
        "you want to decode utterance by utterance.\n";

    ParseOptions po(usage);

    std::string word_syms_rxfilename;

    // feature_opts includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_opts;
    nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
    DecCoreConfig decoder_opts;
    EndPointerConfig end_pointer_opts;

    BaseFloat chunk_length_secs = 0.18;
    bool do_endpointing = false;
    bool online = true;

    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
    po.Register("online", &online,
                "You can set this to false to disable online iVector estimation "
                "and have all the data for each utterance used, even at "
                "utterance start.  This is useful where you just want the best "
                "results and don't care about online operation.  Setting this to "
                "false has the same effect as setting "
                "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
                "in the file given to --ivector-extraction-config, and "
                "--chunk-length=-1.");
    po.Register("num-threads-startup", &g_num_threads,
                "Number of threads used when initializing iVector extractor.");

    feature_opts.Register(&po);
    decodable_opts.Register(&po);
    decoder_opts.Register(&po);
    end_pointer_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      return 1;
    }

    std::string nnet3_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        spk2utt_rspecifier = po.GetArg(3),
        wav_rspecifier = po.GetArg(4),
        clat_wspecifier = po.GetArg(5),
        interp_lm_filename = po.GetArg(6);

    TransitionModel trans_model;
    nnet3::AmNnetSimple am_nnet;
    {
      bool binary;
      Input ki(nnet3_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    Wfst *la_fst = new Wfst;
    {
      bool binary;
      Input ki(fst_rxfilename, &binary);
      la_fst->Read(ki.Stream(), binary);
    }
 
    BaseFloat interp_scale = 0.2f;
    VectorFst<StdArc> *interp_lm_fst = fst::CastOrConvertToVectorFst(fst::ReadFstKaldiGeneric(interp_lm_filename));
    ApplyProbabilityScale(interp_scale, interp_lm_fst);
    fst::BackoffDeterministicOnDemandFst<StdArc> interp_lm_det_fst(*interp_lm_fst);
    fst::CacheDeterministicOnDemandFst<StdArc> interp_lm_det_cache_fst(&interp_lm_det_fst);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;

    int32 num_done = 0, num_err = 0;
    double tot_like = 0.0;
    int64 num_frames = 0;

    SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
    RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
    CompactLatticeWriter clat_writer(clat_wspecifier);

    OnlineTimingStats timing_stats;

    Decoder decoder(la_fst,
                    NULL,
                    trans_model,
                    am_nnet,
                    feature_opts,
                    decodable_opts,
                    decoder_opts);

    if (do_endpointing) {
      decoder.EnableEndPointer(end_pointer_opts);
    }

    for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
      std::string spk = spk2utt_reader.Key();
      const std::vector<std::string> &uttlist = spk2utt_reader.Value();  
      for (size_t i = 0; i < uttlist.size(); i++) {
        std::string utt = uttlist[i];
        if (!wav_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find audio for utterance " << utt;
          num_err++;
          continue;
        }
        const WaveData &wave_data = wav_reader.Value(utt);
        SubVector<BaseFloat> audio(wave_data.Data(), 0); // only use channel 0
        BaseFloat sample_rate = wave_data.SampFreq();

        AudioFormat audio_format = UNKNOWN_AUDIO_FORMAT;
        if (sample_rate == 8000) {
          audio_format = FLOAT_8K;
        } else if (sample_rate == 16000) {
          audio_format = FLOAT_16K;
        }
        KALDI_ASSERT(audio_format != UNKNOWN_AUDIO_FORMAT);

        int32 chunk_samples;
        if (chunk_length_secs > 0) {
          chunk_samples = int32(sample_rate * chunk_length_secs);
          if (chunk_samples == 0) chunk_samples = 1;
        } else {
          chunk_samples = std::numeric_limits<int32>::max();
        }

        decoder.StartSession(utt.c_str());
        OnlineTimer decoding_timer(utt);

        int32 samples_done = 0;
        while (samples_done < audio.Dim()) {
          int32 samples_remaining = audio.Dim() - samples_done;
          int32 n = chunk_samples < samples_remaining ? chunk_samples : samples_remaining;

          SubVector<BaseFloat> audio_chunk(audio, samples_done, n);
          decoder.AcceptAudio(audio_chunk.Data(), audio_chunk.SizeInBytes(), audio_format);

          samples_done += n;
          decoding_timer.WaitUntil(samples_done / sample_rate);     

          if (do_endpointing && decoder.EndpointDetected()) {
            break;
          }
        }
        decoder.StopSession();

        CompactLattice clat;
        decoder.GetLattice(true, &clat); // use_final_prob = true

        GetDiagnosticsAndPrintOutput(utt, word_syms, clat, &num_frames, &tot_like);

        decoding_timer.OutputStats(&timing_stats);
        // we want to output the lattice with un-scaled acoustics.
        BaseFloat inv_acoustic_scale = 1.0 / decodable_opts.acoustic_scale;
        ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

        clat_writer.Write(utt, clat);
        KALDI_LOG << "Decoded utterance " << utt;
        num_done++;
      }
    }
    //timing_stats.Print(online);
    timing_stats.Print(false);

    KALDI_LOG << "Decoded " << num_done << " utterances, "
              << num_err << " with errors.";
    KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
              << " per frame over " << num_frames << " frames.";
    delete la_fst;
    delete word_syms; // will delete if non-NULL.
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()
