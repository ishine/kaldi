
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "nnet2/decodable-am-nnet.h"
#include "base/timer.h"

#include "iot/dec-core.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet2;
    using namespace kaldi::iot;

    const char *usage =
        "Usage: iot [options] <nnet-in> <fst> <features-rspecifier>";

    ParseOptions po(usage);
    Timer timer;

    BaseFloat acoustic_scale = 0.1;
    DecCoreOptions config;

    config.Register(&po, false);
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        feature_rspecifier = po.GetArg(3);

    TransitionModel trans_model;
    AmNnet am_nnet;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    }

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatCuMatrixReader feature_reader(feature_rspecifier);

      Wfst *decode_fst = new Wfst;
      {
        bool binary;
        Input ki(fst_in_str, &binary);
        decode_fst->Read(ki.Stream(), binary);
      }

      timer.Reset();

      {
        DecCore decoder(decode_fst, NULL, config);

        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          const CuMatrix<BaseFloat> &features (feature_reader.Value());
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }
          
          KALDI_LOG << utt;

          bool pad_input = true;
          DecodableAmNnet nnet_decodable(trans_model,
                                         am_nnet,
                                         features,
                                         pad_input,
                                         acoustic_scale);
          //decoder.InitDecoding();
          //decoder.AdvanceDecoding(&nnet_decodable);
          //decoder.FinalizeDecoding();
          decoder.Decode(&nnet_decodable);
          decoder.GetBestTransitionPath();
        }
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    }

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
