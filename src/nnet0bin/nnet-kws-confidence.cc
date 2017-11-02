// nnetbin/nnet-kws-confidence.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
// Copyright 2016-2017   Shanghai Jiao Tong University (author: Wei Deng)

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

#include <limits>

#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-loss.h"
#include "nnet0/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet0;
  try {
    const char *usage =
        "Perform forward pass through Neural Network to get key words confidence score.\n"
        "\n"
        "Usage:  nnet-kws-confidence [options] <model-in> <feature-rspecifier> <feature-wspecifier> <smooth_wspecifier> <confidence_wspecifier>\n"
        "e.g.: \n"
        " nnet-kws-confidence nnet ark:features.ark ark:mlpoutput.ark ark:smooth.ark ark:confidence.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = false;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    using namespace kaldi;
    using namespace kaldi::nnet0;
    typedef kaldi::int32 int32;

    int32 time_shift = 0;
    po.Register("time-shift", &time_shift, "LSTM : repeat last input frame N-times, discrad N initial output frames."); 

    int32 w_smooth = 10;
    po.Register("smooth-window", &w_smooth, "Smooth the posteriors over a fixed time window of size smooth-window.");
    int32 w_max = 40;
    po.Register("sliding-window", &w_max, "The confidence score is computed within a sliding window of size sliding-window.");


    std::string keywords_str;
    po.Register("keywords-id", &keywords_str, "keywords index in network output.");
    std::vector<int32> keywords;
    if (!kaldi::SplitStringToIntegers(keywords_str, ":", false, &keywords))
    	KALDI_ERR << "Invalid keywords id string " << keywords_str;

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3),
    	smooth_wspecifier = po.GetArg(4),
		confidence_wspecifier = po.GetArg(5);
        
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    // optionally remove softmax,
    Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
    if (no_softmax) {
      if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
        KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename;
        nnet.RemoveComponent(nnet.NumComponents()-1);
      } else {
        KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
      }
    }

    // avoid some bad option combinations,
    if (apply_log && no_softmax) {
      KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
    }

    // we will subtract log-priors later,
    PdfPrior pdf_prior(prior_opts); 

    // disable dropout,
    nnet_transf.SetDropoutRetention(1.0);
    nnet.SetDropoutRetention(1.0);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);
    BaseFloatMatrixWriter smooth_writer(smooth_wspecifier);
    BaseFloatMatrixWriter confidence_writer(confidence_wspecifier);
    //BaseFloatVectorWriter confidence_writer(confidence_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host, post_smooth;
    Matrix<BaseFloat> confidence;


    Timer time;
    double time_now = 0;
    int32 num_done = 0;
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::string utt = feature_reader.Key();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << utt
                    << ", " << mat.NumRows() << "frm";

      
      if (!KALDI_ISFINITE(mat.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in features for " << utt;
      }

      // time-shift, copy the last frame of LSTM input N-times,
      if (time_shift > 0) {
        int32 last_row = mat.NumRows() - 1; // last row,
        mat.Resize(mat.NumRows() + time_shift, mat.NumCols(), kCopyData);
        for (int32 r = last_row+1; r<mat.NumRows(); r++) {
          mat.CopyRowFromVec(mat.Row(last_row), r); // copy last row,
        }
      }
      
      // push it to gpu,
      feats = mat;

      // fwd-pass, feature transform,
      nnet_transf.Feedforward(feats, &feats_transf);
      if (!KALDI_ISFINITE(feats_transf.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in transformed-features for " << utt;
      }

      // fwd-pass, nnet,
      nnet.Feedforward(feats_transf, &nnet_out);
      if (!KALDI_ISFINITE(nnet_out.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in nn-output for " << utt;
      }
      
      // convert posteriors to log-posteriors,
      if (apply_log) {
        if (!(nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0)) {
          KALDI_WARN << utt << " "
                     << "Applying 'log' to data which don't seem to be probabilities "
                     << "(is there a softmax somwhere?)";
        }
        nnet_out.Add(1e-20); // avoid log(0),
        nnet_out.ApplyLog();
      }
     
      // subtract log-priors from log-posteriors or pre-softmax,
      if (prior_opts.class_frame_counts != "") {
        if (nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0) {
          KALDI_WARN << utt << " " 
                     << "Subtracting log-prior on 'probability-like' data in range [0..1] " 
                     << "(Did you forget --no-softmax=true or --apply-log=true ?)";
        }
        pdf_prior.SubtractOnLogpost(&nnet_out);
      }

      // download from GPU,
      nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
      nnet_out.CopyToMat(&nnet_out_host);

      // time-shift, remove N first frames of LSTM output,
      if (time_shift > 0) {
        Matrix<BaseFloat> tmp(nnet_out_host);
        nnet_out_host = tmp.RowRange(time_shift, tmp.NumRows() - time_shift);
      }

      // kws confidence
      int rows = nnet_out_host.NumRows();
      //int cols = nnet_out_host.NumCols();
      int cols = keywords.size()+1;
      post_smooth.Resize(rows, cols);
      confidence.Resize(rows, 2*cols);
      int hs, hm;
      float sum, max, maxid, mul;

      // posterior smoothing
      for (int j = 0; j < rows; j++) {
    	  for (int i = 0; i < cols; i++) {
    		  hs = j-w_smooth+1 > 0 ? j-w_smooth+1 : 0;
    		  sum = 0;
    		  for (int k = hs; k <= j; k++) {
    			  sum += nnet_out_host(k, keywords[i]);
    		  }
    		  post_smooth(j, i) = sum/(j-hs+1);
    	  }
      }

      // compute confidence score
      // confidence.Set(1.0);
      for (int j = 0; j < rows; j++) {
          mul = 1.0;
    	  for (int i = 1; i < cols; i++) { // 1,2,...,n-1 keywords
    		  hm = j-w_max+1 > 0 ? j-w_max+1 : 0;
    		  max = 0;
              maxid = hm;
			  for (int k = hm; k <= j; k++) {
                  if (max < post_smooth(k, i)) {
                       max = post_smooth(k, i);
                       maxid = k;
                  } 
			  }
			  confidence(j,2*i) = max;
			  confidence(j,2*i+1) = maxid;
              mul *= max;
    	  }
    	  confidence(j,0) = pow(mul, 1.0/(cols-1));
    	  confidence(j,1) = j;
      }

      // write,
      if (!KALDI_ISFINITE(nnet_out_host.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
      }
      feature_writer.Write(feature_reader.Key(), nnet_out_host);
      smooth_writer.Write(feature_reader.Key(), post_smooth);
      confidence_writer.Write(feature_reader.Key(), confidence);

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
