// nnetbin/compute-svector.cc

// Copyright 2014       Brno University of Technology (Author: Karel Vesely)

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
#include "nnet0/nnet-nnet.h"

/** @brief Convert features into posterior format, is used to specify NN training targets. */
int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi;
  using namespace kaldi::nnet0;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Compute sex vector\n"
        "e.g.:\n"
        " compute-svector scp:feats.scp scp:ali.scp ark,t:svector.txt nnet\n";

    ParseOptions po(usage);

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
          targets_rspecifier = po.GetArg(2),
		  wspecifier = po.GetArg(3),
          model_filename = po.GetArg(4);

    int32 num_done = 0;
    SequentialBaseFloatMatrixReader feats_reader(feature_rspecifier);
    RandomAccessInt32VectorReader num_ali_reader(targets_rspecifier);
    BaseFloatVectorWriter swriter(wspecifier);

     Nnet nnet_transf;
     if (feature_transform != "") {
       nnet_transf.Read(feature_transform);
     }

     Nnet nnet;
     nnet.Read(model_filename);

     CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
     Matrix<BaseFloat> nnet_out_host;

	int out_dim = nnet.OutputDim();
	Vector<BaseFloat> final_svector(out_dim), svec(out_dim);


	for (; !feats_reader.Done(); feats_reader.Next()) {
	  const Matrix<BaseFloat> &mat = feats_reader.Value();
	  std::string utt = feats_reader.Key();
	  const std::vector<int32> &label = num_ali_reader.Value(utt);

	  feats = mat;
      // fwd-pass, feature transform,
      nnet_transf.Feedforward(feats, &feats_transf);

      // fwd-pass, nnet,
      nnet.Feedforward(feats_transf, &nnet_out);

      // download from GPU,
      nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
      nnet_out.CopyToMat(&nnet_out_host);


	  int32 min = std::min(mat.NumRows(), label.size());
	  svec.SetZero();
	  int num_frames = 0;
	  for (int i = 0; i < min; i++) {
		  if(label[i] != 0) {
			  svec.AddVec(1.0, nnet_out_host.Row(i));
			  num_frames++;
		  }
	  }
	  svec.Scale(1.0/num_frames);
	  final_svector.AddVec(1.0, svec);
	  num_done++;
	}

	final_svector.Scale(1.0/num_done);
	swriter.Write("svector", final_svector);

	KALDI_LOG << "Computed " << num_done << " utterance.";
	return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


