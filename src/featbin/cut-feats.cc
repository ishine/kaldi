// featbin/cut-feats.cc

// Copyright 2018-2019  Alibaba Inc (author: Wei Deng)

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
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;
    typedef kaldi::int32 int32;

    const char *usage =
        "Cut feature files,\n"
        "Usage: cut-feats <cut_timestamp> <in-rxfilename> <out-wxfilename>\n"
        " e.g. cut-feats scp:cut_timestamp.scp scp:feats_in.scp ark,scp:feats_out.ark,feats_out.scp \n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string cut_timestamp_rspecifier, feature_rspecifier, feature_wspecifier;

    cut_timestamp_rspecifier = po.GetArg(1),
	feature_rspecifier = po.GetArg(2),
	feature_wspecifier = po.GetArg(3);

    RandomAccessInt32VectorReader timestamp_reader(cut_timestamp_rspecifier);
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);
    int num_done = 0;

    for ( ; !feature_reader.Done(); feature_reader.Next()) {
    	std::string utt = feature_reader.Key();
		if (!timestamp_reader.HasKey(utt)) {
			KALDI_WARN << utt << ", missing time stamp";
			continue;
		}

		std::vector<int32> timstp = timestamp_reader.Value(utt);
		const Matrix<BaseFloat> &mat = feature_reader.Value();

        int len = mat.NumRows();
		if (timstp[1] <= timstp[0] || timstp[1] > len+5) {
			KALDI_WARN << utt << ", invalid time stamp ";
			continue;
		}

        if (timstp[1] > len) {
            KALDI_WARN << utt << ", feature length = " << len << ", while time stamp = " << timstp[1] << ", reset to feature length";
            timstp[1] = len;
        }

        Matrix<BaseFloat> cut_mat(timstp[1]-timstp[0], mat.NumCols(), kUndefined); 
        cut_mat.CopyFromMat(mat.RowRange(timstp[0], timstp[1]-timstp[0]));
		feature_writer.Write(utt, cut_mat);
        num_done++;
    }

    KALDI_LOG << "Cut " << num_done << " features from " << PrintableRxfilename(feature_rspecifier)
              << " to " << PrintableWxfilename(feature_wspecifier);
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
