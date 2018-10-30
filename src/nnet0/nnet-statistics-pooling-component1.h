// nnet0/nnet-statistics-pooling-component.h

// Copyright 2018  Alibaba Inc (author: Wei Deng)

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


#ifndef KALDI_NNET_NNET_STATISTICS_POOLING_COMPONENT_H_
#define KALDI_NNET_NNET_STATISTICS_POOLING_COMPONENT_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet0 {

/**
 * StatisticPoolingComponent :
 * extract moving-average mean and standard-deviation statistics.
 */
class StatisticsPoolingComponent : public Component {
 public:
	StatisticsPoolingComponent(int32 dim_in, int32 dim_out)
    : Component(dim_in, dim_out), use_mean_(1), use_stddev_(1), batch_size_(20)
  { }
  ~StatisticsPoolingComponent()
  { }

  Component* Copy() const { return new StatisticsPoolingComponent(*this); }
  ComponentType GetType() const { return kStatisticsPoolingComponent; }
  
  void InitData(std::istream &is) {
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<Mean>") ReadBasicType(is, false, &use_mean_);
      else if (token == "<Stddev>") ReadBasicType(is, false, &use_stddev_);
      else if (token == "<BatchSize>") ReadBasicType(is, false, &batch_size_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (Mean|Stddev|BatchSize)";
      is >> std::ws; // eat-up whitespace
    }
    // check
    KALDI_ASSERT((use_mean_ == 0 || use_mean_ == 1) && (use_stddev_ == 0 || use_stddev_ == 1));
  }

  void ReadData(std::istream &is, bool binary) {
    // statistics pooling hyperparameters
    ExpectToken(is, binary, "<Mean>");
    ReadBasicType(is, binary, &use_mean_);
    ExpectToken(is, binary, "<Stddev>");
    ReadBasicType(is, binary, &use_stddev_);
    ExpectToken(is, binary, "<BatchSize>");
    ReadBasicType(is, binary, &batch_size_);

    //
    // Sanity checks:
    //
    // check output dim:
    KALDI_ASSERT(output_dim_ == (use_mean_+use_stddev_)*input_dim_);
    //
  }

  void WriteData(std::ostream &os, bool binary) const {
    // pooling hyperparameters
    WriteToken(os, binary, "<Mean>");
    WriteBasicType(os, binary, use_mean_);
    WriteToken(os, binary, "<Stddev>");
    WriteBasicType(os, binary, use_stddev_);
    WriteToken(os, binary, "<BatchSize>");
    WriteBasicType(os, binary, batch_size_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {

	KALDI_ASSERT(out->NumRows() == nstream_);
	KALDI_ASSERT(in.NumRows() == nstream_ * batch_size_);


	int feat_dim = in.NumCols();

	mean_.Resize(nstream_, feat_dim, kSetZero);
	//square_mean_.Resize(nstream_, feat_dim, kSetZero);
	stddev_.Resize(nstream_, feat_dim, kSetZero);

	// mean
	for (int s = 0; s < nstream_; s++) {
	  if (use_mean_ == 1 && utt_num_frame_[s] > 0) {
		  mean_.Row(s).AddRowSumMat(1.0, in.RowRange(s*batch_size_, utt_num_frame_[s]), 0.0);
		  mean_.Row(s).Scale(1.0/utt_num_frame_[s]);
	  }
	}

	// stddev
	if (use_stddev_ == 1) {
		indexes_.resize(in.NumRows());
		for (int i = 0; i < in.NumRows(); i++) {
			int s = i / batch_size_;
			int t = i % batch_size_;
			indexes_[i] = t < utt_num_frame_[s] ? s : -1;
		}
		CuArray<int> cu_indexes(indexes_);

		// xn-mean
		CuMatrix<BaseFloat> square_in = in;
		square_in.AddRows(-1.0, mean_, cu_indexes);
		square_in.ApplyPow(2.0);

		for (int s = 0; s < nstream_; s++) {
			if (utt_num_frame_[s] > 0) {
			  stddev_.Row(s).AddRowSumMat(1.0, square_in.RowRange(s*batch_size_, utt_num_frame_[s]), 0.0);
			  stddev_.Row(s).Scale(1.0/utt_num_frame_[s]);
			}
		}
		stddev_.ApplyPow(0.5);
	}

	/*
	// stddev = (E(x^2) - E(x)^2)^(1/2)
	if (use_stddev_ == 1) {
	  stddev_.CopyFromMat(mean_);
	  stddev_.ApplyPow(2.0);
	  stddev_.Scale(-1.0);
	  stddev_.AddMat(1.0, square_mean_);
	  stddev_.ApplyPow(0.5);
	}
	*/

	out->ColRange(0, feat_dim).CopyFromMat(mean_);

	if (use_stddev_ == 1)
		out->ColRange(feat_dim, feat_dim).CopyFromMat(stddev_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {

	int feat_dim = in.NumCols(), input_rows = in.NumRows();
	int s, t;

	CuSubMatrix<BaseFloat> mean_deriv(out_diff, 0, nstream_, 0, feat_dim);
	CuArray<int> cu_indexes;

	in_diff->SetZero();
	//
	indexes_.resize(input_rows);
	if (use_stddev_ == 1) {
		for (int i = 0; i < input_rows; i++) {
		  s = i / batch_size_;
		  t = i % batch_size_;
		  indexes_[i] = t < utt_num_frame_[s] ? i : -1;
		}
		cu_indexes = indexes_;
		in_diff->AddRows(1.0, in, cu_indexes);
	}


	for (int i = 0; i < input_rows; i++) {
	  s = i / batch_size_;
	  t = i % batch_size_;
	  indexes_[i] = t < utt_num_frame_[s] ? s : -1;
	}
	cu_indexes = indexes_;

	/*
	Matrix<BaseFloat> tmp_stddev_(stddev_);
	Matrix<BaseFloat> tmp_mean_(mean_);
	Matrix<BaseFloat> tmp_square_mean_(square_mean_);
	Matrix<BaseFloat> tmp_in(in);
	Matrix<BaseFloat> tmp_out(out);
	Matrix<BaseFloat> tmp_mean_deriv(mean_deriv);
	*/


	// stddev diff
	if (use_stddev_ == 1) {
		in_diff->AddRows(-1.0, mean_, cu_indexes);
		//Matrix<BaseFloat> tmp_in_diff1(*in_diff);
		in_diff->DivRows(stddev_, cu_indexes);

		CuSubMatrix<BaseFloat> stddev_deriv(out_diff, 0, nstream_, feat_dim, feat_dim);
		in_diff->MulRows(stddev_deriv, cu_indexes);
	}

	/*
	Matrix<BaseFloat> tmp_in_diff(*in_diff);
	Matrix<BaseFloat> tmp_stddev_deriv(stddev_deriv);

	for (int i = 0; i < in_diff->NumRows(); i++) {
	for (int j = 0; j < in_diff->NumCols(); j++) {
		if (!KALDI_ISFINITE(tmp_in_diff(i,j)))
			std::cout<<"i = " << i << " j = " << j << " " << tmp_in_diff1(i,j) << " " << tmp_stddev_(indexes_[i],j) << std::endl;
	}
	}

	std::cout<< tmp_stddev_.NumCols() << " " << tmp_mean_.NumCols() << " " << tmp_square_mean_.NumCols() << " " << tmp_in.NumCols() << " " << tmp_out.NumCols() << " " << tmp_stddev_deriv.NumCols() << " " << tmp_mean_deriv.NumCols() << " " << tmp_in_diff.NumCols() << std::endl;
	*/

	// mean diff
	if (use_mean_ == 1)
		in_diff->AddRows(1.0, mean_deriv, cu_indexes);

	/*
	// scale
	for (int s = 0; s < nstream_; s++) {
		if(use_mean_ == 1 && utt_num_frame_[s] > 0)
			in_diff->RowRange(s*batch_size_, batch_size_).Scale(1.0/utt_num_frame_[s]);
	}
	*/
  }

  int32 GetSubSampleRate() {
    return batch_size_;
  }

  void SetStream(int nstream) {
    nstream_ = nstream;
  }

  int32 GetStream() {
      return nstream_;
  }

  /// set the utterance length used for parallel training
  void SetSeqLengths(const std::vector<int32> &sequence_lengths) {
	  utt_num_frame_ = sequence_lengths;
  }

 private:
  int32 use_mean_;
  int32 use_stddev_;
  int32	nstream_;
  int32 batch_size_;

  std::vector<int> utt_num_frame_;
  std::vector<int> indexes_;
  CuMatrix<BaseFloat> mean_;
  CuMatrix<BaseFloat> square_mean_;
  CuMatrix<BaseFloat> stddev_;
};

} // namespace nnet0
} // namespace kaldi

#endif
