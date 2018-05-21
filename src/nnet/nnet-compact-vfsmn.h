// nnet/nnet-compact-vfsmn.h

// Copyright 2018 (author: Kaituo Xu)

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


#ifndef KALDI_NNET_NNET_COMPACT_VFSMN_H_
#define KALDI_NNET_NNET_COMPACT_VFSMN_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class CompactVfsmn : public UpdatableComponent {
 public:
  CompactVfsmn(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out)
  { }
  ~CompactVfsmn()
  { }

  Component* Copy() const { return new CompactVfsmn(*this); }
  ComponentType GetType() const { return kCompactVfsmn; }

  void InitData(std::istream &is) {
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<Order>")         ReadBasicType(is, false, &order_);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (Order|LearnRateCoef)";
    }

    //
    // Initialize trainable parameters,
    //
    // Glorot Uniform
    filter_.Resize(order_, InputDim());
    float mean = 0.0;
    float range = sqrt(6.0 / (float)(1 + order_))*2;
    RandUniform(mean, range, &filter_);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'O': ExpectToken(is, binary, "<Order>");
          ReadBasicType(is, binary, &order_);
          break;
        case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        default:
          std::string token;
          ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    // Read the data (data follow the tokens),

    // filter matrix,
    filter_.Read(is, binary);

    KALDI_ASSERT(filter_.NumRows() == order_);
    KALDI_ASSERT(filter_.NumCols() == input_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<Order>");
    WriteBasicType(os, binary, order_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    if (!binary) os << "\n";
    // filter
    filter_.Write(os, binary);
  }

  int32 NumParams() const {
    return filter_.NumRows()*filter_.NumCols();
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 filter_num_elem = filter_.NumRows()*filter_.NumCols();
    gradient->Range(0, filter_num_elem).CopyRowsFromMat(filter_corr_);
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 filter_num_elem = filter_.NumRows()*filter_.NumCols();
    params->Range(0, filter_num_elem).CopyRowsFromMat(filter_);
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 filter_num_elem = filter_.NumRows()*filter_.NumCols();
    filter_.CopyRowsFromVec(params.Range(0, filter_num_elem));
  }

  std::string Info() const {
    return std::string("\n  filter") +
      MomentStatistics(filter_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }
  std::string InfoGradient() const {
    return std::string("\n  filter_grad") +
      MomentStatistics(filter_corr_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  void Prepare(const ExtraInfo &info) {
    position_.Resize(info.bposition.NumRows(), info.bposition.NumCols());
    position_.CopyFromMat(info.bposition);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // in       : T x D; T = sum(T1, T2, .., Tk), k sentence
    // filter_  : N x D; N is order_
    // position_: T x 1; auxiliary information
    out->VfsmnMemory(in, filter_, position_);
    // KALDI_LOG << in;
    // KALDI_LOG << filter_;
    // KALDI_LOG << position_;
    // KALDI_LOG << *out;
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // out_diff : T x D
    // filter_  : N x D
    // position_: T x 1
    in_diff->ComputeVfsmnHiddenDiff(out_diff, filter_, position_);
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    filter_.UpdateVfsmnFilter(diff, input, position_, -lr);
  }

  const CuMatrixBase<BaseFloat>& GetFilter() const { return filter_; }

  void SetFilter(const CuMatrixBase<BaseFloat>& filter) {
    KALDI_ASSERT(filter.NumRows() == filter_.NumRows());
    KALDI_ASSERT(filter.NumCols() == filter_.NumCols());
    filter_.CopyFromMat(filter);
  }

 protected:
  int32 order_;
  CuMatrix<BaseFloat> filter_;  // [a0, a1, ..., aN].T

  CuMatrix<BaseFloat> filter_corr_;
  CuMatrix<BaseFloat> position_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_COMPACT_VFSMN_H_
