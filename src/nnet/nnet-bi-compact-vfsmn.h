// nnet/nnet-bi-compact-vfsmn.h

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


#ifndef KALDI_NNET_NNET_BI_COMPACT_VFSMN_H_
#define KALDI_NNET_NNET_BI_COMPACT_VFSMN_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class BiCompactVfsmn : public UpdatableComponent {
 public:
  BiCompactVfsmn(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out)
  { }
  ~BiCompactVfsmn()
  { }

  Component* Copy() const { return new BiCompactVfsmn(*this); }
  ComponentType GetType() const { return kBiCompactVfsmn; }

  void InitData(std::istream &is) {
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<BackOrder>")  ReadBasicType(is, false, &lookback_order_);
      else if (token == "<AheadOrder>") ReadBasicType(is, false, &lookahead_order_);
      else if (token == "<LearnRateCoef>")  ReadBasicType(is, false, &learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (LookbackOrder|LookaheadOrder|LearnRateCoef)";
    }

    //
    // Initialize trainable parameters,
    //
    // Glorot Uniform
    bfilter_.Resize(lookback_order_ + 1, InputDim());  // NOTE +1
    float mean = 0.0;
    float range = sqrt(6.0 / (float)(InputDim() + lookback_order_ + 1));
    RandUniform(mean, range, &bfilter_);

    ffilter_.Resize(lookahead_order_, InputDim());
    range = sqrt(6.0 / (float)(InputDim() + lookahead_order_));
    RandUniform(mean, range, &ffilter_);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'B': ExpectToken(is, binary, "<BackOrder>");
          ReadBasicType(is, binary, &lookback_order_);
          break;
        case 'A': ExpectToken(is, binary, "<AheadOrder>");
          ReadBasicType(is, binary, &lookahead_order_);
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
    bfilter_.Read(is, binary);
    ffilter_.Read(is, binary);

    KALDI_ASSERT(bfilter_.NumRows() == lookback_order_ + 1);
    KALDI_ASSERT(bfilter_.NumCols() == input_dim_);
    KALDI_ASSERT(ffilter_.NumRows() == lookahead_order_);
    KALDI_ASSERT(ffilter_.NumCols() == input_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<BackOrder>");
    WriteBasicType(os, binary, lookback_order_);
    WriteToken(os, binary, "<AheadOrder>");
    WriteBasicType(os, binary, lookahead_order_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    if (!binary) os << "\n";
    // filter
    bfilter_.Write(os, binary);
    ffilter_.Write(os, binary);
  }

  int32 NumParams() const {
    return bfilter_.NumRows() * bfilter_.NumCols() + 
           ffilter_.NumRows() * ffilter_.NumCols();
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 bfilter_num_elem = bfilter_.NumRows() * bfilter_.NumCols();
    gradient->Range(0, bfilter_num_elem).CopyRowsFromMat(bfilter_corr_);
    int32 ffilter_num_elem = ffilter_.NumRows() * ffilter_.NumCols();
    gradient->Range(bfilter_num_elem, ffilter_num_elem).CopyRowsFromMat(ffilter_corr_);
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 bfilter_num_elem = bfilter_.NumRows() * bfilter_.NumCols();
    params->Range(0, bfilter_num_elem).CopyRowsFromMat(bfilter_);
    int32 ffilter_num_elem = ffilter_.NumRows() * ffilter_.NumCols();
    params->Range(bfilter_num_elem, ffilter_num_elem).CopyRowsFromMat(ffilter_);
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 bfilter_num_elem = bfilter_.NumRows() * bfilter_.NumCols();
    bfilter_.CopyRowsFromVec(params.Range(0, bfilter_num_elem));
    int32 ffilter_num_elem = ffilter_.NumRows() * ffilter_.NumCols();
    ffilter_.CopyRowsFromVec(params.Range(bfilter_num_elem, ffilter_num_elem));
  }

  std::string Info() const {
    return std::string("\n  lookback filter") +
      MomentStatistics(bfilter_) +
      std::string("\n  lookahead filter") +
      MomentStatistics(ffilter_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }
  std::string InfoGradient() const {
    return std::string("\n  lookback filter_grad") +
      MomentStatistics(bfilter_corr_) +
      std::string("\n  lookahead filter_grad") +
      MomentStatistics(ffilter_corr_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  void Prepare(const ExtraInfo &info) {
    bposition_.Resize(info.bposition.NumRows(), info.bposition.NumCols());
    bposition_.CopyFromMat(info.bposition);
    fposition_.Resize(info.fposition.NumRows(), info.fposition.NumCols());
    fposition_.CopyFromMat(info.fposition);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    // in        : T x D; T = sum(T1, T2, .., Tk), k sentence
    // bfilter_  : (N1+1) x D; N1 is lookback order 
    // ffilter_  : N2 x D; N2 is lookahead order
    // bposition_: T x 1; auxiliary information
    // fposition_: T x 1; auxiliary information
    // KALDI_LOG << in;
    // KALDI_LOG << bfilter_;
    // KALDI_LOG << ffilter_;
    // KALDI_LOG << bposition_;
    // KALDI_LOG << fposition_;
    out->BiVfsmnMemory(in, bfilter_, ffilter_, bposition_, fposition_);
    // KALDI_LOG << *out;
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // out_diff : T x D
    // filter_  : N x D
    // position_: T x 1
    in_diff->BiComputeVfsmnHiddenDiff(out_diff, bfilter_, ffilter_,
                                      bposition_, fposition_);
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    // TODO: Just compute gradient here
    bfilter_.BiUpdateVfsmnBackfilter(diff, input, bposition_, -lr);
    ffilter_.BiUpdateVfsmnAheadfilter(diff, input, fposition_, -lr);
  }

  const CuMatrixBase<BaseFloat>& GetBackfilter() const { return bfilter_; }

  const CuMatrixBase<BaseFloat>& GetAheadfilter() const { return ffilter_; }

  void SetBackfilter(const CuMatrixBase<BaseFloat>& filter) {
    KALDI_ASSERT(filter.NumRows() == bfilter_.NumRows());
    KALDI_ASSERT(filter.NumCols() == bfilter_.NumCols());
    bfilter_.CopyFromMat(filter);
  }

  void SetAheadfilter(const CuMatrixBase<BaseFloat>& filter) {
    KALDI_ASSERT(filter.NumRows() == ffilter_.NumRows());
    KALDI_ASSERT(filter.NumCols() == ffilter_.NumCols());
    ffilter_.CopyFromMat(filter);
  }

 protected:
  int32 lookback_order_;   // N1 (do not include current frame)
  int32 lookahead_order_;  // N2
  // weight matrix
  CuMatrix<BaseFloat> bfilter_;  // (N1+1) rows, [a0, a1, ..., aN1].T, b means look backward
  CuMatrix<BaseFloat> ffilter_;  // (N2) rows,   [c0, c1, ..., cN2].T, f means look forward

  // weight gradient
  CuMatrix<BaseFloat> bfilter_corr_;
  CuMatrix<BaseFloat> ffilter_corr_;
  // auxiliary information for CUDA kernel function
  CuMatrix<BaseFloat> bposition_;
  CuMatrix<BaseFloat> fposition_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_BI_COMPACT_VFSMN_H_
