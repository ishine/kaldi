// nnet/nnet-deep-fsmn.h

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


#ifndef KALDI_NNET_NNET_DEEP_FSMN_H_
#define KALDI_NNET_NNET_DEEP_FSMN_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

/****************************************************************
 *
 * DeepFsmn = [input -> Affine+ReLU -> Affine -> vFSMN -> output]
 *               |                                 ^
 *               |---------------------------------|
 *
 * input: the output of last vFSMN layer
 ****************************************************************/

class DeepFsmn : public UpdatableComponent {
 public:
  DeepFsmn(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out)
  { }
  ~DeepFsmn()
  { }

  Component* Copy() const { return new DeepFsmn(*this); }
  ComponentType GetType() const { return kDeepFsmn; }

  void InitData(std::istream &is) {
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<BackOrder>")  ReadBasicType(is, false, &lookback_order_);
      else if (token == "<AheadOrder>") ReadBasicType(is, false, &lookahead_order_);
      else if (token == "<HiddenSize>") ReadBasicType(is, false, &hid_size_);
      else if (token == "<LearnRateCoef>")  ReadBasicType(is, false, &learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (BackOrder|AheadOrder|HiddenSize|LearnRateCoef)";
    }

    //
    // Initialize trainable parameters,
    //
    // Glorot Uniform
    // 1. hidden layer
    float mean = 0.0;
    float range = sqrt(6.0/(hid_size_ + input_dim_));
    Wh_.Resize(hid_size_, input_dim_, kSetZero);
    bh_.Resize(hid_size_, kSetZero);
    RandUniform(mean, range, &Wh_);

    // 2. projection layer
    range = sqrt(6.0/(output_dim_ + hid_size_));
    Wp_.Resize(output_dim_, hid_size_, kSetZero);
    RandUniform(mean, range, &Wp_);

    // 3. vfsmn layer
    bfilter_.Resize(lookback_order_ + 1, output_dim_);  // NOTE +1
    range = sqrt(6.0 / (float)(output_dim_ + lookback_order_ + 1));
    RandUniform(mean, range, &bfilter_);

    ffilter_.Resize(lookahead_order_, output_dim_);
    range = sqrt(6.0 / (float)(output_dim_ + lookahead_order_));
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
        case 'H': ExpectToken(is, binary, "<HiddenSize>");
          ReadBasicType(is, binary, &hid_size_);
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

    // weight
    Wh_.Read(is, binary);
    bh_.Read(is, binary);
    Wp_.Read(is, binary);
    bfilter_.Read(is, binary);
    ffilter_.Read(is, binary);

    KALDI_ASSERT(Wh_.NumRows() == hid_size_);
    KALDI_ASSERT(Wh_.NumCols() == input_dim_);
    KALDI_ASSERT(bh_.Dim() == hid_size_);
    KALDI_ASSERT(Wp_.NumRows() == output_dim_);
    KALDI_ASSERT(Wp_.NumCols() == hid_size_);
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
    WriteToken(os, binary, "<HiddenSize>");
    WriteBasicType(os, binary, hid_size_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    if (!binary) os << "\n";
    // weight
    Wh_.Write(os, binary);
    bh_.Write(os, binary);
    Wp_.Write(os, binary);
    bfilter_.Write(os, binary);
    ffilter_.Write(os, binary);
  }

  int32 NumParams() const {
    return Wh_.NumRows() * Wh_.NumCols() +
           bh_.Dim() +
           Wp_.NumRows() * Wp_.NumCols() +
           bfilter_.NumRows() * bfilter_.NumCols() + 
           ffilter_.NumRows() * ffilter_.NumCols();
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 Wh_num = Wh_.NumRows() * Wh_.NumCols();
    int32 bh_num = bh_.Dim();
    int32 Wp_num = Wp_.NumRows() * Wp_.NumCols();
    int32 bf_num = bfilter_.NumRows() * bfilter_.NumCols();
    int32 ff_num = ffilter_.NumRows() * ffilter_.NumCols();

    int32 offset = 0;
    gradient->Range(offset, Wh_num).CopyRowsFromMat(dWh_);
    offset += Wh_num;
    gradient->Range(offset, bh_num).CopyFromVec(dbh_);
    offset += bh_num;
    gradient->Range(offset, Wp_num).CopyRowsFromMat(dWp_);
    offset += Wp_num;
    // fake gradient
    gradient->Range(offset, bf_num).Set(0.0);
    offset += bf_num;
    gradient->Range(offset, ff_num).Set(0.0);
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 Wh_num = Wh_.NumRows() * Wh_.NumCols();
    int32 bh_num = bh_.Dim();
    int32 Wp_num = Wp_.NumRows() * Wp_.NumCols();
    int32 bf_num = bfilter_.NumRows() * bfilter_.NumCols();
    int32 ff_num = ffilter_.NumRows() * ffilter_.NumCols();

    int32 offset = 0;
    params->Range(offset, Wh_num).CopyRowsFromMat(Wh_);
    offset += Wh_num;
    params->Range(offset, bh_num).CopyFromVec(bh_);
    offset += bh_num;
    params->Range(offset, Wp_num).CopyRowsFromMat(Wp_);
    offset += Wp_num;
    params->Range(offset, bf_num).CopyRowsFromMat(bfilter_);
    offset += bf_num;
    params->Range(offset, ff_num).CopyRowsFromMat(ffilter_);
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 Wh_num = Wh_.NumRows() * Wh_.NumCols();
    int32 bh_num = bh_.Dim();
    int32 Wp_num = Wp_.NumRows() * Wp_.NumCols();
    int32 bf_num = bfilter_.NumRows() * bfilter_.NumCols();
    int32 ff_num = ffilter_.NumRows() * ffilter_.NumCols();

    int32 offset = 0;
    Wh_.CopyRowsFromVec(params.Range(offset, Wh_num));
    offset += Wh_num;
    bh_.CopyFromVec(params.Range(offset, bh_num));
    offset += bh_num;
    Wp_.CopyRowsFromVec(params.Range(offset, Wp_num));
    offset += Wp_num;
    bfilter_.CopyRowsFromVec(params.Range(offset, bf_num));
    offset += bf_num;
    ffilter_.CopyRowsFromVec(params.Range(offset, ff_num));
  }

  std::string Info() const {
    return std::string("\n  hidden layer W") +
      MomentStatistics(Wh_) +
      std::string("\n  hidden layer b") +
      MomentStatistics(bh_) +
      std::string("\n  projection layer W") +
      MomentStatistics(Wp_) +
      std::string("\n  lookback filter") +
      MomentStatistics(bfilter_) +
      std::string("\n  lookahead filter") +
      MomentStatistics(ffilter_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  std::string InfoGradient() const {
    return std::string("\n  hidden layer dW") +
      MomentStatistics(dWh_) +
      std::string("\n  hidden layer db") +
      MomentStatistics(dbh_) +
      std::string("\n  projection layer dW") +
      MomentStatistics(dWp_) +
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
    // 1: hidden layer     (AffineTransform + ReLU)
    //    AffineTransform: h_out_ = in * Wh_.T + bh_
    h_out_.Resize(in.NumRows(), hid_size_, kSetZero);
    h_out_.AddVecToRows(1.0, bh_, 0.0);
    h_out_.AddMatMat(1.0, in, kNoTrans, Wh_, kTrans, 1.0);
    //    ReLU:            h_out_ = ReLU(h_out_)
    h_out_.ApplyFloor(0.0);

    // 2: projection layer (AffineTransform)
    //                     p_out_ = h_out_ * Wp_.T
    p_out_.Resize(in.NumRows(), output_dim_, kSetZero);
    p_out_.AddMatMat(1.0, h_out_, kNoTrans, Wp_, kTrans, 1.0);

    // 3: vfsmn layer      (BiCompactVfsmn)
    //                     out = vFSMN(p_out_, bfilter_, ffilter_)
    out->BiVfsmnMemory(p_out_, bfilter_, ffilter_, bposition_, fposition_);

    // 4. skip connection
    //                     out = out + in
    out->AddMat(1.0, in, kNoTrans);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;

    // lazy initialization of gradient
    if (dWh_.NumRows() == 0) {
      dWh_.Resize(hid_size_, input_dim_, kSetZero);
      dbh_.Resize(hid_size_, kSetZero);
      dWp_.Resize(output_dim_, hid_size_, kSetZero);
    }

    // 1. vfsmn layer
    dp_out_.Resize(out.NumRows(), output_dim_, kSetZero);
    dp_out_.BiComputeVfsmnHiddenDiff(out_diff, bfilter_, ffilter_,
                                     bposition_, fposition_);
    bfilter_.BiUpdateVfsmnBackfilter(out_diff, p_out_, bposition_, -lr);
    ffilter_.BiUpdateVfsmnAheadfilter(out_diff, p_out_, fposition_, -lr);

    // 2. projection layer
    dh_out_.Resize(out.NumRows(), hid_size_, kSetZero);
    dh_out_.AddMatMat(1.0, dp_out_, kNoTrans, Wp_, kNoTrans, 0.0);
    dWp_.AddMatMat(1.0, dp_out_, kTrans, h_out_, kNoTrans, mmt);

    // 3. hidden layer
    //    ReLU
    h_out_.ApplyHeaviside();
    dh_out_.MulElements(h_out_);
    //    AffineTransform
    in_diff->AddMatMat(1.0, dh_out_, kNoTrans, Wh_, kNoTrans, 0.0);
    dWh_.AddMatMat(1.0, dh_out_, kTrans, in, kNoTrans, mmt);
    dbh_.AddRowSumMat(1.0, dh_out_, mmt);

    // 4. skip connection
    in_diff->AddMat(1.0, out_diff, kNoTrans);
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
    Wp_.AddMat(-lr, dWp_);
    Wh_.AddMat(-lr, dWh_);
    bh_.AddVec(-lr, dbh_);
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
  // 1: hidden layer     (AffineTransform + ReLU)
  //    weight matrix & gradient
  CuMatrix<BaseFloat> Wh_;
  CuVector<BaseFloat> bh_;
  CuMatrix<BaseFloat> dWh_;
  CuVector<BaseFloat> dbh_;
  //    hidden output & gradient
  CuMatrix<BaseFloat> h_out_;
  CuMatrix<BaseFloat> dh_out_;

  // 2: projection layer (AffineTransform)
  //    weight matrix & gradient
  CuMatrix<BaseFloat> Wp_;
  CuMatrix<BaseFloat> dWp_;
  //    projection output & gradient
  CuMatrix<BaseFloat> p_out_;
  CuMatrix<BaseFloat> dp_out_;

  // 3: vfsmn layer      (BiCompactVfsmn)
  //    weight matrix
  CuMatrix<BaseFloat> bfilter_;  // (N1+1) rows, [a0, a1, ..., aN1].T, b means look backward
  CuMatrix<BaseFloat> ffilter_;  // (N2) rows,   [c0, c1, ..., cN2].T, f means look forward
  //    auxiliary information for CUDA kernel function
  CuMatrix<BaseFloat> bposition_;  // Tx1
  CuMatrix<BaseFloat> fposition_;  // Tx1

  // provideed by nnet.proto
  int32 hid_size_;
  int32 lookback_order_;   // N1 (do not include current frame)
  int32 lookahead_order_;  // N2
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_DEEP_FSMN_H_
