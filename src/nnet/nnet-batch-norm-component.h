// nnet/nnet-batch-norm-component.h

// Copyright 2017  Kaituo XU

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


#ifndef KALDI_NNET_NNET_BATCH_NORM_COMPONENT_H_
#define KALDI_NNET_NNET_BATCH_NORM_COMPONENT_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {
namespace nnet1 {

class BatchNormComponent : public UpdatableComponent {
 public:
  BatchNormComponent(int32 input_dim, int32 output_dim):
    UpdatableComponent(input_dim, output_dim),
    mode_("train"), running_decay_rate_(0.9), var_floor_(1e-10)
  { }

  ~BatchNormComponent() 
  { }

  Component* Copy() const { return new BatchNormComponent(*this); }
  ComponentType GetType() const { return kBatchNormComponent; }

  void InitData(std::istream &is) {
    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<RunningDecayRate>") ReadBasicType(is, false, &running_decay_rate_);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (RunningDecayRate|LearnRateCoef|BiasLearnRateCoef)";
    }

    KALDI_ASSERT(InputDim() == OutputDim());
    KALDI_ASSERT(running_decay_rate_ >= 0.0);

    gamma_.Resize(InputDim());
    gamma_.Set(1.0);
    beta_.Resize(InputDim(), kSetZero);
    running_mean_.Resize(InputDim(), kSetZero);
    running_var_.Resize(InputDim(), kSetZero);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
          ReadBasicType(is, binary, &bias_learn_rate_coef_);
          break;
        case 'R': ExpectToken(is, binary, "<RunningDecayRate>");
          ReadBasicType(is, binary, &running_decay_rate_);
          break;
        default:
          std::string token;
          ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    // Read the data (data follow the tokens),
    gamma_.Read(is, binary);
    beta_.Read(is, binary);
    running_mean_.Read(is, binary);
    running_var_.Read(is, binary);

    KALDI_ASSERT(input_dim_ == output_dim_);
    KALDI_ASSERT(gamma_.Dim() == input_dim_);
    KALDI_ASSERT(beta_.Dim() == input_dim_);
    KALDI_ASSERT(running_mean_.Dim() == input_dim_);
    KALDI_ASSERT(running_var_.Dim() == input_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);

    WriteToken(os, binary, "<RunningDecayRate>");
    WriteBasicType(os, binary, running_decay_rate_);
    if (!binary) os << "\n";
    gamma_.Write(os, binary);
    beta_.Write(os, binary);
    running_mean_.Write(os, binary);
    running_var_.Write(os, binary);
  }

  int32 NumParams() const {
    return gamma_.Dim() + beta_.Dim();
  }

  void GetGradient(VectorBase<BaseFloat> *gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 gamma_num_elem = gamma_.Dim();
    gradient->Range(0, gamma_num_elem).CopyFromVec(gamma_corr_);
    gradient->Range(gamma_num_elem, beta_.Dim()).CopyFromVec(beta_corr_);
  }

  void GetParams(VectorBase<BaseFloat> *params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 gamma_num_elem = gamma_.Dim();
    params->Range(0, gamma_num_elem).CopyFromVec(gamma_);
    params->Range(gamma_num_elem, beta_.Dim()).CopyFromVec(beta_);
  }

  void SetParams(const VectorBase<BaseFloat> &params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 gamma_num_elem = gamma_.Dim();
    gamma_.CopyFromVec(params.Range(0, gamma_num_elem));
    beta_.CopyFromVec(params.Range(gamma_num_elem, beta_.Dim()));
  }

  std::string Info() const {
    return std::string("\n  gamma") + MomentStatistics(gamma_) +
      ", lr-coef " + ToString(learn_rate_coef_) +
      "\n  beta" + MomentStatistics(beta_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  std::string InfoGradient() const {
    return std::string("\n  gamma_grad") +
      MomentStatistics(gamma_corr_) +
      ", lr-coef " + ToString(learn_rate_coef_) +
      "\n  beta_grad" + MomentStatistics(beta_corr_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    if (mode_ == "train") {
      // default to zero
      CuMatrix<BaseFloat> tmp_var(in.NumRows(), in.NumCols());
      CuVector<BaseFloat> mean(input_dim_), var(input_dim_); 
      inv_std_buf_.Resize(input_dim_, kSetZero);
      normalized_input_buf_.Resize(in.NumRows(), in.NumCols(), kSetZero);

      // mean
      mean.AddRowSumMat(1.0, in, 1.0);
      mean.Scale(1.0 / in.NumRows());
      // variance
      tmp_var.CopyFromMat(in);
      tmp_var.AddVecToRows(-1.0, mean);
      tmp_var.ApplyPow(2.0);
      var.AddRowSumMat(1.0, tmp_var, 1.0);
      var.Scale(1.0 / in.NumRows());
      // inverse standard deviation
      inv_std_buf_.CopyFromVec(var);
      inv_std_buf_.Add(var_floor_);
      inv_std_buf_.ApplyPow(-0.5);
      // normalize input
      normalized_input_buf_.CopyFromMat(in);
      normalized_input_buf_.AddVecToRows(-1.0, mean);
      normalized_input_buf_.MulColsVec(inv_std_buf_);
      // output
      out->CopyFromMat(normalized_input_buf_);
      out->MulColsVec(gamma_);
      out->AddVecToRows(1.0, beta_);

      // update running mean
      running_mean_.Scale(running_decay_rate_);
      running_mean_.AddVec(1 - running_decay_rate_, mean);
      // update running varirance
      running_var_.Scale(running_decay_rate_);
      running_var_.AddVec(1 - running_decay_rate_, var);
    } else if (mode_ == "test") {
      // default is zero
      CuVector<BaseFloat> inv_std(input_dim_);

      inv_std.CopyFromVec(running_var_);
      inv_std.Add(var_floor_);
      inv_std.ApplyPow(-0.5);

      out->CopyFromMat(in);
      out->AddVecToRows(-1.0, running_mean_);
      out->MulColsVec(inv_std);
      out->MulColsVec(gamma_);
      out->AddVecToRows(1.0, beta_);
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // lazy initialization of udpate buffers,
    if (gamma_corr_.Dim() == 0) {
      gamma_corr_.Resize(input_dim_, kSetZero);
      beta_corr_.Resize(input_dim_, kSetZero);
    }
    const BaseFloat mmt = opts_.momentum;

    // normi --> normalized input
    // dnormi --> d_normalized_input
    CuMatrix<BaseFloat> normi(in.NumRows(), in.NumCols());
    CuMatrix<BaseFloat> dnormi(in.NumRows(), in.NumCols());
    CuMatrix<BaseFloat> dnormi_normi(in.NumRows(), in.NumCols());
    CuVector<BaseFloat> dnormi_mean(input_dim_);
    CuVector<BaseFloat> dnormi_normi_mean(input_dim_);

    dnormi.CopyFromMat(out_diff);
    dnormi.MulColsVec(gamma_);
    dnormi_mean.AddRowSumMat(1.0, dnormi, 1.0);
    dnormi_mean.Scale(1.0 / in.NumRows());

    dnormi_normi.CopyFromMat(dnormi);
    dnormi_normi.MulElements(normalized_input_buf_);
    dnormi_normi_mean.AddRowSumMat(1.0, dnormi_normi, 1.0);
    dnormi_normi_mean.Scale(1.0 / in.NumRows());
    
    normi.CopyFromMat(normalized_input_buf_);
    normi.MulColsVec(dnormi_normi_mean);

    in_diff->CopyFromMat(dnormi);
    in_diff->AddVecToRows(-1.0, dnormi_mean);
    in_diff->AddMat(-1.0, normi);
    in_diff->MulColsVec(inv_std_buf_);

    normalized_input_buf_.MulElements(out_diff);
    gamma_corr_.AddRowSumMat(1.0, normalized_input_buf_, mmt);
    beta_corr_.AddRowSumMat(1.0, out_diff, mmt);
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr = opts_.learn_rate;
    gamma_.AddVec(-lr * learn_rate_coef_, gamma_corr_);
    beta_.AddVec(-lr * learn_rate_coef_, beta_corr_);
  }

  void SetBatchNormMode(std::string mode) {
    mode_ = mode;
    KALDI_ASSERT(mode_ == "train" || mode_ == "test");
  }

 private:
  // "train" or "test"
  std::string mode_;

  // parameters, scale the input
  CuVector<BaseFloat> gamma_;
  // parameters, shift the input
  CuVector<BaseFloat> beta_;

  // gradient
  CuVector<BaseFloat> gamma_corr_;
  CuVector<BaseFloat> beta_corr_;

  // using in test mode
  CuVector<BaseFloat> running_mean_;
  CuVector<BaseFloat> running_var_;
  BaseFloat running_decay_rate_;

  // propagate buffers, using in backpropagation
  CuMatrix<BaseFloat> normalized_input_buf_;
  CuVector<BaseFloat> inv_std_buf_;
  BaseFloat var_floor_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_BATCH_NORM_COMPONENT_H_
