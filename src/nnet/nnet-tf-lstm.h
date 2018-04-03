// nnet/nnet-tf-lstm.h

// Copyright 2015-2016  Brno University of Technology (author: Karel Vesely)
// Copyright 2014  Jiayu DU (Jerry), Wei Li

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


#ifndef KALDI_NNET_NNET_TF_LSTM_H_
#define KALDI_NNET_NNET_TF_LSTM_H_

#include <string>
#include <vector>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 * g: squashing neuron near input
 * i: Input gate
 * f: Forget gate
 * o: Output gate
 * c: memory Cell (CEC)
 * n: squashing neuron near output
 * h: output neuron of Memory block
 * y: output neuron of LSTMP
 *************************************/

namespace kaldi {
namespace nnet1 {

class TfLstm : public MultistreamComponent {
 public:
  TfLstm(int32 input_dim, int32 output_dim):
    MultistreamComponent(input_dim, output_dim),
    cell_dim_(0),
    cell_clip_(50.0),
    diff_clip_(1.0),
    cell_diff_clip_(0.0),
    grad_clip_(250.0)
  { }

  ~TfLstm()
  { }

  Component* Copy() const { return new TfLstm(*this); }
  ComponentType GetType() const { return kTfLstm; }

  void InitData(std::istream &is) {
    // define options,
    float param_range = 0.1;
    // parse the line from prototype,
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<ParamRange>") ReadBasicType(is, false, &param_range);
      else if (token == "<CellDim>") ReadBasicType(is, false, &cell_dim_);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<CellClip>") ReadBasicType(is, false, &cell_clip_);
      else if (token == "<DiffClip>") ReadBasicType(is, false, &diff_clip_);
      else if (token == "<CellDiffClip>") ReadBasicType(is, false, &cell_diff_clip_);
      else if (token == "<GradClip>") ReadBasicType(is, false, &grad_clip_);
      else if (token == "<F>") ReadBasicType(is, false, &filter_size_);
      else if (token == "<S>") ReadBasicType(is, false, &stride_size_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamRange|CellDim|LearnRateCoef|BiasLearnRateCoef|CellClip|DiffClip|GradClip)";
    }

    // init the weights and biases (from uniform dist.),
    w_gifo_x_.Resize(4*cell_dim_, filter_size_, kUndefined);
    w_gifo_r_.Resize(4*cell_dim_, cell_dim_, kUndefined);
    w_gifo_k_.Resize(4*cell_dim_, cell_dim_, kUndefined);
    bias_.Resize(4*cell_dim_, kUndefined);
    peephole_i_c_.Resize(cell_dim_, kUndefined);
    peephole_f_c_.Resize(cell_dim_, kUndefined);
    peephole_o_c_.Resize(cell_dim_, kUndefined);
    //       (mean), (range)
    RandUniform(0.0, 2.0 * param_range, &w_gifo_x_);
    RandUniform(0.0, 2.0 * param_range, &w_gifo_r_);
    RandUniform(0.0, 2.0 * param_range, &w_gifo_k_);
    RandUniform(0.0, 2.0 * param_range, &bias_);
    RandUniform(0.0, 2.0 * param_range, &peephole_i_c_);
    RandUniform(0.0, 2.0 * param_range, &peephole_f_c_);
    RandUniform(0.0, 2.0 * param_range, &peephole_o_c_);

    KALDI_ASSERT(cell_dim_ > 0);
    KALDI_ASSERT(stride_size_ > 0);
    KALDI_ASSERT((InputDim() - filter_size_) % stride_size_ == 0);
    KALDI_ASSERT(OutputDim() == ((InputDim() - filter_size_) / stride_size_ + 1) * cell_dim_);
    KALDI_ASSERT(learn_rate_coef_ >= 0.0);
    KALDI_ASSERT(bias_learn_rate_coef_ >= 0.0);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      std::string token;
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'C': ReadToken(is, false, &token);
          /**/ if (token == "<CellDim>") ReadBasicType(is, binary, &cell_dim_);
          else if (token == "<CellClip>") ReadBasicType(is, binary, &cell_clip_);
          else if (token == "<CellDiffClip>") ReadBasicType(is, binary, &cell_diff_clip_);
          else if (token == "<ClipGradient>") ReadBasicType(is, binary, &grad_clip_); // bwd-compat.
          else KALDI_ERR << "Unknown token: " << token;
          break;
        case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
          ReadBasicType(is, binary, &bias_learn_rate_coef_);
          break;
        case 'D': ExpectToken(is, binary, "<DiffClip>");
          ReadBasicType(is, binary, &diff_clip_);
          break;
        case 'G': ExpectToken(is, binary, "<GradClip>");
          ReadBasicType(is, binary, &grad_clip_);
          break;
        case 'F': ExpectToken(is, binary, "<F>");
          ReadBasicType(is, binary, &filter_size_);
          break;
        case 'S': ExpectToken(is, binary, "<S>");
          ReadBasicType(is, binary, &stride_size_);
          break;
        default: ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    KALDI_ASSERT(cell_dim_ != 0);

    // Read the model parameters,
    w_gifo_x_.Read(is, binary);
    w_gifo_r_.Read(is, binary);
    w_gifo_k_.Read(is, binary);
    bias_.Read(is, binary);

    peephole_i_c_.Read(is, binary);
    peephole_f_c_.Read(is, binary);
    peephole_o_c_.Read(is, binary);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<CellDim>");
    WriteBasicType(os, binary, cell_dim_);
    WriteToken(os, binary, "<F>");
    WriteBasicType(os, binary, filter_size_);
    WriteToken(os, binary, "<S>");
    WriteBasicType(os, binary, stride_size_);

    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);

    WriteToken(os, binary, "<CellClip>");
    WriteBasicType(os, binary, cell_clip_);
    WriteToken(os, binary, "<DiffClip>");
    WriteBasicType(os, binary, diff_clip_);
    WriteToken(os, binary, "<CellDiffClip>");
    WriteBasicType(os, binary, cell_diff_clip_);
    WriteToken(os, binary, "<GradClip>");
    WriteBasicType(os, binary, grad_clip_);

    // write model parameters,
    if (!binary) os << "\n";
    w_gifo_x_.Write(os, binary);
    w_gifo_r_.Write(os, binary);
    w_gifo_k_.Write(os, binary);
    bias_.Write(os, binary);

    peephole_i_c_.Write(os, binary);
    peephole_f_c_.Write(os, binary);
    peephole_o_c_.Write(os, binary);
  }

  int32 NumParams() const {
    return ( w_gifo_x_.NumRows() * w_gifo_x_.NumCols() +
         w_gifo_r_.NumRows() * w_gifo_r_.NumCols() +
         w_gifo_k_.NumRows() * w_gifo_k_.NumCols() +
         bias_.Dim() +
         peephole_i_c_.Dim() +
         peephole_f_c_.Dim() +
         peephole_o_c_.Dim() );
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_gifo_x_corr_);

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_gifo_r_corr_);

    offset += len; len = w_gifo_k_.NumRows() * w_gifo_k_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_gifo_k_corr_);

    offset += len; len = bias_.Dim();
    gradient->Range(offset, len).CopyFromVec(bias_corr_);

    offset += len; len = peephole_i_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(peephole_i_c_corr_);

    offset += len; len = peephole_f_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(peephole_f_c_corr_);

    offset += len; len = peephole_o_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(peephole_o_c_corr_);

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_gifo_x_);

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_gifo_r_);

    offset += len; len = w_gifo_k_.NumRows() * w_gifo_k_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_gifo_k_);

    offset += len; len = bias_.Dim();
    params->Range(offset, len).CopyFromVec(bias_);

    offset += len; len = peephole_i_c_.Dim();
    params->Range(offset, len).CopyFromVec(peephole_i_c_);

    offset += len; len = peephole_f_c_.Dim();
    params->Range(offset, len).CopyFromVec(peephole_f_c_);

    offset += len; len = peephole_o_c_.Dim();
    params->Range(offset, len).CopyFromVec(peephole_o_c_);

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    w_gifo_x_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    w_gifo_r_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = w_gifo_k_.NumRows() * w_gifo_k_.NumCols();
    w_gifo_k_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = bias_.Dim();
    bias_.CopyFromVec(params.Range(offset, len));

    offset += len; len = peephole_i_c_.Dim();
    peephole_i_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = peephole_f_c_.Dim();
    peephole_f_c_.CopyFromVec(params.Range(offset, len));

    offset += len; len = peephole_o_c_.Dim();
    peephole_o_c_.CopyFromVec(params.Range(offset, len));

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  std::string Info() const {
    return std::string("cell-dim ") + ToString(cell_dim_) + " " +
      "filter size " + ToString(filter_size_) + " " +
      "stride size " + ToString(stride_size_) + " " +
      "( learn_rate_coef_ " + ToString(learn_rate_coef_) +
      ", bias_learn_rate_coef_ " + ToString(bias_learn_rate_coef_) +
      ", cell_clip_ " + ToString(cell_clip_) +
      ", diff_clip_ " + ToString(diff_clip_) +
      ", grad_clip_ " + ToString(grad_clip_) + " )" +
      "\n  w_gifo_x_  "   + MomentStatistics(w_gifo_x_) +
      "\n  w_gifo_r_  "   + MomentStatistics(w_gifo_r_) +
      "\n  w_gifo_k_  "   + MomentStatistics(w_gifo_k_) +
      "\n  bias_  "     + MomentStatistics(bias_) +
      "\n  peephole_i_c_  " + MomentStatistics(peephole_i_c_) +
      "\n  peephole_f_c_  " + MomentStatistics(peephole_f_c_) +
      "\n  peephole_o_c_  " + MomentStatistics(peephole_o_c_);
  }

  // TODO
  std::string InfoGradient() const {
    // // disassemble forward-propagation buffer into different neurons,
    // const CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    // // disassemble backpropagate buffer into different neurons,
    // const CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    // const CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(7*cell_dim_, proj_dim_));

    return std::string("") +
      "( learn_rate_coef_ " + ToString(learn_rate_coef_) +
      ", bias_learn_rate_coef_ " + ToString(bias_learn_rate_coef_) +
      ", cell_clip_ " + ToString(cell_clip_) +
      ", diff_clip_ " + ToString(diff_clip_) +
      ", grad_clip_ " + ToString(grad_clip_) + " )" +
      "\n  ### Gradients " +
      "\n  w_gifo_x_corr_  "   + MomentStatistics(w_gifo_x_corr_) +
      "\n  w_gifo_r_corr_  "   + MomentStatistics(w_gifo_r_corr_) +
      "\n  w_gifo_k_corr_  "   + MomentStatistics(w_gifo_k_corr_) +
      "\n  bias_corr_  "     + MomentStatistics(bias_corr_) +
      "\n  peephole_i_c_corr_  " + MomentStatistics(peephole_i_c_corr_) +
      "\n  peephole_f_c_corr_  " + MomentStatistics(peephole_f_c_corr_) +
      "\n  peephole_o_c_corr_  " + MomentStatistics(peephole_o_c_corr_);
      // "\n  ### Activations (mostly after non-linearities)" +
      // "\n  YI(0..1)^  " + MomentStatistics(YI) +
      // "\n  YF(0..1)^  " + MomentStatistics(YF) +
      // "\n  YO(0..1)^  " + MomentStatistics(YO) +
      // "\n  YG(-1..1)  " + MomentStatistics(YG) +
      // "\n  YC(-R..R)* " + MomentStatistics(YC) +
      // "\n  YH(-1..1)  " + MomentStatistics(YH) +
      // "\n  YM(-1..1)  " + MomentStatistics(YM) +
      // "\n  YR(-R..R)  " + MomentStatistics(YR) +
      // "\n  ### Derivatives (w.r.t. inputs of non-linearities)" +
      // "\n  DI^ " + MomentStatistics(DI) +
      // "\n  DF^ " + MomentStatistics(DF) +
      // "\n  DO^ " + MomentStatistics(DO) +
      // "\n  DG  " + MomentStatistics(DG) +
      // "\n  DC* " + MomentStatistics(DC) +
      // "\n  DH  " + MomentStatistics(DH) +
      // "\n  DM  " + MomentStatistics(DM) +
      // "\n  DR  " + MomentStatistics(DR);
  }

  /**
   * TODO: Do we really need this?
   */
  void ResetStreams(const std::vector<int32>& stream_reset_flag) {
    KALDI_ASSERT(NumStreams() == stream_reset_flag.size());
    if (prev_nnet_state_.NumRows() != stream_reset_flag.size()) {
      int32 B = (InputDim() - filter_size_) / stride_size_ + 1;
      prev_nnet_state_.Resize(NumStreams(), 8*(B+2)*cell_dim_, kSetZero);
    } else {
      for (int s = 0; s < NumStreams(); s++) {
        if (stream_reset_flag[s] == 1) {
          prev_nnet_state_.Row(s).SetZero();
        }
      }
    }
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {

    // reset context on each sentence if 'sequence_lengths_' not set
    // (happens in 'nnet-forward' or 'single-stream' training),
    if (sequence_lengths_.size() == 0) {
      ResetStreams(std::vector<int32>(1, 1));
    }

    KALDI_ASSERT(in.NumRows() % NumStreams() == 0);
    KALDI_ASSERT((in.NumCols() - filter_size_) % stride_size_ == 0);
    int32 T = in.NumRows() / NumStreams();
    int32 S = NumStreams();
    int32 B = (in.NumCols() - filter_size_) / stride_size_ + 1;

    // buffers,
    propagate_buf_.Resize((T+2)*S, 8*(B+2)*cell_dim_, kSetZero);
    if (prev_nnet_state_.NumRows() != NumStreams()) {
      prev_nnet_state_.Resize(NumStreams(), 8*(B+2)*cell_dim_, kSetZero); // lazy init,
    } else {
      propagate_buf_.RowRange(0, S).CopyFromMat(prev_nnet_state_); // use the 'previous-state',
    }

    // BufferPadding [T0]:dummy, [1, T]:current sequence, [T+1]:dummy
    for (int t = 1; t <= T; t++) {
      CuSubMatrix<BaseFloat> x_t(in.RowRange((t-1)*S, S));  // in is 0-based
      // multistream buffers for current time-step,
      CuSubMatrix<BaseFloat> y_all_t(propagate_buf_.RowRange(t*S, S));

      // [B0]:dummy, [1, B]: current block, [B+1]:dummy
      for (int b = 1; b <= B; b++) {
        // buffer
        CuSubMatrix<BaseFloat> y_all(propagate_buf_.ColRange(b*8*cell_dim_, 8*cell_dim_));
        CuSubMatrix<BaseFloat> y_all_ptb(y_all.RowRange((t-1)*S, S)); // pt means prev_t, i.e. t-1
        CuSubMatrix<BaseFloat> y_all_tb(y_all_t.ColRange(b*8*cell_dim_, 8*cell_dim_));
        CuSubMatrix<BaseFloat> y_all_tpb(y_all_t.ColRange((b-1)*8*cell_dim_, 8*cell_dim_)); // pb means prev block, i.e. b-1
        // split activations by neuron types,
        CuSubMatrix<BaseFloat> y_g(y_all_tb.ColRange(0*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_i(y_all_tb.ColRange(1*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_f(y_all_tb.ColRange(2*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_o(y_all_tb.ColRange(3*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_c(y_all_tb.ColRange(4*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_n(y_all_tb.ColRange(5*cell_dim_, cell_dim_));  // tanh(ct)
        CuSubMatrix<BaseFloat> y_h(y_all_tb.ColRange(6*cell_dim_, cell_dim_));  // output ht
        CuSubMatrix<BaseFloat> y_k(y_all_tb.ColRange(7*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_gifo(y_all_tb.ColRange(0, 4*cell_dim_));
        // inputs of block
        CuSubMatrix<BaseFloat> x_t_k(x_t.ColRange((b-1)*stride_size_, filter_size_));  // SxF
        CuSubMatrix<BaseFloat> block_ph(y_all_ptb.ColRange(6*cell_dim_, cell_dim_));  // ph = prev h, pt = prev t, SxH
        CuSubMatrix<BaseFloat> block_pc(y_all_ptb.ColRange(4*cell_dim_, cell_dim_));  // pc = prev c, pt = prev t, SxH
        CuSubMatrix<BaseFloat> block_pk(y_all_tpb.ColRange(7*cell_dim_, cell_dim_));  // pk = prev k, pb = prev b, SxH
        // TODO: replace above with below, don't need y_k actually
        // CuSubMatrix<BaseFloat> block_pk(y_all_tpb.ColRange(6*cell_dim_, cell_dim_));  // pk = prev k, pb = prev b, SxH

        y_gifo.AddMatMat(1.0, x_t_k, kNoTrans, w_gifo_x_, kTrans, 0.0);
        y_gifo.AddMatMat(1.0, block_ph, kNoTrans, w_gifo_r_, kTrans, 1.0);
        y_gifo.AddMatMat(1.0, block_pk, kNoTrans, w_gifo_k_, kTrans, 1.0);
        y_gifo.AddVecToRows(1.0, bias_);

        y_i.AddMatDiagVec(1.0, block_pc, kNoTrans, peephole_i_c_, 1.0);
        y_f.AddMatDiagVec(1.0, block_pc, kNoTrans, peephole_f_c_, 1.0);
        y_i.Sigmoid(y_i);
        y_f.Sigmoid(y_f);
        y_g.Tanh(y_g);

        y_c.AddMatMatElements(1.0, y_g, y_i, 0.0);
        y_c.AddMatMatElements(1.0, block_pc, y_f, 1.0);

        // if (cell_clip_ > 0.0) {
        //   y_c.ApplyFloor(-cell_clip_);   // optional clipping of cell activation,
        //   y_c.ApplyCeiling(cell_clip_);  // google paper Interspeech2014: LSTM for LVCSR
        // }

        y_o.AddMatDiagVec(1.0, y_c, kNoTrans, peephole_o_c_, 1.0);
        y_o.Sigmoid(y_o);

        y_n.Tanh(y_c);

        y_h.AddMatMatElements(1.0, y_n, y_o, 0.0);  // SxH

        y_k.CopyFromMat(y_h);  // TODO: use y_h to replace y_k

        // fill layer output
        out->RowRange((t-1)*S, S).ColRange((b-1)*cell_dim_, cell_dim_).CopyFromMat(y_h);
      }  // for b, block

      // set zeros to padded frames,
      if (sequence_lengths_.size() > 0) {
        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s]) {
            y_all_t.Row(s).SetZero();
          }
        }
      }
    }  // for t

    // the state in the last 'frame' is transferred (can be zero vector)
    prev_nnet_state_.CopyFromMat(propagate_buf_.RowRange(T*S, S));
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {

    // the number of sequences to be processed in parallel
    int32 T = in.NumRows() / NumStreams();
    int32 S = NumStreams();
    int32 B = (in.NumCols() - filter_size_) / stride_size_ + 1;

    // buffer,
    backpropagate_buf_.Resize((T+2)*S, 8*(B+2)*cell_dim_, kSetZero);

    for (int t = T; t >= 1; t--) {
      CuSubMatrix<BaseFloat> x_t(in.RowRange((t-1)*S, S));  // in is 0-based
      CuSubMatrix<BaseFloat> y_all_t(propagate_buf_.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> d_all_t(backpropagate_buf_.RowRange(t*S, S));
      for (int b = B; b >= 1; b--) {
        CuSubMatrix<BaseFloat> y_all(propagate_buf_.ColRange(b*8*cell_dim_, 8*cell_dim_));
        CuSubMatrix<BaseFloat> y_all_ptb(y_all.RowRange((t-1)*S, S));  // pt means prev t, t-1, b means frequency block
        CuSubMatrix<BaseFloat> y_all_ntb(y_all.RowRange((t+1)*S, S));  // nt means next t, t+1
        CuSubMatrix<BaseFloat> y_all_tb(y_all_t.ColRange(b*8*cell_dim_, 8*cell_dim_));
        CuSubMatrix<BaseFloat> y_all_tpb(y_all_t.ColRange((b-1)*8*cell_dim_, 8*cell_dim_)); // pb means prev block, b-1
        // split activations by neuron types,
        CuSubMatrix<BaseFloat> y_g(y_all_tb.ColRange(0*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_i(y_all_tb.ColRange(1*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_f(y_all_tb.ColRange(2*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_o(y_all_tb.ColRange(3*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_c(y_all_tb.ColRange(4*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_n(y_all_tb.ColRange(5*cell_dim_, cell_dim_));  // tanh(ct)
        CuSubMatrix<BaseFloat> y_h(y_all_tb.ColRange(6*cell_dim_, cell_dim_));  // output ht
        CuSubMatrix<BaseFloat> y_k(y_all_tb.ColRange(7*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> y_gifo(y_all_tb.ColRange(0, 4*cell_dim_));
        // inputs of block
        CuSubMatrix<BaseFloat> x_t_k(x_t.ColRange((b-1)*stride_size_, filter_size_));  // SxF
        CuSubMatrix<BaseFloat> block_ph(y_all_ptb.ColRange(6*cell_dim_, cell_dim_));  // ph = prev h, pt = prev t, SxH
        CuSubMatrix<BaseFloat> block_pc(y_all_ptb.ColRange(4*cell_dim_, cell_dim_));  // pc = prev c, pt = prev t, SxH
        CuSubMatrix<BaseFloat> block_pk(y_all_tpb.ColRange(7*cell_dim_, cell_dim_));  // pk = prev k, pb = prev b, SxH
        //
        CuSubMatrix<BaseFloat> y_f_ntb(y_all_ntb.ColRange(2*cell_dim_, cell_dim_));

        // backpropagation
        CuSubMatrix<BaseFloat> d_all(backpropagate_buf_.ColRange(b*8*cell_dim_, 8*cell_dim_));
        CuSubMatrix<BaseFloat> d_all_ntb(d_all.RowRange((t+1)*S, S)); // nt = next t, t+1
        CuSubMatrix<BaseFloat> d_all_tb(d_all_t.ColRange(b*8*cell_dim_, 8*cell_dim_));
        CuSubMatrix<BaseFloat> d_all_tnb(d_all_t.ColRange((b+1)*8*cell_dim_, 8*cell_dim_)); // nb = next b, b+1
        // split activations by neuron types,
        CuSubMatrix<BaseFloat> d_g(d_all_tb.ColRange(0*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> d_i(d_all_tb.ColRange(1*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> d_f(d_all_tb.ColRange(2*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> d_o(d_all_tb.ColRange(3*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> d_c(d_all_tb.ColRange(4*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> d_n(d_all_tb.ColRange(5*cell_dim_, cell_dim_));  // tanh(ct)
        CuSubMatrix<BaseFloat> d_h(d_all_tb.ColRange(6*cell_dim_, cell_dim_));  // output ht
        CuSubMatrix<BaseFloat> d_k(d_all_tb.ColRange(7*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> d_gifo(d_all_tb.ColRange(0, 4*cell_dim_));
        // 
        CuSubMatrix<BaseFloat> d_gifo_ntb(d_all_ntb.ColRange(0, 4*cell_dim_));
        CuSubMatrix<BaseFloat> d_i_ntb(d_all_ntb.ColRange(1*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> d_f_ntb(d_all_ntb.ColRange(2*cell_dim_, cell_dim_));
        CuSubMatrix<BaseFloat> d_c_ntb(d_all_ntb.ColRange(4*cell_dim_, cell_dim_));
        //
        CuSubMatrix<BaseFloat> d_gifo_tnb(d_all_tnb.ColRange(0, 4*cell_dim_));


        // out_diff: T*S x B*H
        d_h.CopyFromMat(out_diff.RowRange((t-1)*S, S).ColRange((b-1)*cell_dim_, cell_dim_));
        d_h.AddMatMat(1.0, d_gifo_ntb, kNoTrans, w_gifo_r_, kNoTrans, 1.0);
        d_h.AddMatMat(1.0, d_gifo_tnb, kNoTrans, w_gifo_k_, kNoTrans, 1.0);

        d_n.AddMatMatElements(1.0, d_h, y_o, 0.0);
        d_n.DiffTanh(y_n, d_n);

        d_o.AddMatMatElements(1.0, d_h, y_n, 0.0);
        d_o.DiffSigmoid(y_o, d_o);

        d_c.AddMat(1.0, d_n);
        d_c.AddMatMatElements(1.0, d_c_ntb, y_f_ntb, 1.0);
        d_c.AddMatDiagVec(1.0, d_i_ntb, kNoTrans, peephole_i_c_, 1.0);
        d_c.AddMatDiagVec(1.0, d_f_ntb, kNoTrans, peephole_f_c_, 1.0);
        d_c.AddMatDiagVec(1.0, d_o,     kNoTrans, peephole_o_c_, 1.0);

        // if (cell_diff_clip_ > 0.0) {
        //   d_c.ApplyFloor(-cell_diff_clip_);
        //   d_c.ApplyCeiling(cell_diff_clip_);
        // }

        d_f.AddMatMatElements(1.0, d_c, block_pc, 0.0);
        d_f.DiffSigmoid(y_f, d_f);

        d_i.AddMatMatElements(1.0, d_c, y_g, 0.0);
        d_i.DiffSigmoid(y_i, d_i);

        d_g.AddMatMatElements(1.0, d_c, y_i, 0.0);
        d_g.DiffTanh(y_g, d_g);

        // if (diff_clip_ > 0.0) {
        //   d_gifo.ApplyFloor(-diff_clip_);
        //   d_gifo.ApplyCeiling(diff_clip_);
        // }

        // lazy initialization of udpate buffers,
        if (w_gifo_x_corr_.NumRows() == 0) {
          w_gifo_x_corr_.Resize(4*cell_dim_, filter_size_, kSetZero);
          w_gifo_r_corr_.Resize(4*cell_dim_, cell_dim_, kSetZero);
          w_gifo_k_corr_.Resize(4*cell_dim_, cell_dim_, kSetZero);
          bias_corr_.Resize(4*cell_dim_, kSetZero);
          peephole_i_c_corr_.Resize(cell_dim_, kSetZero);
          peephole_f_c_corr_.Resize(cell_dim_, kSetZero);
          peephole_o_c_corr_.Resize(cell_dim_, kSetZero);
        }
    
        // in_diff: T*S x D
        in_diff->RowRange((t-1)*S, S).ColRange((b-1)*stride_size_, filter_size_).AddMatMat(1.0, d_gifo, kNoTrans, w_gifo_x_, kNoTrans, 1.0);

        // calculate delta
        BaseFloat mmt = 1.0;
        if (t == T && b == B) {
            mmt = opts_.momentum;
        }

        w_gifo_x_corr_.AddMatMat(1.0, d_gifo, kTrans,    x_t_k, kNoTrans, mmt);
        w_gifo_r_corr_.AddMatMat(1.0, d_gifo, kTrans, block_ph, kNoTrans, mmt);
        w_gifo_k_corr_.AddMatMat(1.0, d_gifo, kTrans, block_pk, kNoTrans, mmt);
        bias_corr_.AddRowSumMat(1.0, d_gifo, mmt);
        peephole_i_c_corr_.AddDiagMatMat(1.0, d_i, kTrans, block_pc, kNoTrans, mmt);
        peephole_f_c_corr_.AddDiagMatMat(1.0, d_f, kTrans, block_pc, kNoTrans, mmt);
        peephole_o_c_corr_.AddDiagMatMat(1.0, d_o, kTrans,      y_c, kNoTrans, mmt);
      }  // for b, block

      if (sequence_lengths_.size() > 0) {
        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s]) {
            d_all_t.Row(s).SetZero();
          }
        }
      }
    }  // for t, time
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {

    // apply the gradient clipping,
    if (grad_clip_ > 0.0) {
      w_gifo_x_corr_.ApplyFloor(-grad_clip_);
      w_gifo_x_corr_.ApplyCeiling(grad_clip_);
      w_gifo_r_corr_.ApplyFloor(-grad_clip_);
      w_gifo_r_corr_.ApplyCeiling(grad_clip_);
      w_gifo_k_corr_.ApplyFloor(-grad_clip_);
      w_gifo_k_corr_.ApplyCeiling(grad_clip_);
      bias_corr_.ApplyFloor(-grad_clip_);
      bias_corr_.ApplyCeiling(grad_clip_);
      peephole_i_c_corr_.ApplyFloor(-grad_clip_);
      peephole_i_c_corr_.ApplyCeiling(grad_clip_);
      peephole_f_c_corr_.ApplyFloor(-grad_clip_);
      peephole_f_c_corr_.ApplyCeiling(grad_clip_);
      peephole_o_c_corr_.ApplyFloor(-grad_clip_);
      peephole_o_c_corr_.ApplyCeiling(grad_clip_);
    }

    const BaseFloat lr  = opts_.learn_rate;

    w_gifo_x_.AddMat(-lr * learn_rate_coef_, w_gifo_x_corr_);
    w_gifo_r_.AddMat(-lr * learn_rate_coef_, w_gifo_r_corr_);
    w_gifo_k_.AddMat(-lr * learn_rate_coef_, w_gifo_k_corr_);
    bias_.AddVec(-lr * bias_learn_rate_coef_, bias_corr_, 1.0);

    peephole_i_c_.AddVec(-lr * bias_learn_rate_coef_, peephole_i_c_corr_, 1.0);
    peephole_f_c_.AddVec(-lr * bias_learn_rate_coef_, peephole_f_c_corr_, 1.0);
    peephole_o_c_.AddVec(-lr * bias_learn_rate_coef_, peephole_o_c_corr_, 1.0);
  }

 private:
  // dims
  int32 cell_dim_;
  int32 filter_size_;  // F
  int32 stride_size_;  // S

  BaseFloat cell_clip_;  ///< Clipping of 'cell-values' in forward pass (per-frame),
  BaseFloat diff_clip_;  ///< Clipping of 'derivatives' in backprop (per-frame),
  BaseFloat cell_diff_clip_; ///< Clipping of 'cell-derivatives' accumulated over CEC (per-frame),
  BaseFloat grad_clip_;  ///< Clipping of the updates,

  // buffer for transfering state across batches,
  CuMatrix<BaseFloat> prev_nnet_state_;

  // feed-forward connections: from x to [g, i, f, o]
  CuMatrix<BaseFloat> w_gifo_x_;
  CuMatrix<BaseFloat> w_gifo_x_corr_;

  // recurrent projection connections: from r to [g, i, f, o]
  CuMatrix<BaseFloat> w_gifo_r_;
  CuMatrix<BaseFloat> w_gifo_r_corr_;

  // frequency recurrent connections
  CuMatrix<BaseFloat> w_gifo_k_;
  CuMatrix<BaseFloat> w_gifo_k_corr_;

  // biases of [g, i, f, o]
  CuVector<BaseFloat> bias_;
  CuVector<BaseFloat> bias_corr_;

  // peephole from c to i, f, g
  // peephole connections are block-internal, so we use vector form
  CuVector<BaseFloat> peephole_i_c_;
  CuVector<BaseFloat> peephole_f_c_;
  CuVector<BaseFloat> peephole_o_c_;

  CuVector<BaseFloat> peephole_i_c_corr_;
  CuVector<BaseFloat> peephole_f_c_corr_;
  CuVector<BaseFloat> peephole_o_c_corr_;

  // propagate buffer: output of [g, i, f, o, c, n, h, k] (n = tanh(ct))
  // Horizontal layout:
  // |   b = 0  |   b = 1  | ... |   b = B  |   b = B+1|
  // | gifocnhk | gifocnhk | ... | gifocnhk | gifocnhk |
  // Vertical layout:
  // -------
  // t=0 s=1
  // t=0 s=2
  // t=0 s=...
  // t=0 s=S
  // -------
  // t=1 s=1
  // t=1 s=2
  // t=1 s=...
  // t=1 s=S
  // -------
  CuMatrix<BaseFloat> propagate_buf_;

  // back-propagate buffer: diff-input of [g, i, f, o, c, n, h, k]
  // layout like propagate_buf_
  CuMatrix<BaseFloat> backpropagate_buf_;
};  // class TfLstm

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_TF_LSTM_H_
