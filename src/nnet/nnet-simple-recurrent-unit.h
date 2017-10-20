// nnet/nnet-simple-recurrent-unit.h

// Copyright 2017  Kaituo XU
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


#ifndef KALDI_NNET_NNET_SIMPLE_RECURRENT_UNIT_H_
#define KALDI_NNET_NNET_SIMPLE_RECURRENT_UNIT_H_

#include <string>
#include <vector>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 *************************************/

namespace kaldi {
namespace nnet1 {

class SimpleRecurrentUnit : public MultistreamComponent {
 public:
  SimpleRecurrentUnit(int32 input_dim, int32 output_dim):
    MultistreamComponent(input_dim, output_dim),
    cell_dim_(0),
    // cell_clip_(50.0),
    // diff_clip_(1.0),
    // cell_diff_clip_(0.0),
    grad_clip_(250.0)
  { }

  ~SimpleRecurrentUnit()
  { }

  Component* Copy() const { return new SimpleRecurrentUnit(*this); }
  ComponentType GetType() const { return kSimpleRecurrentUnit; }

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
      // else if (token == "<CellClip>") ReadBasicType(is, false, &cell_clip_);
      // else if (token == "<DiffClip>") ReadBasicType(is, false, &diff_clip_);
      // else if (token == "<CellDiffClip>") ReadBasicType(is, false, &cell_diff_clip_);
      else if (token == "<GradClip>") ReadBasicType(is, false, &grad_clip_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamRange|CellDim|LearnRateCoef|BiasLearnRateCoef|CellClip|DiffClip|GradClip)";
    }

    // init the weights and biases (from uniform dist.),
    w_xfrh_.Resize(4*cell_dim_, input_dim_, kUndefined);
    bias_f_.Resize(cell_dim_, kUndefined);
    bias_r_.Resize(cell_dim_, kUndefined);
    //       (mean), (range)
    RandUniform(0.0, 2.0 * param_range, &w_xfrh_);
    RandUniform(0.0, 2.0 * param_range, &bias_f_);
    RandUniform(0.0, 2.0 * param_range, &bias_r_);

    KALDI_ASSERT(cell_dim_ > 0);
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
          // else if (token == "<CellClip>") ReadBasicType(is, binary, &cell_clip_);
          // else if (token == "<CellDiffClip>") ReadBasicType(is, binary, &cell_diff_clip_);
          else KALDI_ERR << "Unknown token: " << token;
          break;
        case 'L': ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
          ReadBasicType(is, binary, &bias_learn_rate_coef_);
          break;
        // case 'D': ExpectToken(is, binary, "<DiffClip>");
        //   ReadBasicType(is, binary, &diff_clip_);
        //   break;
        case 'G': ExpectToken(is, binary, "<GradClip>");
          ReadBasicType(is, binary, &grad_clip_);
          break;
        default: ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    KALDI_ASSERT(cell_dim_ != 0);

    // Read the model parameters,
    w_xfrh_.Read(is, binary);
    bias_f_.Read(is, binary);
    bias_r_.Read(is, binary);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<CellDim>");
    WriteBasicType(os, binary, cell_dim_);

    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);

    // WriteToken(os, binary, "<CellClip>");
    // WriteBasicType(os, binary, cell_clip_);
    // WriteToken(os, binary, "<DiffClip>");
    // WriteBasicType(os, binary, diff_clip_);
    // WriteToken(os, binary, "<CellDiffClip>");
    // WriteBasicType(os, binary, cell_diff_clip_);
    WriteToken(os, binary, "<GradClip>");
    WriteBasicType(os, binary, grad_clip_);

    // write model parameters,
    if (!binary) os << "\n";
    w_xfrh_.Write(os, binary);
    bias_f_.Write(os, binary);
    bias_r_.Write(os, binary);
  }

  int32 NumParams() const {
    return ( w_xfrh_.NumRows() * w_xfrh_.NumCols() +
             bias_f_.Dim() + bias_r_.Dim() );
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    /*
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_gifo_x_corr_);

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_gifo_r_corr_);

    offset += len; len = bias_.Dim();
    gradient->Range(offset, len).CopyFromVec(bias_corr_);

    offset += len; len = peephole_i_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(peephole_i_c_corr_);

    offset += len; len = peephole_f_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(peephole_f_c_corr_);

    offset += len; len = peephole_o_c_.Dim();
    gradient->Range(offset, len).CopyFromVec(peephole_o_c_corr_);

    offset += len; len = w_r_m_.NumRows() * w_r_m_.NumCols();
    gradient->Range(offset, len).CopyRowsFromMat(w_r_m_corr_);

    offset += len;
    KALDI_ASSERT(offset == NumParams());
    */
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    /*
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_gifo_x_);

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_gifo_r_);

    offset += len; len = bias_.Dim();
    params->Range(offset, len).CopyFromVec(bias_);

    offset += len; len = peephole_i_c_.Dim();
    params->Range(offset, len).CopyFromVec(peephole_i_c_);

    offset += len; len = peephole_f_c_.Dim();
    params->Range(offset, len).CopyFromVec(peephole_f_c_);

    offset += len; len = peephole_o_c_.Dim();
    params->Range(offset, len).CopyFromVec(peephole_o_c_);

    offset += len; len = w_r_m_.NumRows() * w_r_m_.NumCols();
    params->Range(offset, len).CopyRowsFromMat(w_r_m_);

    offset += len;
    KALDI_ASSERT(offset == NumParams());
    */
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 offset, len;

    offset = 0;    len = w_xfrh_.NumRows() * w_xfrh_.NumCols();
    w_xfrh_.CopyRowsFromVec(params.Range(offset, len));

    offset += len; len = bias_f_.Dim();
    bias_f_.CopyFromVec(params.Range(offset, len));

    offset += len; len = bias_r_.Dim();
    bias_r_.CopyFromVec(params.Range(offset, len));

    offset += len;
    KALDI_ASSERT(offset == NumParams());
  }

  std::string Info() const {
    return std::string("cell-dim ") + ToString(cell_dim_) + " " +
      "( learn_rate_coef_ " + ToString(learn_rate_coef_) +
      ", bias_learn_rate_coef_ " + ToString(bias_learn_rate_coef_) +
      // ", cell_clip_ " + ToString(cell_clip_) +
      // ", diff_clip_ " + ToString(diff_clip_) +
      ", grad_clip_ " + ToString(grad_clip_) + " )" +
      "\n  w_xfrh_  "   + MomentStatistics(w_xfrh_) +
      "\n  bias_f_  "     + MomentStatistics(bias_f_) +
      "\n  bias_r_  "     + MomentStatistics(bias_r_);
  }

  std::string InfoGradient() const {
    // disassemble forward-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> AX(propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat>  F(propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat>  R(propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> AH(propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat>  C(propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> GC(propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat>  H(propagate_buf_.ColRange(6*cell_dim_, cell_dim_));

    // split derivatives by neuron types,
    const CuSubMatrix<BaseFloat> DAX(backpropagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat>  DF(backpropagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat>  DR(backpropagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DAH(backpropagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat>  DC(backpropagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat> DGC(backpropagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    const CuSubMatrix<BaseFloat>  DH(backpropagate_buf_.ColRange(6*cell_dim_, cell_dim_));

    return std::string("") +
      "( learn_rate_coef_ " + ToString(learn_rate_coef_) +
      ", bias_learn_rate_coef_ " + ToString(bias_learn_rate_coef_) +
      // ", cell_clip_ " + ToString(cell_clip_) +
      // ", diff_clip_ " + ToString(diff_clip_) +
      ", grad_clip_ " + ToString(grad_clip_) + " )" +
      "\n  ### Gradients " +
      "\n  w_xfrh_corr_  "   + MomentStatistics(w_xfrh_corr_) +
      "\n  bias_f_corr_  "     + MomentStatistics(bias_f_corr_) +
      "\n  bias_r_corr_  "     + MomentStatistics(bias_r_corr_);
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
      prev_nnet_state_.Resize(NumStreams(), 7*cell_dim_, kSetZero);
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
    int32 T = in.NumRows() / NumStreams();
    int32 S = NumStreams();

    // buffers
    propagate_buf_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);
    if (prev_nnet_state_.NumRows() != NumStreams()) {
      prev_nnet_state_.Resize(NumStreams(), 7*cell_dim_, kSetZero); // lazy init,
    } else {
      propagate_buf_.RowRange(0, S).CopyFromMat(prev_nnet_state_); // use the 'previous-state',
    }

    // split activations by neuron types,
    CuSubMatrix<BaseFloat> AX(propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  F(propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  R(propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> AH(propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  C(propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> GC(propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  H(propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> XFRH(propagate_buf_.ColRange(0, 4*cell_dim_));

    // x -> x', f, r, h', not recurrent, do it all in once
    XFRH.RowRange(1*S, T*S).AddMatMat(1.0, in, kNoTrans, w_xfrh_, kTrans, 0.0);

    F.RowRange(1*S, T*S).AddVecToRows(1.0, bias_f_);
    R.RowRange(1*S, T*S).AddVecToRows(1.0, bias_r_);

    // BufferPadding [T0]:dummy, [1, T]:current sequence, [T+1]:dummy
    for (int t = 1; t <= T; t++) {
      // multistream buffers for current time-step,
      CuSubMatrix<BaseFloat> all(propagate_buf_.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> ax(AX.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  f( F.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  r( R.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> ah(AH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  c( C.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> gc(GC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  h( H.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> xfrh(XFRH.RowRange(t*S, S));

      f.Sigmoid(f);
      r.Sigmoid(r);
      // c_t = f_t * c_t-1 + (1 - f_t) * ax_t = ax - f_t * ax_t + f_t * c_t-1
      c.AddMat(1.0, ax);
      c.AddMatMatElements(-1.0, f, ax, 1.0);
      c.AddMatMatElements(1.0, f, C.RowRange((t-1)*S, S), 1.0);
      // gc_t = c_t
      gc.AddMat(1.0, c);
      // h_t = r_t * gc_t + (1 - r_t) * ah_t = ah_t - r_t * ah + r_t * gc_t
      h.AddMat(1.0, ah);
      h.AddMatMatElements(-1.0, r, ah, 1.0);
      h.AddMatMatElements(1.0, r, gc, 1.0);
      
      // set zeros to padded frames,
      if (sequence_lengths_.size() > 0) {
        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s]) {
            all.Row(s).SetZero();
          }
        }
      }
    }  // for loop

    out->CopyFromMat(H.RowRange(1*S, T*S));

    // the state in the last 'frame' is transferred (can be zero vector)
    prev_nnet_state_.CopyFromMat(propagate_buf_.RowRange(T*S, S));
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // TODO: clip gradient

    // the number of sequences to be processed in parallel
    int32 T = in.NumRows() / NumStreams();
    int32 S = NumStreams();

    // buffer
    backpropagate_buf_.Resize((T+2)*S, 7 * cell_dim_, kSetZero);

    // split activations by neuron types,
    CuSubMatrix<BaseFloat> AX(propagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  F(propagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  R(propagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> AH(propagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  C(propagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> GC(propagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  H(propagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> XFRH(propagate_buf_.ColRange(0, 4*cell_dim_));

    // split derivatives by neuron types,
    CuSubMatrix<BaseFloat> DAX(backpropagate_buf_.ColRange(0*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  DF(backpropagate_buf_.ColRange(1*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  DR(backpropagate_buf_.ColRange(2*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DAH(backpropagate_buf_.ColRange(3*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  DC(backpropagate_buf_.ColRange(4*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DGC(backpropagate_buf_.ColRange(5*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat>  DH(backpropagate_buf_.ColRange(6*cell_dim_, cell_dim_));
    CuSubMatrix<BaseFloat> DXFRH(backpropagate_buf_.ColRange(0, 4*cell_dim_));

    // pre-copy partial derivatives from the SRU output,
    DH.RowRange(1*S, T*S).CopyFromMat(out_diff);

    // BufferPadding [T0]:dummy, [1,T]:current sequence, [T+1]: dummy,
    for (int t = T; t >= 1; t--) {
      CuSubMatrix<BaseFloat> ax(AX.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  f( F.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  r( R.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> ah(AH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  c( C.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> gc(GC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  h( H.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> xfrh(XFRH.RowRange(t*S, S));

      CuSubMatrix<BaseFloat> dall(backpropagate_buf_.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> dax(DAX.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  df( DF.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  dr( DR.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> dah(DAH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  dc( DC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> dgc(DGC.RowRange(t*S, S));
      CuSubMatrix<BaseFloat>  dh( DH.RowRange(t*S, S));
      CuSubMatrix<BaseFloat> dxfrh(DXFRH.RowRange(t*S, S));

      // dr = dh * (gc - ah) = dh * gc - dh * ah
      dr.AddMatMatElements(1.0, dh, gc, 1.0);
      dr.AddMatMatElements(-1.0, dh, ah, 1.0);
      // dgc = dh * r
      dgc.AddMatMatElements(1.0, dh, r, 1.0);
      // dah = dh * (1 - r) = dh - dh * r
      dah.AddMat(1.0, dh);
      dah.AddMatMatElements(-1.0, dh, r, 1.0);
      // dc_t = dc_t+1 * f_t+1
      // dc_t += dgc
      dc.AddMatMatElements(1.0, DC.RowRange((t+1)*S, S), F.RowRange((t+1)*S, S), 1.0);
      dc.AddMat(1.0, dgc);
      // df = dc_t * (c_t-1 - ax) = dc_t * c_t-1 - dc_t * ax
      df.AddMatMatElements(1.0, dc, C.RowRange((t-1)*S, S), 1.0);
      df.AddMatMatElements(-1.0, dc, ax, 1.0);
      // dax = dc_t * (1 - f) = dc_t - dc_t * f
      dax.AddMat(1.0, dc);
      dax.AddMatMatElements(-1.0, dc, f, 1.0);
      // dr = dar = dr * (1 - r) * r 
      dr.DiffSigmoid(r, dr);
      // df = daf = df * (1 -f ) * f
      df.DiffSigmoid(f, df);

      // set zeros to padded frames,
      if (sequence_lengths_.size() > 0) {
        for (int s = 0; s < S; s++) {
          if (t > sequence_lengths_[s]) {
            dall.Row(s).SetZero();
          }
        }
      }
    }  // for loop

    // dx = dxfrh.dot(W)
    in_diff->AddMatMat(1.0, DXFRH.RowRange(1*S, T*S), kNoTrans, w_xfrh_, kNoTrans, 0.0);

    // lazy initialization of udpate buffers,
    if (w_xfrh_corr_.NumRows() == 0) {
      w_xfrh_corr_.Resize(4*cell_dim_, input_dim_, kSetZero);
      bias_f_corr_.Resize(cell_dim_, kSetZero);
      bias_r_corr_.Resize(cell_dim_, kSetZero);
    }

    // calculate delta
    const BaseFloat mmt = opts_.momentum;

    // dW = np.dot(x2d.T, dxfrh)
    w_xfrh_corr_.AddMatMat(1.0, DXFRH.RowRange(1 * S, T * S), kTrans, 
                                in                          , kNoTrans, mmt);
    bias_f_corr_.AddRowSumMat(1.0, DF.RowRange(1*S, T*S), mmt);
    bias_r_corr_.AddRowSumMat(1.0, DR.RowRange(1*S, T*S), mmt);
  }

  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {

    // apply the gradient clipping
    if (grad_clip_ > 0.0) {
      w_xfrh_corr_.ApplyFloor(-grad_clip_);
      w_xfrh_corr_.ApplyCeiling(grad_clip_);
      bias_f_corr_.ApplyFloor(-grad_clip_);
      bias_f_corr_.ApplyCeiling(grad_clip_);
      bias_r_corr_.ApplyFloor(-grad_clip_);
      bias_r_corr_.ApplyCeiling(grad_clip_);
    }
    
    const BaseFloat lr  = opts_.learn_rate;

    w_xfrh_.AddMat(-lr * learn_rate_coef_, w_xfrh_corr_);
    bias_f_.AddVec(-lr * bias_learn_rate_coef_, bias_f_corr_, 1.0);
    bias_r_.AddVec(-lr * bias_learn_rate_coef_, bias_r_corr_, 1.0);
  }

 private:
  // dims
  int32 cell_dim_;

  // buffer for transfering state across batches,
  CuMatrix<BaseFloat> prev_nnet_state_;

  // feed-forward connections: from x to [x', f, r, h']
  CuMatrix<BaseFloat> w_xfrh_;
  CuMatrix<BaseFloat> w_xfrh_corr_;
  
  // biases of f
  CuVector<BaseFloat> bias_f_;
  CuVector<BaseFloat> bias_f_corr_;

  // biases of r
  CuVector<BaseFloat> bias_r_;
  CuVector<BaseFloat> bias_r_corr_;

  // propagate buffer: output of [x'(ax), f(f), r(r), h'(ah), c(c), g(gc), h]
  CuMatrix<BaseFloat> propagate_buf_;

  // back-propagate buffer: diff-input of 
  CuMatrix<BaseFloat> backpropagate_buf_;

  // gradient-clipping value,
  BaseFloat grad_clip_;
};  // class SimpleRecurrentUnit

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_SIMPLE_RECURRENT_UNIT_H
