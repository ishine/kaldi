// nnet0/nnet-rnnt-join-transform.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)
// Copyright 2018-2019  Shanghai Jiao Tong University (author: Wei Deng)

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


#ifndef KALDI_NNET_NNET_RNNT_JOIN_TRANSFORM_H_
#define KALDI_NNET_NNET_RNNT_JOIN_TRANSFORM_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"


namespace kaldi {

namespace lm {
class LmModelSync;
}

namespace nnet0 {

class RNNTJoinTransform : public UpdatableComponent {

	friend class NnetModelSync;
	friend class lm::LmModelSync;

 public:
	RNNTJoinTransform(int32 dim_in_, int32 dim_out_)
    : UpdatableComponent(dim_in_, dim_out_),
      learn_rate_coef_(1.0), bias_learn_rate_coef_(1.0), max_norm_(0.0)
  { }
  ~RNNTJoinTransform()
  { }

  Component* Copy() const { return new RNNTJoinTransform(*this); }
  ComponentType GetType() const { return kRNNTJoinTransform; }
  
  void InitData(std::istream &is) {
    // define options
    float bias_mean = -2.0, bias_range = 2.0, param_stddev = 0.1, param_range = 0.0;
    float learn_rate_coef = 1.0, bias_learn_rate_coef = 1.0;
    float max_norm = 0.0;
    int32 encoder_dim = 128, predict_dim = 128, join_dim = 1024;
    int xavier_flag = 0;
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<ParamStddev>") ReadBasicType(is, false, &param_stddev);
      else if (token == "<ParamRange>")   ReadBasicType(is, false, &param_range);
      else if (token == "<BiasMean>")    ReadBasicType(is, false, &bias_mean);
      else if (token == "<BiasRange>")   ReadBasicType(is, false, &bias_range);
      else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
      else if (token == "<BiasLearnRateCoef>") ReadBasicType(is, false, &bias_learn_rate_coef);
      else if (token == "<MaxNorm>") ReadBasicType(is, false, &max_norm);
      else if (token == "<EncoderDim>") ReadBasicType(is, false, &encoder_dim);
      else if (token == "<PredictDim>") ReadBasicType(is,false,&predict_dim);
      else if (token == "<JoinDim>") ReadBasicType(is,false,&predict_dim);
      else if (token == "<Xavier>") ReadBasicType(is, false, &xavier_flag);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (ParamStddev|BiasMean|BiasRange|LearnRateCoef|BiasLearnRateCoef)";
      is >> std::ws; // eat-up whitespace
    }

    int32 max_dim = encoder_dim > predict_dim ? encoder_dim : predict_dim;
    KALDI_ASSERT(max_dim == input_dim_);
    //
    // Initialize trainable parameters,
    // if Xavier_flag=1, use the “Xavier” initialization
    if(xavier_flag){
      // Uniform,
      float re = sqrt(6)/sqrt(join_dim + encoder_dim),
            rp = sqrt(6)/sqrt(join_dim + predict_dim),
            rj = sqrt(6)/sqrt(output_dim_ + join_dim);
      encoder_linearity_.Resize(join_dim, encoder_dim, kSetZero);
      predict_linearity_.Resize(join_dim, predict_dim, kSetZero);
      join_linearity_.Resize(output_dim_, join_dim, kSetZero);
      RandUniform(0.0, re, &encoder_linearity_);
      RandUniform(0.0, rp, &predict_linearity_);
      RandUniform(0.0, rj, &join_linearity_);
      bias_.Resize(join_dim, kSetZero);
      join_bias_.Resize(output_dim_, kSetZero);
    } else if (param_range != 0.0) {
      // Uniform,
      encoder_linearity_.Resize(join_dim, encoder_dim, kSetZero);
      predict_linearity_.Resize(join_dim, predict_dim, kSetZero);
      join_linearity_.Resize(output_dim_, join_dim, kSetZero);
      RandUniform(0.0, param_range, &encoder_linearity_);
      RandUniform(0.0, param_range, &predict_linearity_);
      RandUniform(0.0, param_range, &join_linearity_);
      bias_.Resize(join_dim, kSetZero);
      join_bias_.Resize(output_dim_, kSetZero);
      RandUniform(bias_mean, bias_range, &bias_);
      RandUniform(bias_mean, bias_range, &join_bias_);
    } else {
      // Gaussian with given std_dev (mean = 0),
      encoder_linearity_.Resize(join_dim, encoder_dim, kSetZero);
      predict_linearity_.Resize(join_dim, predict_dim, kSetZero);
      join_linearity_.Resize(output_dim_, join_dim, kSetZero);
      RandGauss(0.0, param_stddev, &encoder_linearity_);
      RandGauss(0.0, param_stddev, &predict_linearity_);
      RandGauss(0.0, param_stddev, &join_linearity_);
      // Uniform,
      bias_.Resize(join_dim, kSetZero);
      join_bias_.Resize(output_dim_, kSetZero);
      RandUniform(bias_mean, bias_range, &bias_);
      RandUniform(bias_mean, bias_range, &join_bias_);
    }

    learn_rate_coef_ = learn_rate_coef;
    bias_learn_rate_coef_ = bias_learn_rate_coef;
    max_norm_ = max_norm;
    encoder_dim_ = encoder_dim;
    predict_dim_ = predict_dim;
    join_dim_ = join_dim;
  }

  void ReadData(std::istream &is, bool binary) {
    // optional learning-rate coefs
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<LearnRateCoef>");
      ReadBasicType(is, binary, &learn_rate_coef_);
      ExpectToken(is, binary, "<BiasLearnRateCoef>");
      ReadBasicType(is, binary, &bias_learn_rate_coef_);
    }
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<MaxNorm>");
      ReadBasicType(is, binary, &max_norm_);

      ExpectToken(is,binary, "<EncoderDim>");
      ReadBasicType(is, binary, &encoder_dim_);
      ExpectToken(is,binary, "<PredictDim>");
      ReadBasicType(is, binary, &predict_dim_);
      ExpectToken(is,binary, "<JoinDim>");
      ReadBasicType(is, binary, &join_dim_);
    }

    // weights
    encoder_linearity_.Read(is, binary);
    predict_linearity_.Read(is, binary);
    bias_.Read(is, binary);

    join_linearity_.Read(is, binary);
    join_bias_.Read(is, binary);

    KALDI_ASSERT(encoder_dim_ == encoder_linearity_.NumCols());
    KALDI_ASSERT(predict_dim_ == predict_linearity_.NumCols());
    KALDI_ASSERT(encoder_linearity_.NumRows() == predict_linearity_.NumRows());
    KALDI_ASSERT(join_dim_ == encoder_linearity_.NumRows());
    KALDI_ASSERT(join_dim_ == bias_.Dim());
    KALDI_ASSERT(join_dim_ == join_linearity_.NumCols());

    int32 max_dim = encoder_dim_ > predict_dim_ ? encoder_dim_ : predict_dim_;
    KALDI_ASSERT(max_dim == input_dim_);
    KALDI_ASSERT(join_linearity_.NumRows() == output_dim_);
    KALDI_ASSERT(join_bias_.Dim() == output_dim_);

    // init delta buffers
    encoder_linearity_corr_.Resize(join_dim_, encoder_dim_, kSetZero);
    predict_linearity_corr_.Resize(join_dim_, predict_dim_, kSetZero);
    bias_corr_.Resize(join_dim_, kSetZero);
    join_linearity_corr_.Resize(output_dim_, join_dim_, kSetZero);
    join_bias_corr_.Resize(output_dim_, kSetZero);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<MaxNorm>");
    WriteBasicType(os, binary, max_norm_);

    WriteToken(os, binary, "<EncoderDim>");
	WriteBasicType(os, binary, encoder_dim_);
    WriteToken(os, binary, "<PredictDim>");
	WriteBasicType(os, binary, predict_dim_);
    WriteToken(os, binary, "<JoinDim>");
	WriteBasicType(os, binary, join_dim_);


    // weights
	encoder_linearity_.Write(os, binary);
	predict_linearity_.Write(os, binary);
	bias_.Write(os, binary);

	join_linearity_.Write(os, binary);
	join_bias_.Write(os, binary);
  }

  int32 NumParams() const {
	return ( encoder_linearity_.NumRows() * encoder_linearity_.NumCols() +
		predict_linearity_.NumRows() * predict_linearity_.NumCols() +
		bias_.Dim() +
		join_linearity_.NumRows() * join_linearity_.NumCols() +
		join_bias_.Dim() );
  }
  
  int32 GetDim() const {
	return ( encoder_linearity_.SizeInBytes()/sizeof(BaseFloat) +
		predict_linearity_.SizeInBytes()/sizeof(BaseFloat) +
		bias_.Dim() +
		join_linearity_.SizeInBytes()/sizeof(BaseFloat) +
		join_bias_.Dim() );
  }

  void GetParams(Vector<BaseFloat>* wei_copy) const {
	wei_copy->Resize(NumParams());

	int32 offset, len;

	offset = 0;
	len = encoder_linearity_.NumRows() * encoder_linearity_.NumCols();
	wei_copy->Range(offset, len).CopyRowsFromMat(encoder_linearity_);

	offset += len;
	len = predict_linearity_.NumRows() * predict_linearity_.NumCols();
	wei_copy->Range(offset, len).CopyRowsFromMat(predict_linearity_);

    offset += len;
    len = bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(bias_);

	offset += len;
	len = join_linearity_.NumRows() * join_linearity_.NumCols();
	wei_copy->Range(offset, len).CopyRowsFromMat(join_linearity_);

    offset += len;
    len = join_bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(join_bias_);
    return;
  }
  
  std::string Info() const {
	return std::string("  ") +
	  "\n  encoder_linearity_  "   + MomentStatistics(encoder_linearity_) +
	  "\n  predict_linearity_  "   + MomentStatistics(predict_linearity_) +
	  "\n  bias_  "     + MomentStatistics(bias_) +
	  "\n  join_linearity_  " + MomentStatistics(join_linearity_) +
	  "\n  join_bias_  "    + MomentStatistics(join_bias_);
  }

  std::string InfoGradient() const {
	return std::string("  ") +
	  "\n  Gradients:" +
	  "\n  encoder_linearity_corr_  "   + MomentStatistics(encoder_linearity_corr_) +
	  "\n  predict_linearity_corr_  "   + MomentStatistics(predict_linearity_corr_) +
	  "\n  bias_corr_  "     + MomentStatistics(bias_corr_) +
	  "\n  join_linearity_corr_  "   + MomentStatistics(join_linearity_corr_) +
	  "\n  join_bias_corr_  "     + MomentStatistics(join_bias_corr_);
  }


  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
	  int32 num_frames = in.NumRows();
	  int en_frames = nstream_*maxT_;
	  int pre_frames = nstream_*maxU_;

	  KALDI_ASSERT(en_frames+pre_frames == num_frames);
	  out->AddVecToRows(1.0, join_bias_, 0.0);

	  CuSubMatrix<BaseFloat> encoder_in(in, 0, en_frames, 0, encoder_dim_);
	  CuSubMatrix<BaseFloat> predict_in(in, en_frames, pre_frames, 0, predict_dim_);
	  Ah_t_.Resize(encoder_in.NumRows(), encoder_linearity_.NumRows(), kUndefined);
	  Bp_u_.Resize(predict_in.NumRows(), predict_linearity_.NumRows(), kUndefined);
	  h_t_u_.Resize(nstream_*maxT_*maxU_, join_dim_, kUndefined);

	  Ah_t_.AddMatMat(1.0, encoder_in, kNoTrans, encoder_linearity_, kTrans, 1.0);
	  Bp_u_.AddMatMat(1.0, predict_in, kNoTrans, predict_linearity_, kTrans, 1.0);
	  for(int t = 0; t < maxT_; t++) {
		  CuSubMatrix<BaseFloat> T(Ah_t_, t*nstream_, nstream_, 0, join_dim_);
		  for (int u = 0; u < maxU_; u++) {
			  CuSubMatrix<BaseFloat> U(Bp_u_, u*nstream_, nstream_, 0, join_dim_);
			  CuSubMatrix<BaseFloat> TU(h_t_u_, (t*maxU_+u)*nstream_, nstream_, 0, join_dim_);
			  TU.CopyFromMat(T);
			  TU.AddMat(1.0, U);
		  }
	  }
	  h_t_u_.AddVecToRows(1.0, bias_, 1.0);
	  h_t_u_.Tanh(h_t_u_);

	  // z_t_u
	  out->AddMatMat(1.0, h_t_u_, kNoTrans, join_linearity_, kTrans, 1.0);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
	  int en_frames = nstream_*maxT_;
	  int pre_frames = nstream_*maxU_;

	  // z_t_u -> h_t_u
	  d_h_t_u_.Resize(nstream_*maxT_*maxU_, join_dim_, kUndefined);
	  d_h_t_u_.AddMatMat(1.0, out_diff, kNoTrans, join_linearity_, kNoTrans, 0.0);

	  // tanh
	  d_h_t_u_.DiffTanh(h_t_u_, d_h_t_u_);

	  int in_dim = encoder_dim_ > predict_dim_ ? encoder_dim_ : predict_dim_;
	  join_diff_.Resize(en_frames+pre_frames, join_dim_, kSetZero);
	  in_diff_.Resize(en_frames+pre_frames, in_dim, kUndefined);

	  // T deriv
	  for(int t = 0; t < maxT_; t++) {
		  CuSubMatrix<BaseFloat> D_T(join_diff_, t*nstream_, nstream_, 0, join_dim_);
		  for (int u = 0; u < maxU_; u++) {
			  CuSubMatrix<BaseFloat> D_TU(d_h_t_u_, (t*maxU_+u)*nstream_, nstream_, 0, join_dim_);
			  D_T.AddMat(1.0, D_TU);
		  }
	  }

	  // U deriv
	  for(int u = 0; u < maxU_; u++) {
		  CuSubMatrix<BaseFloat> D_U(join_diff_, (u+maxT_)*nstream_, nstream_, 0, join_dim_);
		  for (int t = 0; t < maxT_; t++) {
			  CuSubMatrix<BaseFloat> D_TU(d_h_t_u_, (t*maxU_+u)*nstream_, nstream_, 0, join_dim_);
			  D_U.AddMat(1.0, D_TU);
		  }
	  }

	  // multiply error derivative by weights
	  CuSubMatrix<BaseFloat> d_join_encoder(join_diff_, 0, en_frames, 0, join_dim_);
	  CuSubMatrix<BaseFloat> d_join_predict(join_diff_, en_frames, pre_frames, 0, join_dim_);
	  CuSubMatrix<BaseFloat> d_encoder(in_diff_, 0, en_frames, 0, encoder_dim_);
	  CuSubMatrix<BaseFloat> d_predict(in_diff_, en_frames, pre_frames, 0, predict_dim_);
	  in_diff_.AddMatMat(1.0, d_join_encoder, kNoTrans, encoder_linearity_, kNoTrans, 0.0);
	  in_diff_.AddMatMat(1.0, d_join_predict, kNoTrans, predict_linearity_, kNoTrans, 0.0);
  }

  void ResetGradient() {
		encoder_linearity_corr_.SetZero();
		predict_linearity_corr_.SetZero();
		bias_corr_.SetZero();
		join_linearity_corr_.SetZero();
		join_bias_corr_.SetZero();
  }

  void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
		// we use following hyperparameters from the option class
		const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
		const BaseFloat lr_bias = opts_.learn_rate * bias_learn_rate_coef_;
		const BaseFloat mmt = opts_.momentum;
		const BaseFloat l2 = opts_.l2_penalty;
		const BaseFloat l1 = opts_.l1_penalty;
		// we will also need the number of frames in the mini-batch
		const int32 num_frames = input.NumRows();
		int en_frames = nstream_*maxT_;
		int pre_frames = nstream_*maxU_;
		local_lrate = -lr;
		local_lrate_bias = -lr_bias;

		// compute gradient (incl. momentum)
		join_linearity_corr_.AddMatMat(1.0, diff, kTrans, h_t_u_, kNoTrans, mmt);
		join_bias_corr_.AddRowSumMat(1.0, diff, mmt);

		CuSubMatrix<BaseFloat> encoder_in(input, 0, en_frames, 0, encoder_dim_);
		CuSubMatrix<BaseFloat> predict_in(input, en_frames, pre_frames, 0, predict_dim_);
		CuSubMatrix<BaseFloat> d_join_encoder(join_diff_, 0, en_frames, 0, join_dim_);
		CuSubMatrix<BaseFloat> d_join_predict(join_diff_, en_frames, pre_frames, 0, join_dim_);
		encoder_linearity_corr_.AddMatMat(1.0, d_join_encoder, kTrans, encoder_in, kNoTrans, mmt);
		predict_linearity_corr_.AddMatMat(1.0, d_join_predict, kTrans, predict_in, kNoTrans, mmt);
		bias_corr_.AddRowSumMat(1.0, join_diff_, mmt);

		// l2 regularization
		if (l2 != 0.0) {
			encoder_linearity_.AddMat(-lr*l2*en_frames, encoder_linearity_);
			predict_linearity_.AddMat(-lr*l2*pre_frames, predict_linearity_);
			join_linearity_.AddMat(-lr*l2*num_frames, join_linearity_);
		}
		// l1 regularization
		if (l1 != 0.0) {
			cu::RegularizeL1(&encoder_linearity_, &encoder_linearity_corr_, lr*l1*en_frames, lr);
			cu::RegularizeL1(&predict_linearity_, &predict_linearity_corr_, lr*l1*pre_frames, lr);
			cu::RegularizeL1(&join_linearity_, &join_linearity_corr_, lr*l1*num_frames, lr);
		}
  }

  void UpdateGradient() {
		// update
		encoder_linearity_.AddMat(local_lrate, encoder_linearity_corr_);
		predict_linearity_.AddMat(local_lrate, predict_linearity_corr_);
		bias_.AddVec(local_lrate_bias, bias_corr_);

		join_linearity_.AddMat(local_lrate, join_linearity_corr_);
		join_bias_.AddVec(local_lrate_bias, join_bias_corr_);
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
	  // not implemented yet
  }

  void SetRNNTStreamSize(const std::vector<int32>& encoder_utt_frames,
		  const std::vector<int32>& predict_utt_words, int maxT, int maxU) {
	  encoder_utt_frames_ = encoder_utt_frames;
	  predict_utt_words_ = predict_utt_words;
	  maxT_ = maxT;
	  maxU_ = maxU;
	  nstream_ = encoder_utt_frames_.size();
	  KALDI_ASSERT(encoder_utt_frames_.size() == predict_utt_words_.size());
  }

  int32 OutputRow(int32 in_row) {
	  KALDI_ASSERT((maxT_+maxU_)*nstream_ == in_row);
      return nstream_*maxT_*maxU_; 
  }

  CuMatrix<BaseFloat> GetInputDiff() {
	  return in_diff_;
  }

  int WeightCopy(void *host, int direction, int copykind) {
#if HAVE_CUDA == 1
  if (CuDevice::Instantiate().Enabled()) {
        CuTimer tim;

        int32 dst_pitch, src_pitch, width,  size;
        int pos = 0;
        void *src, *dst;
        MatrixDim dim;
        cudaMemcpyKind kind;
        switch(copykind)
        {
            case 0:
                kind = cudaMemcpyHostToHost;
                break;
            case 1:
                kind = cudaMemcpyHostToDevice;
                break;
            case 2:
                kind = cudaMemcpyDeviceToHost;
                break;
            case 3:
                kind = cudaMemcpyDeviceToDevice;
                break;
            default:
                KALDI_ERR << "Default based unified virtual address space";
                break;
        }

		dim = encoder_linearity_.Dim();
		src_pitch = dim.stride*sizeof(BaseFloat);
		dst_pitch = src_pitch;
		width = dim.cols*sizeof(BaseFloat);
        dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)encoder_linearity_.Data());
		src = (void*) (direction==0 ? (char *)encoder_linearity_.Data() : ((char *)host+pos));
		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
		pos += encoder_linearity_.SizeInBytes();

		dim = predict_linearity_.Dim();
		src_pitch = dim.stride*sizeof(BaseFloat);
		dst_pitch = src_pitch;
		width = dim.cols*sizeof(BaseFloat);
        dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)predict_linearity_.Data());
		src = (void*) (direction==0 ? (char *)predict_linearity_.Data() : ((char *)host+pos));
		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
		pos += predict_linearity_.SizeInBytes();

		size = bias_.Dim()*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)bias_.Data());
		src = (void*) (direction==0 ? (char *)bias_.Data() : ((char *)host+pos));
		cudaMemcpy(dst, src, size, kind);
		pos += size;

		dim = join_linearity_.Dim();
		src_pitch = dim.stride*sizeof(BaseFloat);
		dst_pitch = src_pitch;
		width = dim.cols*sizeof(BaseFloat);
        dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)join_linearity_.Data());
		src = (void*) (direction==0 ? (char *)join_linearity_.Data() : ((char *)host+pos));
		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
		pos += join_linearity_.SizeInBytes();

		size = join_bias_.Dim()*sizeof(BaseFloat);
		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)join_bias_.Data());
		src = (void*) (direction==0 ? (char *)join_bias_.Data() : ((char *)host+pos));
		cudaMemcpy(dst, src, size, kind);
		pos += size;

  	  CU_SAFE_CALL(cudaGetLastError());

  	  CuDevice::Instantiate().AccuProfile(__func__, tim);

  	  return pos;
  }else
#endif
  	{
  		// not implemented for CPU yet
  		return 0;
  	}
  }

protected:
  int32 encoder_dim_;
  int32 predict_dim_;
  int32 join_dim_;

  int32 nstream_;
  int32 maxT_;
  int32 maxU_;

  // weights
  CuMatrix<BaseFloat> encoder_linearity_;
  CuMatrix<BaseFloat> predict_linearity_;
  CuVector<BaseFloat> bias_;
  CuMatrix<BaseFloat> encoder_linearity_corr_;
  CuMatrix<BaseFloat> predict_linearity_corr_;
  CuVector<BaseFloat> bias_corr_;

  CuMatrix<BaseFloat> join_linearity_;
  CuVector<BaseFloat> join_bias_;
  CuMatrix<BaseFloat> join_linearity_corr_;
  CuVector<BaseFloat> join_bias_corr_;
  CuMatrix<BaseFloat> join_diff_, in_diff_;

  CuMatrix<BaseFloat> Ah_t_, Bp_u_, h_t_u_;
  CuMatrix<BaseFloat> d_Ah_t_, d_Bp_u_, d_h_t_u_;

  std::vector<int32> encoder_utt_frames_;
  std::vector<int32> predict_utt_words_;

  BaseFloat learn_rate_coef_;
  BaseFloat bias_learn_rate_coef_;
  BaseFloat max_norm_;

  BaseFloat local_lrate;
  BaseFloat local_lrate_bias;
};

} // namespace nnet0
} // namespace kaldi

#endif
