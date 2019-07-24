// nnet0/nnet-deep-fsmn.h

// Copyright 2018 Alibaba.Inc (Author: Shiliang Zhang) 
// Copyright 2018 Alibaba.Inc (Author: Wei Deng) 
//
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


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-kernels.h"

namespace kaldi {
namespace nnet0 {
 class DeepFsmn : public UpdatableComponent {
  public:
   DeepFsmn(int32 dim_in, int32 dim_out)
     : UpdatableComponent(dim_in, dim_out),
     learn_rate_coef_(1.0),
	 clip_gradient_(0.0)
   {
   }
   ~DeepFsmn()
   { }

   Component* Copy() const { return new DeepFsmn(*this); }
   ComponentType GetType() const { return kDeepFsmn; }

   void SetFlags(const Vector<BaseFloat> &flags) {
     flags_.Resize(flags.Dim(), kSetZero);
     flags_.CopyFromVec(flags);
   }

   void InitData(std::istream  &is) {
     // define options
     float learn_rate_coef = 1.0;
     int hid_size;
     int l_order = 1, r_order = 1;
     int l_stride = 1, r_stride = 1;
     float range = 0.0;
     // parse config
     std::string token;
     while (is >> std::ws, !is.eof()) {
       ReadToken(is, false, &token);
       /**/ if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
       else if (token == "<HidSize>") ReadBasicType(is, false, &hid_size);
       else if (token == "<LOrder>") ReadBasicType(is, false, &l_order);
       else if (token == "<ROrder>") ReadBasicType(is, false, &r_order);
       else if (token == "<LStride>") ReadBasicType(is, false, &l_stride);
       else if (token == "<RStride>") ReadBasicType(is, false, &r_stride);
       else if (token == "<ClipGradient>") ReadBasicType(is, false, &clip_gradient_);
       else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
         << " (LearnRateCoef|HidSize|LOrder|ROrder|LStride|LStride|ClipGradient)";
     }
     //parameters
     learn_rate_coef_ = learn_rate_coef;
     l_order_ = l_order;
     r_order_ = r_order;
     l_stride_ = l_stride;
     r_stride_ = r_stride;
     hid_size_ = hid_size;
     // initialize 
     range = sqrt(6)/sqrt(l_order_ + output_dim_);
     l_filter_.Resize(l_order_, output_dim_, kSetZero);
     RandUniform(0.0, range, &l_filter_);

     range = sqrt(6)/sqrt(r_order_ + input_dim_);
     r_filter_.Resize(r_order_, output_dim_, kSetZero);
     RandUniform(0.0, range, &r_filter_);

     //linear transform
     range = sqrt(6)/sqrt(hid_size_ + output_dim_);
     p_weight_.Resize(output_dim_, hid_size_, kSetZero);
     RandUniform(0.0, range, &p_weight_);

     ///affine transform + nonlinear activation
     range = sqrt(6)/sqrt(hid_size_ + input_dim_);
     linearity_.Resize(hid_size_, input_dim_, kSetZero);
     RandUniform(0.0, range, &linearity_);

     bias_.Resize(hid_size_, kSetZero);

     //gradient related
     p_weight_corr_.Resize(output_dim_, hid_size_, kSetZero);
     linearity_corr_.Resize(hid_size_, input_dim_, kSetZero);
     bias_corr_.Resize(hid_size_, kSetZero);

     KALDI_ASSERT(clip_gradient_ >= 0.0);
   }

   void ReadData(std::istream &is, bool binary) {
     // optional learning-rate coefs
	 std::string token;
	 while ('<' == Peek(is, binary)) {
         ReadToken(is, false, &token);
         if (token == "<LearnRateCoef>")
           ReadBasicType(is, binary, &learn_rate_coef_);
         else if (token == "<HidSize>")
           ReadBasicType(is, binary, &hid_size_);
         else if (token == "<LOrder>")
           ReadBasicType(is, binary, &l_order_);
         else if (token == "<ROrder>")
           ReadBasicType(is, binary, &r_order_);
         else if (token == "<LStride>")
           ReadBasicType(is, binary, &l_stride_);
         else if (token == "<RStride>")
           ReadBasicType(is, binary, &r_stride_);
         else if (token == "<ClipGradient>")
           ReadBasicType(is, binary, &clip_gradient_);
     }

     // weights
     l_filter_.Read(is, binary);
     r_filter_.Read(is, binary);
     p_weight_.Read(is, binary);
     linearity_.Read(is, binary);
     bias_.Read(is, binary);
     KALDI_ASSERT(l_filter_.NumRows() == l_order_);
     KALDI_ASSERT(l_filter_.NumCols() == input_dim_);
     KALDI_ASSERT(r_filter_.NumRows() == r_order_);
     KALDI_ASSERT(r_filter_.NumCols() == input_dim_);
     KALDI_ASSERT(p_weight_.NumRows() == output_dim_);
     KALDI_ASSERT(p_weight_.NumCols() == hid_size_);
     KALDI_ASSERT(linearity_.NumRows() == hid_size_);
     KALDI_ASSERT(linearity_.NumCols() == input_dim_);
     KALDI_ASSERT(bias_.Dim() == hid_size_);

     //gradient related
     l_filter_corr_.Resize(l_order_, output_dim_, kSetZero);
     r_filter_corr_.Resize(r_order_, output_dim_, kSetZero);
     p_weight_corr_.Resize(output_dim_, hid_size_, kSetZero);
     linearity_corr_.Resize(hid_size_, input_dim_, kSetZero);
     bias_corr_.Resize(hid_size_, kSetZero);
   }

   void WriteData(std::ostream &os, bool binary) const {
     WriteToken(os, binary, "<LearnRateCoef>");
     WriteBasicType(os, binary, learn_rate_coef_);
     WriteToken(os, binary, "<HidSize>");
     WriteBasicType(os, binary, hid_size_);
     WriteToken(os, binary, "<LOrder>");
     WriteBasicType(os, binary, l_order_);
     WriteToken(os, binary, "<ROrder>");
     WriteBasicType(os, binary, r_order_);
     WriteToken(os, binary, "<LStride>");
     WriteBasicType(os, binary, l_stride_);
     WriteToken(os, binary, "<RStride>");
     WriteBasicType(os, binary, r_stride_);
     if (clip_gradient_ != 0.0) {
        WriteToken(os, binary, "<ClipGradient>");
        WriteBasicType(os, binary, clip_gradient_);
     }

     // weights
     l_filter_.Write(os, binary);
     r_filter_.Write(os, binary);
     p_weight_.Write(os, binary);
     linearity_.Write(os, binary);
     bias_.Write(os, binary);

   }

   void ResetMomentum(void) {
     l_filter_corr_.Set(0.0);
     r_filter_corr_.Set(0.0);
     p_weight_corr_.Set(0.0);
     linearity_corr_.Set(0.0);
     bias_corr_.Set(0.0);
   }

   void ResetGradient() {
		l_filter_corr_.SetZero();
		r_filter_corr_.SetZero();
		p_weight_corr_.SetZero();
		linearity_corr_.SetZero();
		bias_corr_.SetZero();
   }

   int32 GetDim() const {
    return ( l_filter_.SizeInBytes()/sizeof(BaseFloat) +
         r_filter_.SizeInBytes()/sizeof(BaseFloat) +
         p_weight_.SizeInBytes()/sizeof(BaseFloat) +
         linearity_.SizeInBytes()/sizeof(BaseFloat) +
         bias_.Dim() );
   }

   int32 NumParams() const { 
     return l_filter_.NumRows()*l_filter_.NumCols() + r_filter_.NumRows()*r_filter_.NumCols() + p_weight_.NumRows()*p_weight_.NumCols() 
       + linearity_.NumRows()*linearity_.NumCols() + bias_.Dim();
   }

   void GetParams(Vector<BaseFloat>* wei_copy) const {
     //KALDI_ASSERT(wei_copy->Dim() == NumParams());
     wei_copy->Resize(NumParams());
     int32 l_filter_num_elem = l_filter_.NumRows() * l_filter_.NumCols();
     int32 r_filter_num_elem = r_filter_.NumRows() * r_filter_.NumCols();
     int32 p_weight_num_elem = p_weight_.NumRows()*p_weight_.NumCols();
     int32 linearity_num_elem = linearity_.NumRows()*linearity_.NumCols();
     int32 offset=0;
     wei_copy->Range(offset, l_filter_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(l_filter_));
     offset += l_filter_num_elem;
     wei_copy->Range(offset, r_filter_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(r_filter_));
     offset += r_filter_num_elem;
     wei_copy->Range(offset, p_weight_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(p_weight_));
     offset += p_weight_num_elem;
     wei_copy->Range(offset, linearity_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(linearity_));
     offset += linearity_num_elem;
     wei_copy->Range(offset, bias_.Dim()).CopyFromVec(Vector<BaseFloat>(bias_));
   }

   void SetParams(const VectorBase<BaseFloat> &wei_copy) {
     KALDI_ASSERT(wei_copy.Dim() == NumParams());
     int32 l_filter_num_elem = l_filter_.NumRows() * l_filter_.NumCols();
     int32 r_filter_num_elem = r_filter_.NumRows() * r_filter_.NumCols();
     int32 p_weight_num_elem = p_weight_.NumRows()*p_weight_.NumCols();
     int32 linearity_num_elem = linearity_.NumRows()*linearity_.NumCols();
     int32 offset = 0;
     l_filter_.CopyRowsFromVec(wei_copy.Range(offset, l_filter_num_elem));
     offset += l_filter_num_elem;
     r_filter_.CopyRowsFromVec(wei_copy.Range(offset, r_filter_num_elem));
     offset += r_filter_num_elem;
     p_weight_.CopyRowsFromVec(wei_copy.Range(offset, p_weight_num_elem));
     offset += p_weight_num_elem;
     linearity_.CopyRowsFromVec(wei_copy.Range(offset, linearity_num_elem));
     offset += linearity_num_elem;
     bias_.CopyFromVec(wei_copy.Range(offset, bias_.Dim()));
   }

   void GetGradient(VectorBase<BaseFloat>* wei_copy) const {
     KALDI_ASSERT(wei_copy->Dim() == NumParams());
     int32 p_weight_num_elem = p_weight_corr_.NumRows()*p_weight_corr_.NumCols();
     int32 linearity_num_elem = linearity_corr_.NumRows()*linearity_corr_.NumCols();
     int32 offset = 0;
     wei_copy->Range(offset, p_weight_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(p_weight_corr_));
     offset += p_weight_num_elem;
     wei_copy->Range(offset, linearity_num_elem).CopyRowsFromMat(Matrix<BaseFloat>(linearity_corr_));
     offset += linearity_num_elem;
     wei_copy->Range(offset, bias_.Dim()).CopyFromVec(Vector<BaseFloat>(bias_corr_));
   }

   std::string Info() const {
     return std::string("\n  l_filter") + MomentStatistics(l_filter_) +
       "\n  r_filter" + MomentStatistics(r_filter_) +
       "\n  p_weight" + MomentStatistics(p_weight_) +
       "\n  linearity" + MomentStatistics(linearity_) +
       "\n  bias" + MomentStatistics(bias_);
   }
   std::string InfoGradient() const {
     return std::string("\n, lr-coef ") + ToString(learn_rate_coef_) +
       ", hid_size" + ToString(hid_size_) +
       ", l_order " + ToString(l_order_) +
       ", r_order " + ToString(r_order_) +
       ", l_stride " + ToString(l_stride_) +
       ", r_stride " + ToString(r_stride_) +

	   "\n  l_filter_grad" + MomentStatistics(l_filter_corr_) +
	   "\n  r_filter_grad" + MomentStatistics(r_filter_corr_) +
	   "\n  p_weight_grad" + MomentStatistics(p_weight_corr_) +
	   "\n  linearity_grad" + MomentStatistics(linearity_corr_) +
	   "\n  bias_grad" + MomentStatistics(bias_corr_);
   }

   void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
     // skip frames
     int inframes = in.NumRows();
     int nframes = flags_.Dim();
     KALDI_ASSERT(nframes%inframes == 0);

     int skip_frames = nframes/inframes; 
     if (skip_frames > 1) {
        Vector<BaseFloat> tmp(flags_);
        Vector<BaseFloat> flags(inframes, kUndefined);
        for (int i = 0; i < inframes; i++)
            flags(i) = tmp(skip_frames*i);
        flags_ = flags;
     }

     nframes = in.NumRows();
     //////////////////////////////////////
     //step1. nonlinear affine transform
     hid_out_.Resize(nframes, hid_size_, kUndefined);
     // pre copy bias
     hid_out_.AddVecToRows(1.0, bias_, 0.0);
     // multiply by weights^t
     hid_out_.AddMatMat(1.0, in, kNoTrans, linearity_, kTrans, 1.0);
     // Relu nonlinear activation function
     hid_out_.ApplyFloor(0.0);

     ////Step2. linear affine transform
     p_out_.Resize(nframes, output_dim_, kUndefined);
     p_out_.AddMatMat(1.0, hid_out_, kNoTrans, p_weight_, kTrans, 0.0);

     ////Step3. fsmn layer
     out->GenMemory(p_out_, l_filter_, r_filter_, flags_, l_order_, r_order_, l_stride_, r_stride_);

     ///step4. skip connection
     out->AddMat(1.0, in, kNoTrans);
   }

   void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
     const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
     
     int nframes = in.NumRows();
     //const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
     const BaseFloat mmt = opts_.momentum;
     //Step 1. fsmn layer
     p_out_err_.Resize(nframes, output_dim_, kSetZero);
     p_out_err_.MemoryErrBack(out_diff, l_filter_, r_filter_, flags_, l_order_, r_order_, l_stride_, r_stride_);
     //l_filter_.GetLfilterErr(out_diff, p_out_, flags_, l_order_, l_stride_, lr);
     //r_filter_.GetRfilterErr(out_diff, p_out_, flags_, r_order_, r_stride_, lr);
     
     //Step 2. linear affine transform
     // multiply error derivative by weights
     hid_out_err_.Resize(nframes, hid_size_, kSetZero);
     hid_out_err_.AddMatMat(1.0, p_out_err_, kNoTrans, p_weight_, kNoTrans, 0.0);
     p_weight_corr_.AddMatMat(1.0, p_out_err_, kTrans, hid_out_, kNoTrans, mmt);

     //Step3. nonlinear affine transform
     hid_out_.ApplyHeaviside();
     hid_out_err_.MulElements(hid_out_);

     in_diff->AddMatMat(1.0, hid_out_err_, kNoTrans, linearity_, kNoTrans, 0.0);
     //linearity_corr_.AddMatMat(1.0, hid_out_err_, kTrans, in, kNoTrans, mmt);
     //bias_corr_.AddRowSumMat(1.0, hid_out_err_, mmt);

     //Step4. skip connection
     in_diff->AddMat(1.0, out_diff, kNoTrans);
   }

   void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &out_diff) {

     //const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
     //const BaseFloat l2 = opts_.l2_penalty;
	 const BaseFloat mmt = opts_.momentum;
     // we will also need the number of frames in the mini-batch
     num_frames_ = input.NumRows();
	 //Step 1. fsmn layer
	 //l_filter_corr_.Set(0.0);
	 //r_filter_corr_.Set(0.0);
     l_filter_corr_.GetLfilterErr(out_diff, p_out_, flags_, l_order_, l_stride_, 1.0);
     r_filter_corr_.GetRfilterErr(out_diff, p_out_, flags_, r_order_, r_stride_, 1.0);

	 //Step 2. linear affine transform
	 // multiply error derivative by weights
	 //p_weight_corr_.AddMatMat(1.0, p_out_err_, kTrans, hid_out_, kNoTrans, mmt);

	 //Step3. nonlinear affine transform
	 linearity_corr_.AddMatMat(1.0, hid_out_err_, kTrans, input, kNoTrans, mmt);
	 bias_corr_.AddRowSumMat(1.0, hid_out_err_, mmt);

	 if (clip_gradient_ > 0.0) {
		l_filter_corr_.ApplyFloor(-clip_gradient_);
		l_filter_corr_.ApplyCeiling(clip_gradient_);
		r_filter_corr_.ApplyFloor(-clip_gradient_);
		r_filter_corr_.ApplyCeiling(clip_gradient_);
		p_weight_corr_.ApplyFloor(-clip_gradient_);
		p_weight_corr_.ApplyCeiling(clip_gradient_);
		linearity_corr_.ApplyFloor(-clip_gradient_);
		linearity_corr_.ApplyCeiling(clip_gradient_);
		bias_corr_.ApplyFloor(-clip_gradient_);
		bias_corr_.ApplyCeiling(clip_gradient_);
	 }
   }

   void UpdateGradient() {

     const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
     const BaseFloat l2 = opts_.l2_penalty;

     if (l2 != 0.0) {
        linearity_.AddMat(-lr*l2*num_frames_, linearity_);
        p_weight_.AddMat(-lr*l2*num_frames_,  p_weight_);
     }
     l_filter_.AddMat(-lr, l_filter_corr_);
     r_filter_.AddMat(-lr, r_filter_corr_);
     p_weight_.AddMat(-lr, p_weight_corr_);
     linearity_.AddMat(-lr, linearity_corr_);
     bias_.AddVec(-lr, bias_corr_);
   }

   void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
     
     const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
     const BaseFloat l2 = opts_.l2_penalty;

     if (l2 != 0.0) {
       linearity_.AddMat(-lr*l2, linearity_);
       p_weight_.AddMat(-lr*l2,  p_weight_);
     }
     l_filter_.AddMat(-lr, l_filter_corr_);
     r_filter_.AddMat(-lr, r_filter_corr_);
     p_weight_.AddMat(-lr, p_weight_corr_);
     linearity_.AddMat(-lr, linearity_corr_);
     bias_.AddVec(-lr, bias_corr_);
   }

   int WeightCopy(void *host, int direction, int copykind)
   {
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

  		dim = l_filter_.Dim();
  		src_pitch = dim.stride*sizeof(BaseFloat);
  		dst_pitch = src_pitch;
  		width = dim.cols*sizeof(BaseFloat);
        dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)l_filter_.Data());
  		src = (void*) (direction==0 ? (char *)l_filter_.Data() : ((char *)host+pos));
  		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
  		pos += l_filter_.SizeInBytes();

  		dim = r_filter_.Dim();
  		src_pitch = dim.stride*sizeof(BaseFloat);
  		dst_pitch = src_pitch;
  		width = dim.cols*sizeof(BaseFloat);
        dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)r_filter_.Data());
  		src = (void*) (direction==0 ? (char *)r_filter_.Data() : ((char *)host+pos));
  		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
  		pos += r_filter_.SizeInBytes();

  		dim = p_weight_.Dim();
  		src_pitch = dim.stride*sizeof(BaseFloat);
  		dst_pitch = src_pitch;
  		width = dim.cols*sizeof(BaseFloat);
          dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)p_weight_.Data());
  		src = (void*) (direction==0 ? (char *)p_weight_.Data() : ((char *)host+pos));
  		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
  		pos += p_weight_.SizeInBytes();

 		dim = linearity_.Dim();
 		src_pitch = dim.stride*sizeof(BaseFloat);
 		dst_pitch = src_pitch;
 		width = dim.cols*sizeof(BaseFloat);
         dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)linearity_.Data());
 		src = (void*) (direction==0 ? (char *)linearity_.Data() : ((char *)host+pos));
 		cudaMemcpy2D(dst, dst_pitch, src, src_pitch, width, dim.rows, kind);
 		pos += linearity_.SizeInBytes();

 		size = bias_.Dim()*sizeof(BaseFloat);
 		dst = (void*) (direction==0 ? ((char *)host+pos) : (char *)bias_.Data());
 		src = (void*) (direction==0 ? (char *)bias_.Data() : ((char *)host+pos));
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

 private:
   ///fsmn layer
   CuMatrix<BaseFloat> l_filter_;
   CuMatrix<BaseFloat> l_filter_corr_;
   CuMatrix<BaseFloat> r_filter_;
   CuMatrix<BaseFloat> r_filter_corr_;
   CuVector<BaseFloat> flags_;

   //linear affine transform
   CuMatrix<BaseFloat> p_out_;
   CuMatrix<BaseFloat> p_out_err_;
   CuMatrix<BaseFloat> p_weight_;
   CuMatrix<BaseFloat> p_weight_corr_;

   ///affine transform + nonlinear activation
   CuMatrix<BaseFloat> hid_out_;
   CuMatrix<BaseFloat> hid_out_err_;
   CuMatrix<BaseFloat> linearity_;
   CuVector<BaseFloat> bias_;
   CuMatrix<BaseFloat> linearity_corr_;
   CuVector<BaseFloat> bias_corr_;

   BaseFloat learn_rate_coef_;
   BaseFloat clip_gradient_;
   int l_order_;
   int r_order_;
   int l_stride_;
   int r_stride_;  
   int hid_size_;
   int32 num_frames_;
 };

} // namespace nnet0
} // namespace kaldi

#endif
