// nnet0/nnet-fsmn.h

// Copyright 2018 Alibaba.Inc (Author: ShiLiang Zhang) 
// Copyright 2018 Alibaba.Inc (Author: Wei Deng) 

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


#ifndef KALDI_NNET_NNET_FSMN_H_
#define KALDI_NNET_NNET_FSMN_H_

#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-kernels.h"


namespace kaldi {
namespace nnet0 {
 class Fsmn : public UpdatableComponent {
  public:
   Fsmn(int32 dim_in, int32 dim_out)
     : UpdatableComponent(dim_in, dim_out),
     learn_rate_coef_(1.0),
	 clip_gradient_(0.0)
   {
   }
   ~Fsmn()
   { }

   Component* Copy() const { return new Fsmn(*this); }
   ComponentType GetType() const { return kFsmn; }

   void SetFlags(const Vector<BaseFloat> &flags) {
     flags_.Resize(flags.Dim(), kSetZero);
     flags_.CopyFromVec(flags);
   }

   void InitData(std::istream &is) {
     // define options
     float learn_rate_coef = 1.0;
     int l_order = 1, r_order = 1;
     int l_stride = 1, r_stride = 1;
     float range = 0.0;
     // parse config
     std::string token;
     while (is >> std::ws, !is.eof()){
       ReadToken(is, false, &token);
       /**/ if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
       else if (token == "<LOrder>") ReadBasicType(is, false, &l_order);
       else if (token == "<ROrder>") ReadBasicType(is, false, &r_order);
       else if (token == "<LStride>") ReadBasicType(is, false, &l_stride);
       else if (token == "<RStride>") ReadBasicType(is, false, &r_stride);
       else if (token == "<ClipGradient>") ReadBasicType(is, false, &clip_gradient_);
       else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
         << " (LearnRateCoef|LOrder|ROrder|LStride|LStride|ClipGradient)";
     }

     //init
     learn_rate_coef_ = learn_rate_coef;
     l_order_ = l_order;
     r_order_ = r_order;
     l_stride_ = l_stride;
     r_stride_ = r_stride;
     // initialize filter
     range = sqrt(6)/sqrt(l_order + input_dim_);
     l_filter_.Resize(l_order, input_dim_, kSetZero);
     RandUniform(0.0, range, &l_filter_);

     range = sqrt(6)/sqrt(r_order + input_dim_);
     r_filter_.Resize(r_order, input_dim_, kSetZero);
     RandUniform(0.0, range, &r_filter_);

     //gradient related
     l_filter_corr_.Resize(l_order_, input_dim_, kSetZero);
     r_filter_corr_.Resize(r_order_, input_dim_, kSetZero);

     KALDI_ASSERT(clip_gradient_ >= 0.0);
   }

   void ReadData(std::istream &is, bool binary) {
     // optional learning-rate coefs
	 std::string token;
	 while ('<' == Peek(is, binary)) {
        ReadToken(is, false, &token);
        if (token == "<LearnRateCoef>")
          ReadBasicType(is, binary, &learn_rate_coef_);
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

     KALDI_ASSERT(l_filter_.NumRows() == l_order_);
     KALDI_ASSERT(l_filter_.NumCols() == input_dim_);

     KALDI_ASSERT(r_filter_.NumRows() == r_order_);
     KALDI_ASSERT(r_filter_.NumCols() == input_dim_);

     //gradient related
     l_filter_corr_.Resize(l_order_, input_dim_, kSetZero);
     r_filter_corr_.Resize(r_order_, input_dim_, kSetZero);
   }

   void WriteData(std::ostream &os, bool binary) const {
     WriteToken(os, binary, "<LearnRateCoef>");
     WriteBasicType(os, binary, learn_rate_coef_);
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
   }

   void ResetMomentum(void) {
   }

   void ResetGradient() {
	   l_filter_corr_.SetZero();
	   r_filter_corr_.SetZero();
   }

   int32 NumParams() const { 
     return l_filter_.NumRows()*l_filter_.NumCols() + r_filter_.NumRows()*r_filter_.NumCols(); 
   }

   int32 GetDim() const {
     return ( l_filter_.SizeInBytes()/sizeof(BaseFloat) + r_filter_.SizeInBytes()/sizeof(BaseFloat) );
   }

   void GetParams(Vector<BaseFloat>* wei_copy) const {
     //KALDI_ASSERT(wei_copy->Dim() == NumParams());
     wei_copy->Resize(NumParams());
     int32 l_filter_num_elem = l_filter_.NumRows() * l_filter_.NumCols();
     int32 r_filter_num_elem = r_filter_.NumRows() * r_filter_.NumCols();
     wei_copy->Range(0, l_filter_num_elem).CopyRowsFromMat(l_filter_);
     wei_copy->Range(l_filter_num_elem, r_filter_num_elem).CopyRowsFromMat(r_filter_);
   }

   void SetParams(const VectorBase<BaseFloat> &wei_copy) {
     KALDI_ASSERT(wei_copy.Dim() == NumParams());
     int32 l_filter_num_elem = l_filter_.NumRows() * l_filter_.NumCols();
     int32 r_filter_num_elem = r_filter_.NumRows() * r_filter_.NumCols();
     l_filter_.CopyRowsFromVec(wei_copy.Range(0, l_filter_num_elem));
     r_filter_.CopyRowsFromVec(wei_copy.Range(l_filter_num_elem, r_filter_num_elem));
   }

   void GetGradient(VectorBase<BaseFloat>* wei_copy) const {
     KALDI_ASSERT(wei_copy->Dim() == NumParams());
   }

   std::string Info() const {
     return std::string("\n  l_filter") + MomentStatistics(l_filter_) +
       "\n  r_filter" + MomentStatistics(r_filter_);
   }
   std::string InfoGradient() const {
     return std::string("\n  lr-coef ") + ToString(learn_rate_coef_) +
       ", l_order " + ToString(l_order_) +
       ", r_order " + ToString(r_order_) +
       ", l_stride " + ToString(l_stride_) +
       ", r_stride " + ToString(r_stride_) +

	   "\n  l_filter_grad" + MomentStatistics(l_filter_corr_) +
	   "\n  r_filter_grad" + MomentStatistics(r_filter_corr_);

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

     out->GenMemory(in, l_filter_, r_filter_, flags_, l_order_, r_order_, l_stride_, r_stride_);
   }

   void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
     const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
   
     //const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;

     in_diff->MemoryErrBack(out_diff, l_filter_, r_filter_, flags_, l_order_, r_order_, l_stride_, r_stride_);

     //l_filter_.GetLfilterErr(out_diff, in, flags_, l_order_, l_stride_, lr);
     //r_filter_.GetRfilterErr(out_diff, in, flags_, r_order_, r_stride_, lr);
   }

   void Gradient(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out_diff) {

     //l_filter_corr_.Set(0.0);
     //r_filter_corr_.Set(0.0);
     l_filter_corr_.GetLfilterErr(out_diff, in, flags_, l_order_, l_stride_, 1.0);
     r_filter_corr_.GetRfilterErr(out_diff, in, flags_, r_order_, r_stride_, 1.0);

     if (clip_gradient_ > 0.0) {
		l_filter_corr_.ApplyFloor(-clip_gradient_);
		l_filter_corr_.ApplyCeiling(clip_gradient_);
		r_filter_corr_.ApplyFloor(-clip_gradient_);
		r_filter_corr_.ApplyCeiling(clip_gradient_);
     }
   }

   void UpdateGradient() {
     const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
     l_filter_.AddMat(-lr, l_filter_corr_);
     r_filter_.AddMat(-lr, r_filter_corr_);
   }

   void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
     const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
     l_filter_.AddMat(-lr, l_filter_corr_);
     r_filter_.AddMat(-lr, r_filter_corr_);
   }

   int WeightCopy(void *host, int direction, int copykind)
   {   
 #if HAVE_CUDA == 1
   if (CuDevice::Instantiate().Enabled()) {
         CuTimer tim;

         int32 dst_pitch, src_pitch, width;
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
   CuMatrix<BaseFloat> l_filter_;
   CuMatrix<BaseFloat> l_filter_corr_;
   CuMatrix<BaseFloat> r_filter_;
   CuMatrix<BaseFloat> r_filter_corr_;

   CuVector<BaseFloat> flags_;
   BaseFloat learn_rate_coef_;
   BaseFloat clip_gradient_;
   int l_order_;
   int r_order_;
   int l_stride_;
   int r_stride_;  
 };

} // namespace nnet0
} // namespace kaldi

#endif
