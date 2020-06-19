// nnet3/attention.cc

// Copyright      2017  Johns Hopkins University (author: Daniel Povey)
//                      Hossein Hadian

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

#include <iterator>
#include <sstream>
#include <iomanip>
#include "nnet3/attention.h"
#include "nnet3/nnet-parse.h"

namespace kaldi {
namespace nnet3 {
namespace attention {


void GetAttentionDotProducts(BaseFloat alpha,
                             const CuMatrixBase<BaseFloat> &A,
                             const CuMatrixBase<BaseFloat> &B,
                             CuMatrixBase<BaseFloat> *C) {
  KALDI_ASSERT(A.NumCols() == B.NumCols() &&
               A.NumRows() == C->NumRows());
  int32 num_output_rows = A.NumRows(),
      input_num_cols = A.NumCols(),
      num_extra_rows = B.NumRows() - A.NumRows(),
      context_dim = C->NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C->NumCols(),
                             C->NumRows());
  for (int32 o = 0; o < context_dim; o++) {
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    c_col.AddDiagMatMat(alpha, A, kNoTrans, B_part, kTrans, 0.0);
  }
  C->CopyFromMat(Ctrans, kTrans);
}

void ApplyScalesToOutput(BaseFloat alpha,
                         const CuMatrixBase<BaseFloat> &B,
                         const CuMatrixBase<BaseFloat> &C,
                         CuMatrixBase<BaseFloat> *A) {
  KALDI_ASSERT(A->NumCols() == B.NumCols() &&
               A->NumRows() == C.NumRows());
  int32 num_output_rows = A->NumRows(),
      input_num_cols = A->NumCols(),
      num_extra_rows = B.NumRows() - A->NumRows(),
      context_dim = C.NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C, kTrans);
  for (int32 o = 0; o < context_dim; o++) {
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    A->AddDiagVecMat(alpha, c_col, B_part, kNoTrans, 1.0);
  }
}

void ApplyScalesToInput(BaseFloat alpha,
                        const CuMatrixBase<BaseFloat> &A,
                        const CuMatrixBase<BaseFloat> &C,
                        CuMatrixBase<BaseFloat> *B) {
  KALDI_ASSERT(A.NumCols() == B->NumCols() &&
               A.NumRows() == C.NumRows());
  int32 num_output_rows = A.NumRows(),
      input_num_cols = A.NumCols(),
      num_extra_rows = B->NumRows() - A.NumRows(),
      context_dim = C.NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C, kTrans);
  for (int32 o = 0; o < context_dim; o++) {
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(*B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    B_part.AddDiagVecMat(alpha, c_col, A, kNoTrans, 1.0);
  }
}

void AttentionForward(BaseFloat key_scale,
                      const CuMatrixBase<BaseFloat> &keys,
                      const CuMatrixBase<BaseFloat> &queries,
                      const CuMatrixBase<BaseFloat> &values,
                      CuMatrixBase<BaseFloat> *c,
                      CuMatrixBase<BaseFloat> *output) {
  // First check the dimensions and values.
  KALDI_ASSERT(key_scale > 0.0);
  int32 num_input_rows = keys.NumRows(),
      key_dim = keys.NumCols(),
      num_output_rows = queries.NumRows(),
      context_dim = queries.NumCols() - key_dim,
      value_dim = values.NumCols();
  KALDI_ASSERT(num_input_rows > 0 && key_dim > 0 &&
               num_input_rows > num_output_rows &&
               context_dim > 0 &&
               (num_input_rows - num_output_rows) % (context_dim - 1) == 0 &&
               values.NumRows() == num_input_rows);
  KALDI_ASSERT(c->NumRows() == num_output_rows &&
               c->NumCols() == context_dim);
  KALDI_ASSERT(output->NumRows() == num_output_rows &&
               (output->NumCols() == value_dim ||
                output->NumCols() == value_dim + context_dim));

  CuSubMatrix<BaseFloat> queries_key_part(
      queries, 0, num_output_rows,
      0, key_dim),
      queries_context_part(
          queries, 0, num_output_rows,
          key_dim, context_dim);

  GetAttentionDotProducts(key_scale,
                          queries_key_part,
                          keys, c);
  // think of 'queries_context_part' as a position-dependent bias term.
  c->AddMat(1.0, queries_context_part);
  // compute the soft-max function.  Up till this point, 'c'
  // actually contained what in attention.h we called 'b', which is
  // the input to the softmax.
  c->SoftMaxPerRow(*c);


  // the part of the output that is weighted
  // combinations of the input values.
  CuSubMatrix<BaseFloat> output_values_part(
      *output, 0, num_output_rows, 0, value_dim);

  ApplyScalesToOutput(1.0, values, *c, &output_values_part);


  if (output->NumCols() == value_dim + context_dim) {
    CuSubMatrix<BaseFloat> output_context_part(
        *output, 0, num_output_rows, value_dim, context_dim);
    output_context_part.CopyFromMat(*c);
  }
}

void AttentionBackward(BaseFloat key_scale,
                       const CuMatrixBase<BaseFloat> &keys,
                       const CuMatrixBase<BaseFloat> &queries,
                       const CuMatrixBase<BaseFloat> &values,
                       const CuMatrixBase<BaseFloat> &c,
                       const CuMatrixBase<BaseFloat> &output_deriv,
                       CuMatrixBase<BaseFloat> *keys_deriv,
                       CuMatrixBase<BaseFloat> *queries_deriv,
                       CuMatrixBase<BaseFloat> *values_deriv) {

  // First check the dimensions and values.
  KALDI_ASSERT(key_scale > 0.0);
  int32 num_input_rows = keys.NumRows(),
      key_dim = keys.NumCols(),
      num_output_rows = queries.NumRows(),
      context_dim = queries.NumCols() - key_dim,
      value_dim = values.NumCols();
  KALDI_ASSERT(num_input_rows > 0 && key_dim > 0 &&
               num_input_rows > num_output_rows &&
               context_dim > 0 &&
               (num_input_rows - num_output_rows) % (context_dim - 1) == 0 &&
               values.NumRows() == num_input_rows);
  KALDI_ASSERT(SameDim(keys, *keys_deriv) &&
               SameDim(queries, *queries_deriv) &&
               SameDim(values, *values_deriv));

  KALDI_ASSERT(c.NumRows() == num_output_rows &&
               c.NumCols() == context_dim);
  KALDI_ASSERT(output_deriv.NumRows() == num_output_rows &&
               (output_deriv.NumCols() == value_dim ||
                output_deriv.NumCols() == value_dim + context_dim));

  CuMatrix<BaseFloat> c_deriv(num_output_rows, context_dim,
                              kUndefined);

  CuSubMatrix<BaseFloat> output_values_part_deriv(
      output_deriv, 0, num_output_rows, 0, value_dim);
  // This is the backprop w.r.t. the forward-pass statement:
  // ApplyScalesToOutput(1.0, values, *c, &output_values_part);
  GetAttentionDotProducts(1.0, output_values_part_deriv,
                          values, &c_deriv);

  if (output_deriv.NumCols() == value_dim + context_dim) {
    CuSubMatrix<BaseFloat> output_deriv_context_part(
        output_deriv, 0, num_output_rows, value_dim, context_dim);
    // this is the backprop w.r.t. the
    // forward-pass statement: output_context_part.CopyFromMat(*c);
    c_deriv.AddMat(1.0, output_deriv_context_part);
  }

  // Propagate the derivatives back through the softmax nonlinearity,
  // in-place; this is the backprop w.r.t. the statement
  // 'c->SoftMaxPerRow(*c);'.  From this point on, c_deriv actually
  // contains the derivative to the pre-softmax values which we call
  // 'b' in the math.
  c_deriv.DiffSoftmaxPerRow(c, c_deriv);


  CuSubMatrix<BaseFloat> queries_key_part(
      queries, 0, num_output_rows,
      0, key_dim),
      queries_key_part_deriv(
          *queries_deriv, 0, num_output_rows,
          0, key_dim),
      queries_context_part_deriv(
          *queries_deriv, 0, num_output_rows,
          key_dim, context_dim);

  // Below is the backprop corresponding to the forward-propagation command:
  // c->AddMat(1.0, queries_context_part)
  queries_context_part_deriv.AddMat(1.0, c_deriv);

  // The following statement is the part of the backprop w.r.t. the
  // statement:
  // GetAttentionDotProducts(key_scale, queries_key_part, keys, c);
  // which propagates the derivative back to 'queries_key_part'.
  ApplyScalesToOutput(key_scale, keys, c_deriv, &queries_key_part_deriv);

  // The following statement is the part of the backprop w.r.t. the
  // statement:
  // GetAttentionDotProducts(key_scale, queries_key_part, keys, c);
  // which propagates the derivative back to 'keys'.
  ApplyScalesToInput(key_scale, queries_key_part, c_deriv, keys_deriv);

  // The followign statement is the part of the backprop w.r.t.
  // the statement:
  // ApplyScalesToOutput(1.0, values, *c, &output_values_part);
  // which propagates the derivative back to 'values'.
  ApplyScalesToInput(1.0, output_values_part_deriv, c,  values_deriv);
}

// Following code is for full-self-attention as in Google's paper "Attention is all you need"  --Zhichao Wang 2019.08.22

/*
void GetMask(const time_height_convolution::ConvolutionComputationIo &io, 
			 CuMatrixBase<BaseFloat> *Cmask1,
			 CuMatrixBase<BaseFloat> *Cmask2) {
  KALDI_ASSERT(io.num_images > 0 && io.num_t_in > 0 &&
	           io.num_t_out > 0 && io.num_t_in >= io.num_t_out &&
			   Cmask1->NumRows() == Cmask2->NumRows() &&
			   Cmask1->NumCols() == Cmask2->NumCols());

  int32 batch_size = io.num_images;
  int32 input_len = io.num_t_in;
  int32 output_len = io.num_t_out;
  
  Matrix<BaseFloat> mask1(Cmask1->NumRows(), Cmask1->NumCols());
  Matrix<BaseFloat> mask2(Cmask2->NumRows(), Cmask2->NumCols());

  Matrix<BaseFloat> one_block_mask1(batch_size, batch_size * input_len);  // contains one batch of one frame
  Matrix<BaseFloat> one_block_mask2(batch_size, batch_size * input_len);

  one_block_mask2.Set(-1000.0);   // Default value 0.0. This mask for softmax computation.


  for (int32 i = 0; i < batch_size; i++) {
	SubVector<BaseFloat> this_mask1(one_block_mask1, i);
	SubVector<BaseFloat> this_mask2(one_block_mask2, i);
	for (int32 j = 0; j < input_len; j++) {
	  this_mask1(batch_size * j + i) = 1.0;
	  this_mask2(batch_size * j + i) = 0.0;
	}
  }

  for (int32 i = 0; i < output_len; i++) {
	SubMatrix<BaseFloat> mask1_part(mask1, i * batch_size, batch_size, 
	                                0, batch_size * input_len);
	SubMatrix<BaseFloat> mask2_part(mask2, i * batch_size, batch_size,
	                                0, batch_size * input_len);
	mask1_part.CopyFromMat(one_block_mask1);
	mask2_part.CopyFromMat(one_block_mask2);
  }

  Cmask1->CopyFromMat(mask1);
  Cmask2->CopyFromMat(mask2);
}

void GetFullAttentionDotProducts(const time_height_convolution::ConvolutionComputationIo &io,
								 BaseFloat alpha,
                                 const CuMatrixBase<BaseFloat> &A,
                                 const CuMatrixBase<BaseFloat> &B,
                                 CuMatrixBase<BaseFloat> *C) {
  KALDI_ASSERT(A.NumCols() == B.NumCols() &&
               A.NumRows() == C->NumRows());

  int32 num_extra_rows = B.NumRows() - A.NumRows();
  KALDI_ASSERT(num_extra_rows >= 0 );
  
  C->AddMatMat(alpha, A, kNoTrans, B, kTrans, 0.0);

  CuMatrix<BaseFloat> Cmask1(C->NumRows(), C->NumCols(),
                             kUndefined);
  CuMatrix<BaseFloat> Cmask2(C->NumRows(), C->NumCols(),
                             kUndefined);
  GetMask(io, &Cmask1, &Cmask2);

  C->MulElements(Cmask1);
  C->AddMat(1.0, Cmask2);
}
*/

void GetMask(const time_height_convolution::ConvolutionComputationIo &io, 
			 CuMatrixBase<BaseFloat> *Cmask,
			 BaseFloat value) {
  KALDI_ASSERT(io.num_images > 0 && io.num_t_in > 0 &&
	           io.num_t_out > 0 && io.num_t_in >= io.num_t_out &&
			   Cmask->NumRows() == io.num_images * io.num_t_out &&
			   Cmask->NumCols() == io.num_images * io.num_t_in);

  int32 batch_size = io.num_images;
  int32 input_len = io.num_t_in;
  int32 output_len = io.num_t_out;

  CuSubMatrix<BaseFloat> mask_part(*Cmask, 0, batch_size, 0, batch_size * input_len);
  mask_part.SetMask(batch_size, input_len, value);
  for (int i = 1; i < output_len; i++)
  {
	CuSubMatrix<BaseFloat> sub_mask(*Cmask, batch_size * i, batch_size, 0, batch_size * input_len);
	sub_mask.CopyFromMat(mask_part);
  }
}

void GetFullAttentionDotProducts(const time_height_convolution::ConvolutionComputationIo &io,
								 BaseFloat alpha,
                                 const CuMatrixBase<BaseFloat> &A,
                                 const CuMatrixBase<BaseFloat> &B,
                                 CuMatrixBase<BaseFloat> *C) {
  KALDI_ASSERT(A.NumCols() == B.NumCols() &&
               A.NumRows() == C->NumRows());

  int32 num_extra_rows = B.NumRows() - A.NumRows();
  KALDI_ASSERT(num_extra_rows >= 0 );
  
  C->AddMatMat(alpha, A, kNoTrans, B, kTrans, 0.0);

  CuMatrix<BaseFloat> Cmask1(C->NumRows(), C->NumCols(),
                             kUndefined);
  CuMatrix<BaseFloat> Cmask2(C->NumRows(), C->NumCols(),
                             kUndefined);

  Cmask1.SetZero();
  Cmask2.Set(-1000.0);

  GetMask(io, &Cmask1, 1.0);
  GetMask(io, &Cmask2, 0.0);

/// use GPU to generate the whole matrix;
//  int32 batch_size = io.num_images;
//  int32 input_len = io.num_t_in;
//  int32 output_len = io.num_t_out;
//  Cmask1.SetMask(batch_size, input_len, 1.0);
//  Cmask2.SetMask(batch_size, input_len, 0.0);

/// end 

/*
  /// test code:
//  int32 batch_size = io.num_images;
//  int32 input_len = io.num_t_in;
//  int32 output_len = io.num_t_out;
  
  Matrix<BaseFloat> mask1(Cmask1.NumRows(), Cmask1.NumCols());
  Matrix<BaseFloat> mask2(Cmask2.NumRows(), Cmask2.NumCols());
  Matrix<BaseFloat> mask1_gpu(Cmask1);
  Matrix<BaseFloat> mask2_gpu(Cmask2);

  Matrix<BaseFloat> one_block_mask1(batch_size, batch_size * input_len);  // contains one batch of one frame
  Matrix<BaseFloat> one_block_mask2(batch_size, batch_size * input_len);

  one_block_mask2.Set(-1000.0);   // Default value 0.0. This mask for softmax computation.


  for (int32 i = 0; i < batch_size; i++) {
	SubVector<BaseFloat> this_mask1(one_block_mask1, i);
	SubVector<BaseFloat> this_mask2(one_block_mask2, i);
	for (int32 j = 0; j < input_len; j++) {
	  this_mask1(batch_size * j + i) = 1.0;
	  this_mask2(batch_size * j + i) = 0.0;
	}
  }

  for (int32 i = 0; i < output_len; i++) {
	SubMatrix<BaseFloat> mask1_part(mask1, i * batch_size, batch_size, 
	                                0, batch_size * input_len);
	SubMatrix<BaseFloat> mask2_part(mask2, i * batch_size, batch_size,
	                                0, batch_size * input_len);
	mask1_part.CopyFromMat(one_block_mask1);
	mask2_part.CopyFromMat(one_block_mask2);
  }

  printf("============ GPU MASK=1.0 =============\n");
  for (int i = 0; i < batch_size * 2; i++) {
	for (int j = 0; j < mask1_gpu.NumCols(); j++) {
	  printf("%f ", mask1_gpu(i,j));
	}
	printf("\n");
  }

  printf("============ CPU MASK=1.0 =============\n");
  for (int i = 0; i < batch_size * 2; i++) {
	for (int j = 0; j < mask1.NumCols(); j++) {
	  printf("%f ", mask1(i,j));
	}
	printf("\n");
  }
*/
  /// end test

  C->MulElements(Cmask1);
  C->AddMat(1.0, Cmask2);
}

void FullAttentionForward(const time_height_convolution::ConvolutionComputationIo &io,
						  BaseFloat key_scale,
                          const CuMatrixBase<BaseFloat> &keys,
                          const CuMatrixBase<BaseFloat> &queries,
                          const CuMatrixBase<BaseFloat> &values,
                          CuMatrixBase<BaseFloat> *c,
                          CuMatrixBase<BaseFloat> *output) {
  // First check the dimensions and values.
  KALDI_ASSERT(key_scale > 0.0);
  int32 num_input_rows = keys.NumRows(),
      key_dim = keys.NumCols(),
      num_output_rows = queries.NumRows(),
      value_dim = values.NumCols();
  KALDI_ASSERT(num_input_rows > 0 && key_dim > 0 &&
               num_input_rows >= num_output_rows &&
               values.NumRows() == num_input_rows);
  KALDI_ASSERT(c->NumRows() == num_output_rows &&
               c->NumCols() == num_input_rows);
  KALDI_ASSERT(output->NumRows() == num_output_rows &&
               output->NumCols() == value_dim );


  GetFullAttentionDotProducts(io, key_scale,
                          queries,
                          keys, c);

  // compute the soft-max function.  Up till this point, 'c'
  // actually contained what in attention.h we called 'b', which is
  // the input to the softmax.
  c->SoftMaxPerRow(*c);

  // compute the output
  output->AddMatMat(1.0, *c, kNoTrans, values, kNoTrans, 1.0);
}

void FullAttentionBackward(BaseFloat key_scale,
                       const CuMatrixBase<BaseFloat> &keys,
                       const CuMatrixBase<BaseFloat> &queries,
                       const CuMatrixBase<BaseFloat> &values,
                       const CuMatrixBase<BaseFloat> &c,
                       const CuMatrixBase<BaseFloat> &output_deriv,
                       CuMatrixBase<BaseFloat> *keys_deriv,
                       CuMatrixBase<BaseFloat> *queries_deriv,
                       CuMatrixBase<BaseFloat> *values_deriv) {

  // First check the dimensions and values.
  KALDI_ASSERT(key_scale > 0.0);
  int32 num_input_rows = keys.NumRows(),
      key_dim = keys.NumCols(),
      num_output_rows = queries.NumRows(),
      value_dim = values.NumCols();
  KALDI_ASSERT(num_input_rows > 0 && key_dim > 0 &&
               num_input_rows >= num_output_rows &&
               values.NumRows() == num_input_rows);
  KALDI_ASSERT(SameDim(keys, *keys_deriv) &&
               SameDim(queries, *queries_deriv) &&
               SameDim(values, *values_deriv));

  KALDI_ASSERT(c.NumRows() == num_output_rows &&
               c.NumCols() == num_input_rows);
  KALDI_ASSERT(output_deriv.NumRows() == num_output_rows &&
               output_deriv.NumCols() == value_dim);

  CuMatrix<BaseFloat> c_deriv(num_output_rows, num_input_rows,
                              kUndefined);

  // This is the backprop w.r.t. the forward-pass statement:
  // output->AddMatMat(1.0, *c, kNoTrans, values, kNoTrans, 1.0);
  c_deriv.AddMatMat(1.0, output_deriv, kNoTrans, values, kTrans, 0.0);

  Matrix<BaseFloat> cpu_c_deriv(c_deriv);

  // Propagate the derivatives back through the softmax nonlinearity,
  // in-place; this is the backprop w.r.t. the statement
  // 'c->SoftMaxPerRow(*c);'.  From this point on, c_deriv actually
  // contains the derivative to the pre-softmax values which we call
  // 'b' in the math.
  c_deriv.DiffSoftmaxPerRow(c, c_deriv);

  cpu_c_deriv.CopyFromMat(c_deriv);

  // The following statement is the part of the backprop w.r.t. the
  // statement:
  // GetFullAttentionDotProducts(io, key_scale, queries_key_part, keys, c);
  // which propagates the derivative back to 'queries_key_part'.
  queries_deriv->AddMatMat(key_scale, c_deriv, kNoTrans, keys, kNoTrans, 0.0);

  // The following statement is the part of the backprop w.r.t. the
  // statement:
  // GetAttentionDotProducts(key_scale, queries_key_part, keys, c);
  // which propagates the derivative back to 'keys'.
  keys_deriv->AddMatMat(key_scale, c_deriv, kTrans, queries, kNoTrans, 0.0);

  // The followign statement is the part of the backprop w.r.t.
  // the statement:
  // ApplyScalesToOutput(1.0, values, *c, &output_values_part);
  // which propagates the derivative back to 'values'.
  values_deriv->AddMatMat(1.0, c, kTrans, output_deriv, kNoTrans, 0.0);
}

} // namespace attention
namespace contextless3attention {


void GetAttentionDotProducts(BaseFloat alpha,
                             const CuMatrixBase<BaseFloat> &A,
                             const CuMatrixBase<BaseFloat> &B,
                             CuMatrixBase<BaseFloat> *C,
                             int32 row_shift) {
  KALDI_ASSERT(A.NumCols() == B.NumCols() &&
               A.NumRows() == C->NumRows() &&
               0 == C->NumRows() % row_shift);
/*
  int32 num_output_rows = A.NumRows(),
      input_num_cols = A.NumCols(),
      num_extra_rows = B.NumRows() - A.NumRows(),
      context_dim = C->NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C->NumCols(),
                             C->NumRows());
*/
  for (int32 o = 0; o < row_shift; o++) {
    CuSubMatrix<BaseFloat> A_part(A.Row(o).Data(), A.NumRows() / row_shift, A.NumCols(), row_shift * A.Stride()),
                           B_part(B.Row(o).Data(), B.NumRows() / row_shift, B.NumCols(), row_shift * B.Stride()),
                           C_part(C->Row(o).Data(), C->NumRows() / row_shift, C->NumCols(), row_shift * C->Stride());
    C_part.AddMatMat(alpha, A_part, kNoTrans, B_part, kTrans, 0.0);
/*
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    c_col.AddDiagMatMat(alpha, A, kNoTrans, B_part, kTrans, 0.0);
*/
  }
//  C->CopyFromMat(Ctrans, kTrans);
}

void ApplyScalesToOutput(BaseFloat alpha,
                         const CuMatrixBase<BaseFloat> &B,
                         const CuMatrixBase<BaseFloat> &C,
                         CuMatrixBase<BaseFloat> *A,
                         int32 row_shift) {
  KALDI_ASSERT(A->NumCols() == B.NumCols() &&
               A->NumRows() == C.NumRows());
/*
  int32 num_output_rows = A->NumRows(),
      input_num_cols = A->NumCols(),
      num_extra_rows = B.NumRows() - A->NumRows(),
      context_dim = C.NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C, kTrans);
*/
  for (int32 o = 0; o < row_shift; o++) {
    CuSubMatrix<BaseFloat> A_part(A->Row(o).Data(), A->NumRows() / row_shift, A->NumCols(), row_shift * A->Stride()),
                           B_part(B.Row(o).Data(), B.NumRows() / row_shift, B.NumCols(), row_shift * B.Stride()),
                           C_part(C.Row(o).Data(), C.NumRows() / row_shift, C.NumCols(), row_shift * C.Stride());
    A_part.AddMatMat(alpha, C_part, kNoTrans, B_part, kNoTrans, 0.0);
/*
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    A->AddDiagVecMat(alpha, c_col, B_part, kNoTrans, 1.0);
*/
  }
}

void ApplyScalesToInput(BaseFloat alpha,
                        const CuMatrixBase<BaseFloat> &A,
                        const CuMatrixBase<BaseFloat> &C,
                        CuMatrixBase<BaseFloat> *B,
                        int32 row_shift) {
  KALDI_ASSERT(A.NumCols() == B->NumCols() &&
               A.NumRows() == C.NumRows());
/*
  int32 num_output_rows = A.NumRows(),
      input_num_cols = A.NumCols(),
      num_extra_rows = B->NumRows() - A.NumRows(),
      context_dim = C.NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C, kTrans);
*/
  for (int32 o = 0; o < row_shift; o++) {
    CuSubMatrix<BaseFloat> A_part(A.Row(o).Data(), A.NumRows() / row_shift, A.NumCols(), row_shift * A.Stride()),
                           B_part(B->Row(o).Data(), B->NumRows() / row_shift, B->NumCols(), row_shift * B->Stride()),
                           C_part(C.Row(o).Data(), C.NumRows() / row_shift, C.NumCols(), row_shift * C.Stride());
    B_part.AddMatMat(alpha, C_part, kNoTrans, A_part, kNoTrans, 0.0);
/*
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(*B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    B_part.AddDiagVecMat(alpha, c_col, A, kNoTrans, 1.0);
*/
  }
}
void ApplyScalesToInputTrans(BaseFloat alpha,
                        const CuMatrixBase<BaseFloat> &A,
                        const CuMatrixBase<BaseFloat> &C,
                        CuMatrixBase<BaseFloat> *B,
                        int32 row_shift) {
  KALDI_ASSERT(A.NumCols() == B->NumCols() &&
               A.NumRows() == C.NumRows());
/*
  int32 num_output_rows = A.NumRows(),
      input_num_cols = A.NumCols(),
      num_extra_rows = B->NumRows() - A.NumRows(),
      context_dim = C.NumCols();
  KALDI_ASSERT(num_extra_rows > 0 && num_extra_rows % (context_dim - 1) == 0);
  int32 row_shift = num_extra_rows / (context_dim - 1);
  CuMatrix<BaseFloat> Ctrans(C, kTrans);
*/
  for (int32 o = 0; o < row_shift; o++) {
    CuSubMatrix<BaseFloat> A_part(A.Row(o).Data(), A.NumRows() / row_shift, A.NumCols(), row_shift * A.Stride()),
                           B_part(B->Row(o).Data(), B->NumRows() / row_shift, B->NumCols(), row_shift * B->Stride()),
                           C_part(C.Row(o).Data(), C.NumRows() / row_shift, C.NumCols(), row_shift * C.Stride());
    B_part.AddMatMat(alpha, C_part, kTrans, A_part, kNoTrans, 0.0);
/*
    CuSubVector<BaseFloat> c_col(Ctrans, o);
    CuSubMatrix<BaseFloat> B_part(*B, o * row_shift, num_output_rows,
                                  0, input_num_cols);
    B_part.AddDiagVecMat(alpha, c_col, A, kNoTrans, 1.0);
*/
  }
}

void AttentionForward(BaseFloat key_scale,
                      const CuMatrixBase<BaseFloat> &keys,
                      const CuMatrixBase<BaseFloat> &queries,
                      const CuMatrixBase<BaseFloat> &values,
                      CuMatrixBase<BaseFloat> *c,
                      CuMatrixBase<BaseFloat> *output,
                      int32 row_shift,
                      int32 effective_context_dim) {
  // First check the dimensions and values.
  KALDI_ASSERT(key_scale > 0.0);
  int32 num_input_rows = keys.NumRows(),
      key_dim = keys.NumCols(),
      num_output_rows = queries.NumRows(),
//      context_dim = queries.NumRows(),
      context_dim = c->NumCols(),
      value_dim = values.NumCols();
  bool add_context = (queries.NumCols() - key_dim)?true:false;
//      row_shift = queries.NumRows() / context_dim;
  KALDI_ASSERT(num_input_rows > 0 && key_dim > 0 &&
               num_input_rows >= num_output_rows &&
               context_dim > 0 &&
//               (num_input_rows - num_output_rows) % (context_dim - 1) == 0 &&
               values.NumRows() == num_input_rows);
  KALDI_ASSERT(c->NumRows() == num_output_rows);
//  &&               c->NumCols() == context_dim);
  KALDI_ASSERT(output->NumRows() == num_output_rows &&
               (output->NumCols() == value_dim ||
                output->NumCols() == value_dim + context_dim));

  if(add_context){
    CuSubMatrix<BaseFloat> queries_key_part(
      queries, 0, num_output_rows,
      0, key_dim),
      queries_context_part(
          queries, 0, num_output_rows,
          key_dim, context_dim);
 
  GetAttentionDotProducts(key_scale,
                          queries_key_part,
                          keys, c, row_shift);
  // think of 'queries_context_part' as a position-dependent bias term.
  c->AddMat(1.0, queries_context_part);
  }
  else{
  GetAttentionDotProducts(key_scale,
                          queries,
                          keys, c, row_shift);
  
  }
  // compute the soft-max function.  Up till this point, 'c'
  // actually contained what in attention.h we called 'b', which is
  // the input to the softmax.
  CuSubMatrix<BaseFloat> c_(*c, 0, c->NumRows(), 0, effective_context_dim);
  c_.SoftMaxPerRow(c_);
  //c->SoftMaxPerRow(*c);


  // the part of the output that is weighted
  // combinations of the input values.
  CuSubMatrix<BaseFloat> output_values_part(
      *output, 0, num_output_rows, 0, value_dim);

  ApplyScalesToOutput(1.0, values, *c, &output_values_part, row_shift);


  if (output->NumCols() == value_dim + context_dim) {
    CuSubMatrix<BaseFloat> output_context_part(
        *output, 0, num_output_rows, value_dim, context_dim);
    output_context_part.CopyFromMat(*c);
  }
}

void AttentionBackward(BaseFloat key_scale,
                       const CuMatrixBase<BaseFloat> &keys,
                       const CuMatrixBase<BaseFloat> &queries,
                       const CuMatrixBase<BaseFloat> &values,
                       const CuMatrixBase<BaseFloat> &c,
                       const CuMatrixBase<BaseFloat> &output_deriv,
                       CuMatrixBase<BaseFloat> *keys_deriv,
                       CuMatrixBase<BaseFloat> *queries_deriv,
                       CuMatrixBase<BaseFloat> *values_deriv,
                      int32 row_shift,
                      int32 effective_context_dim) {

  // First check the dimensions and values.
  KALDI_ASSERT(key_scale > 0.0);
  int32 num_input_rows = keys.NumRows(),
      key_dim = keys.NumCols(),
      num_output_rows = queries.NumRows(),
//      context_dim = queries.NumCols() - key_dim,
      context_dim = c.NumCols(),
//      context_dim = queries.NumRows(),
      value_dim = values.NumCols();
  bool add_context = (queries.NumCols() - key_dim > 0)?true:false;

  KALDI_ASSERT(num_input_rows > 0 && key_dim > 0 &&
               num_input_rows >= num_output_rows &&
               context_dim > 0 &&
//               (num_input_rows - num_output_rows) % (context_dim - 1) == 0 &&
               values.NumRows() == num_input_rows);
  KALDI_ASSERT(SameDim(keys, *keys_deriv) &&
               SameDim(queries, *queries_deriv) &&
               SameDim(values, *values_deriv));

  KALDI_ASSERT(c.NumRows() == num_output_rows &&
               c.NumCols() == context_dim);
  KALDI_ASSERT(output_deriv.NumRows() == num_output_rows &&
               (output_deriv.NumCols() == value_dim ||
                output_deriv.NumCols() == value_dim + context_dim));

  CuMatrix<BaseFloat> c_deriv(num_output_rows, context_dim,
                              kUndefined);

  CuSubMatrix<BaseFloat> output_values_part_deriv(
      output_deriv, 0, num_output_rows, 0, value_dim);
  // This is the backprop w.r.t. the forward-pass statement:
  // ApplyScalesToOutput(1.0, values, *c, &output_values_part);
  GetAttentionDotProducts(1.0, output_values_part_deriv, values, 
                          &c_deriv, row_shift);

  if (output_deriv.NumCols() == value_dim + context_dim) {
    CuSubMatrix<BaseFloat> output_deriv_context_part(
        output_deriv, 0, num_output_rows, value_dim, context_dim);
    // this is the backprop w.r.t. the
    // forward-pass statement: output_context_part.CopyFromMat(*c);
    c_deriv.AddMat(1.0, output_deriv_context_part);
  }

  // Propagate the derivatives back through the softmax nonlinearity,
  // in-place; this is the backprop w.r.t. the statement
  // 'c->SoftMaxPerRow(*c);'.  From this point on, c_deriv actually
  // contains the derivative to the pre-softmax values which we call
  // 'b' in the math.
  CuSubMatrix<BaseFloat> c_deriv_(c_deriv, 0, c_deriv.NumRows(), 0, effective_context_dim);
  //c_deriv.DiffSoftmaxPerRow(c, c_deriv);
  c_deriv_.DiffSoftmaxPerRow(c, c_deriv_);

  if (add_context){
    CuSubMatrix<BaseFloat> queries_key_part(
        queries, 0, num_output_rows,
        0, key_dim),
        queries_key_part_deriv(
            *queries_deriv, 0, num_output_rows,
            0, key_dim),
        queries_context_part_deriv(
            *queries_deriv, 0, num_output_rows,
            key_dim, context_dim);
    queries_context_part_deriv.AddMat(1.0, c_deriv);
    ApplyScalesToOutput(key_scale, keys, c_deriv, &queries_key_part_deriv, row_shift);
    ApplyScalesToInputTrans(key_scale, queries_key_part, c_deriv, keys_deriv, row_shift);
    //ApplyScalesToInput(key_scale, queries_key_part, c_deriv, keys_deriv, row_shift);
  }
  else{
    ApplyScalesToOutput(key_scale, keys, c_deriv, queries_deriv, row_shift);
    //ApplyScalesToInput(key_scale, queries, c_deriv, keys_deriv, row_shift);
    ApplyScalesToInputTrans(key_scale, queries, c_deriv, keys_deriv, row_shift);
  }
  ApplyScalesToInputTrans(1.0, output_values_part_deriv, c,  values_deriv, row_shift);
}

} // namespace contextless3attention
} // namespace nnet3
} // namespace kaldi
