// nnet/nnet-component-test.cc
// Copyright 2014-2015  Brno University of Technology (author: Karel Vesely),
//                      The Johns Hopkins University (author: Sri Harish Mallidi)

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

#include <sstream>
#include <fstream>
#include <algorithm>

#include "nnet/nnet-component.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-convolutional-component.h"
#include "nnet/nnet-convolutional-2d-component.h"
#include "nnet/nnet-max-pooling-component.h"
#include "nnet/nnet-max-pooling-2d-component.h"
#include "nnet/nnet-average-pooling-2d-component.h"
#include "nnet/nnet-simple-recurrent-unit.h"
#include "nnet/nnet-batch-norm-component.h"
#include "nnet/nnet-embedding.h"
#include "nnet/nnet-tf-lstm.h"
#include "nnet/nnet-bi-compact-vfsmn.h"
#include "util/common-utils.h"

namespace kaldi {
namespace nnet1 {

  /*
   * Helper functions
   */
  template<typename Real>
  void ReadCuMatrixFromString(const std::string& s, CuMatrix<Real>* m) {
    std::istringstream is(s + "\n");
    m->Read(is, false);  // false for ascii
  }

  Component* ReadComponentFromString(const std::string& s) {
    std::istringstream is(s + "\n");
    return Component::Read(is, false);  // false for ascii
  }


  /*
   * Unit tests,
   */
  void UnitTestLengthNorm() {
    // make L2-length normalization component,
    Component* c = ReadComponentFromString("<LengthNormComponent> 5 5");
    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 2 3 4 5 \n 2 3 5 6 8 ] ", &mat_in);
    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    // check the length,
    mat_out.MulElements(mat_out);  // ^2,
    CuVector<BaseFloat> check_length_is_one(2);
    check_length_is_one.AddColSumMat(1.0, mat_out, 0.0);  // sum_of_cols(x^2),
    check_length_is_one.ApplyPow(0.5);  // L2norm = sqrt(sum_of_cols(x^2)),
    CuVector<BaseFloat> ones(2);
    ones.Set(1.0);
    AssertEqual(check_length_is_one, ones);
  }

  void UnitTestSimpleSentenceAveragingComponent() {
    // make SimpleSentenceAveraging component,
    Component* c = ReadComponentFromString(
      "<SimpleSentenceAveragingComponent> 2 2 <GradientBoost> 10.0"
    );
    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0.5 \n 1 1 \n 2 1.5 ] ", &mat_in);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    // check the output,
    CuVector<BaseFloat> ones(2);
    ones.Set(1.0);
    for (int32 i = 0; i < mat_out.NumRows(); i++) {
      AssertEqual(mat_out.Row(i), ones);
    }

    // backpropagate,
    CuMatrix<BaseFloat> dummy1(3, 2), dummy2(3, 2), diff_out(mat_in), diff_in;
    // the average 1.0 in 'diff_in' will be boosted by 10.0,
    c->Backpropagate(dummy1, dummy2, diff_out, &diff_in);
    // check the output,
    CuVector<BaseFloat> tens(2); tens.Set(10);
    for (int32 i = 0; i < diff_in.NumRows(); i++) {
      AssertEqual(diff_in.Row(i), tens);
    }
  }

  void UnitTestConvolutionalComponentUnity() {
    // make 'identity' convolutional component,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 5 5 \
      <PatchDim> 1 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
      <MaxNorm> 0 \
      <Filters> [ 1 \
      ] <Bias> [ 0 ]"
    );

    // prepare input,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 2 3 4 5 ] ", &mat_in);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_in, mat_out);

    // backpropagate,
    CuMatrix<BaseFloat> mat_out_diff(mat_in), mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_out_diff " << mat_out_diff
              << " mat_in_diff " << mat_in_diff;
    AssertEqual(mat_out_diff, mat_in_diff);

    // clean,
    delete c;
  }

  void UnitTestConvolutionalComponent3x3() {
    // make 3x3 convolutional component,
    // design such weights and input so output is zero,
    Component* c = ReadComponentFromString("<ConvolutionalComponent> 9 15 \
      <PatchDim> 3 <PatchStep> 1 <PatchStride> 5 \
      <LearnRateCoef> 1.0 <BiasLearnRateCoef> 1.0 \
      <MaxNorm> 0 \
      <Filters> [ -1 -2 -7   0 0 0   1 2 7 ; \
                  -1  0  1  -3 0 3  -2 2 0 ; \
                  -4  0  0  -3 0 3   4 0 0 ] \
      <Bias> [ -20 -20 -20 ]"
    );

    // prepare input, reference output,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1 3 5 7 9  2 4 6 8 10  3 5 7 9 11 ]", &mat_in);
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 0 0 0  0 0 0  0 0 0 ]", &mat_out_ref);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_in" << mat_in << "mat_out" << mat_out;
    AssertEqual(mat_out, mat_out_ref);

    // prepare mat_out_diff, mat_in_diff_ref,
    CuMatrix<BaseFloat> mat_out_diff;
    ReadCuMatrixFromString("[ 1 0 0  1 1 0  1 1 1 ]", &mat_out_diff);
    // hand-computed back-propagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref;
    ReadCuMatrixFromString("[ -1 -4 -15 -8 -6   0 -3 -6 3 6   1 1 14 11 7 ]",
                           &mat_in_diff_ref);

    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    // clean,
    delete c;
  }


  void UnitTestMaxPoolingComponent() {
    // make max-pooling component, assuming 4 conv. neurons,
    // non-overlapping pool of size 3,
    Component* c = Component::Init(
        "<MaxPoolingComponent> <InputDim> 24 <OutputDim> 8 \
         <PoolSize> 3 <PoolStep> 3 <PoolStride> 4"
    );

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 3 8 2 9 \
                              8 3 9 3 \
                              2 4 9 6 \
                              \
                              2 4 2 0 \
                              6 4 9 4 \
                              7 3 0 3;\
                              \
                              5 4 7 8 \
                              3 9 5 6 \
                              3 4 8 9 \
                              \
                              5 4 5 6 \
                              3 1 4 5 \
                              8 2 1 7 ]", &mat_in);

    // expected output (max values in columns),
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 8 8 9 9 \
                              7 4 9 4;\
                              5 9 8 9 \
                              8 4 5 7 ]", &mat_out_ref);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);

    // locations of max values will be shown,
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    mat_out_diff.Set(1);
    // expected backpropagated values (hand-computed),
    CuMatrix<BaseFloat> mat_in_diff_ref;
    ReadCuMatrixFromString("[ 0 1 0 1 \
                              1 0 1 0 \
                              0 0 1 0 \
                              \
                              0 1 0 0 \
                              0 1 1 1 \
                              1 0 0 0;\
                              \
                              1 0 0 0 \
                              0 1 0 0 \
                              0 0 1 1 \
                              \
                              0 1 1 0 \
                              0 0 0 0 \
                              1 0 0 1 ]", &mat_in_diff_ref);
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }

  void UnitTestMaxPooling2DComponent() { /* Implemented by Harish Mallidi */
    // make max-pooling2d component
    Component* c = Component::Init(
      "<MaxPooling2DComponent> <InputDim> 56 <OutputDim> 18 \
       <FmapXLen> 4 <FmapYLen> 7 <PoolXLen> 2 <PoolYLen> 3 \
       <PoolXStep> 1 <PoolYStep> 2"
    );

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 \
      11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 21 21 \
      22 22 23 23 24 24 25 25 26 26 27 27 ]", &mat_in);

    // expected output (max values in the patch)
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 9 9 11 11 13 13 16 16 18 18 \
      20 20 23 23 25 25 27 27 ]", &mat_out_ref);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);


    // locations of max values will be shown
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString(
      "[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 ]", &mat_out_diff
    );

    // expected backpropagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref;  // hand-computed back-propagated values,
    ReadCuMatrixFromString("[ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
      0.25 0.25 0 0 1 1 0 0 0 0 0.75 0.75 0 0 1 1 0 0 2.5 2.5 \
      0 0 0 0 3 3 0 0 3.5 3.5 0 0 8 8 ]", &mat_in_diff_ref
    );

    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }

  void UnitTestAveragePooling2DComponent() { /* Implemented by Harish Mallidi */
    // make average-pooling2d component
    Component* c = Component::Init(
      "<AveragePooling2DComponent> <InputDim> 56 <OutputDim> 18 \
       <FmapXLen> 4 <FmapYLen> 7 <PoolXLen> 2 <PoolYLen> 3 \
       <PoolXStep> 1 <PoolYStep> 2"
    );

    // input matrix,
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 \
      11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 \
      21 21 22 22 23 23 24 24 25 25 26 26 27 27 ]", &mat_in);

    // expected output (max values in the patch)
    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 4.5 4.5 6.5 6.5 8.5 8.5 11.5 11.5 13.5 13.5 \
      15.5 15.5 18.5 18.5 20.5 20.5 22.5 22.5 ]", &mat_out_ref);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out_ref" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);


    // locations of max values will be shown
    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 ]", &mat_out_diff);

    // expected backpropagated values,
    CuMatrix<BaseFloat> mat_in_diff_ref;  // hand-computed back-propagated values,
    ReadCuMatrixFromString("[  0 0 0 0 0.0833333 0.0833333 0.166667 0.166667 \
      0.25 0.25 0.333333 0.333333 0.333333 0.333333 0.25 0.25 0.25 0.25 \
      0.333333 0.333333 0.416667 0.416667 0.5 0.5 0.583333 0.583333 0.583333 \
      0.583333 0.75 0.75 0.75 0.75 0.833333 0.833333 0.916667 0.916667 1 1 \
      1.08333 1.08333 1.08333 1.08333 1 1 1 1 1.08333 1.08333 1.16667 1.16667 \
      1.25 1.25 1.33333 1.33333 1.33333 1.33333 ]", &mat_in_diff_ref
    );

    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }


  void UnitTestConvolutional2DComponent() { /* Implemented by Harish Mallidi */
    // Convolutional2D component
    Component* c = ReadComponentFromString("<Convolutional2DComponent> 18 56 \
      <LearnRateCoef> 0 <BiasLearnRateCoef> 0 <FmapXLen> 4 <FmapYLen> 7 \
      <FiltXLen> 2 <FiltYLen> 3 <FiltXStep> 1 <FiltYStep> 2 <ConnectFmap> 1 \
      <Filters> [ 0 0 1 1 2 2 3 3 4 4 5 5 ; 0 0 1 1 2 2 3 3 4 4 5 5 ] \
      <Bias> [ 0 0 ]"
    );

    // input matrix
    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 \
      11 11 12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 \
      21 21 22 22 23 23 24 24 25 25 26 26 27 27 ]", &mat_in);

    CuMatrix<BaseFloat> mat_out_ref;
    ReadCuMatrixFromString("[ 206 206 266 266 326 326 416 416 476 476 536 536 \
      626 626 686 686 746 746 ]", &mat_out_ref);

    // propagate
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out << "mat_out" << mat_out_ref;
    AssertEqual(mat_out, mat_out_ref);

    // prepare mat_out_diff, mat_in_diff_ref,
    CuMatrix<BaseFloat> mat_out_diff;
    ReadCuMatrixFromString("[ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 ]",
                           &mat_out_diff);

    CuMatrix<BaseFloat> mat_in_diff_ref;
    ReadCuMatrixFromString("[ 0 0 0 0 0 0 2 2 2 2 4 4 8 8 0 0 3 3 4.5 4.5 8 8 \
      9.5 9.5 13 13 20 20 9 9 18 18 19.5 19.5 23 23 24.5 24.5 28 28 41 41 \
      36 36 48 48 51 51 56 56 59 59 64 64 80 80 ]", &mat_in_diff_ref);

    // backpropagate
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff
              << " mat_in_diff_ref " << mat_in_diff_ref;
    AssertEqual(mat_in_diff, mat_in_diff_ref);

    delete c;
  }

  void UnitTestDropoutComponent() {
    Component* c = ReadComponentFromString("<Dropout> 100 100 <DropoutRetention> 0.7");
    // buffers,
    CuMatrix<BaseFloat> in(777, 100),
                        out,
                        out_diff,
                        in_diff;
    // init,
    in.Set(2.0);

    // propagate,
    c->Propagate(in, &out);
    AssertEqual(in.Sum(), out.Sum(), 0.01);

    // backprop,
    out_diff = in;
    c->Backpropagate(in, out, out_diff, &in_diff);
    AssertEqual(in_diff, out);

    delete c;
  }

  void UnitTestSimpleRecurrentUnit() { /* Implemented by Kaituo XU */
    Component* cp = Component::Init(
      "<SimpleRecurrentUnit> <InputDim> 3 <OutputDim> 3 \
      <CellDim> 3"
    );
    SimpleRecurrentUnit* c = dynamic_cast<SimpleRecurrentUnit*>(cp);
    auto t = c->GetType();
    KALDI_LOG << c->TypeToMarker(t);
    KALDI_LOG << c->Info();

    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ -0.4 -0.34117647 -0.28235294 \n\
        0.12941176  0.18823529  0.24705882 \n\
       -0.22352941 -0.16470588 -0.10588235 \n\
        0.30588235  0.36470588  0.42352941 \n\
       -0.04705882  0.01176471  0.07058824 \n\
        0.48235294  0.54117647  0.6       ]", &mat_in);
    KALDI_LOG << mat_in.NumRows() << " " << mat_in.NumCols();
    std::vector<int32> seq_len{3, 3};
    c->SetSeqLengths(seq_len);

    CuMatrix<BaseFloat> W_bf_br;
    ReadCuMatrixFromString("[ -0.2         0.17714286  0.55428571 \
       -0.16857143  0.20857143  0.58571429 \
       -0.13714286  0.24        0.61714286 \
       -0.10571429  0.27142857  0.64857143 \
       -0.07428571  0.30285714  0.68       \
       -0.04285714  0.33428571  0.71142857 \
       -0.01142857  0.36571429  0.74285714 \
        0.02        0.39714286  0.77428571 \
        0.05142857  0.42857143  0.80571429 \
        0.08285714  0.46        0.83714286 \
        0.11428571  0.49142857  0.86857143 \
        0.14571429  0.52285714  0.9        \
        0.2   0.45  0.7 \
        0.2   0.45  0.7 ]", &W_bf_br);

    Vector<BaseFloat> para;
    para.Resize(4*3*3+3*2, kSetZero);
    para.CopyRowsFromMat(W_bf_br);
    KALDI_LOG << para;
    c->SetParams(para);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out;
    /* mat_out should be
    [[-0.2596111  -0.25931769 -0.25462568]
     [ 0.15380283  0.14249242  0.12924175]
     [-0.11801852 -0.12430928 -0.1285231 ]
     [ 0.27003811  0.2517609   0.22933191]
     [ 0.02075762  0.00835634 -0.00479826]
     [ 0.37424206  0.3499653   0.31938951]]
    */

    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString("[ -0.6        -0.52941176 -0.45882353 \n\
     0.03529412  0.10588235  0.17647059 \n\
    -0.38823529 -0.31764706 -0.24705882 \n\
     0.24705882  0.31764706  0.38823529 \n\
    -0.17647059 -0.10588235 -0.03529412 \n\
     0.45882353  0.52941176  0.6       ]", &mat_out_diff);
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff;
    /* mat_in_diff should be
    [[ 0.0110512  -0.5594053  -1.1298618 ]
     [-0.05572256  0.11826764  0.29225785]
     [ 0.00324207 -0.26730506 -0.53785218]
     [-0.04434476  0.17014438  0.38463351]
     [ 0.00098253 -0.0717513  -0.14448512]
     [-0.01109324  0.16023669  0.33156661]]
    */
    delete c;
  }

  void UnitTestBatchNormComponent() { /* Implemented by Kaituo Xu */
    int32 N = 200, D = 3;

    Component* cp = Component::Init("<BatchNormComponent> <InputDim> 3 \
    <OutputDim> 3");
    BatchNormComponent* c = dynamic_cast<BatchNormComponent*>(cp);

    // test propagate
    {
      CuMatrix<BaseFloat> mat_in(N, D);
      BaseFloat mean = 1.2, std = 3.5;
      RandGauss(mean, std, &mat_in);
      KALDI_LOG << "Before batch normalization (mean=1.2, std=3.5)";
      KALDI_LOG << MomentStatistics(mat_in);

      {
        CuMatrix<BaseFloat> param(2, 3);
        ReadCuMatrixFromString("[ 1 1 1 \n 0 0 0 ]", &param);
        Vector<BaseFloat> para(6);
        para.CopyRowsFromMat(param);
        c->SetParams(para);

        CuMatrix<BaseFloat> mat_out;
        c->Propagate(mat_in, &mat_out);

        KALDI_LOG << "After batch normalization (gamma=1, beta=0)";
        KALDI_LOG << MomentStatistics(mat_out);
      }
      {
        CuMatrix<BaseFloat> param(2, D);
        ReadCuMatrixFromString("[ 1 2 3 \n 11 12 13 ]", &param);
        Vector<BaseFloat> para(6);
        para.CopyRowsFromMat(param);
        c->SetParams(para);

        CuMatrix<BaseFloat> mat_out;
        c->Propagate(mat_in, &mat_out);

        CuVector<BaseFloat> mean(D), var(D);
        mean.AddRowSumMat(1.0/N, mat_out);
        mat_out.AddVecToRows(-1.0, mean);
        mat_out.ApplyPow(2.0);
        var.AddRowSumMat(1.0/N, mat_out);

        KALDI_LOG << "After batch normalization (gamma=[1 2 3], beta=[11 12 13])";
        // KALDI_LOG << MomentStatistics(mat_out);
        KALDI_LOG << "mean" << mean;
        KALDI_LOG << "var" << var;
      }

      {
        KALDI_LOG << "Now test \"test\" mode";
        CuMatrix<BaseFloat> param(2, D);
        ReadCuMatrixFromString("[ 1 1 1 \n 0 0 0 ]", &param);
        Vector<BaseFloat> para(6);
        para.CopyRowsFromMat(param);
        c->SetParams(para);

        CuMatrix<BaseFloat> mat_out;
        c->SetBatchNormMode("train");
        RandGauss(mean, std, &mat_in);
        for (int i = 0; i < 50; ++i) {
          c->Propagate(mat_in, &mat_out);
        }
        KALDI_LOG << MomentStatistics(mat_out);

        c->SetBatchNormMode("test");
        RandGauss(mean, std, &mat_in);
        c->Propagate(mat_in, &mat_out);
        KALDI_LOG << MomentStatistics(mat_out);
      }
    }
    // test backpropagate
    {
      N = 4;
      D = 3;
      c->SetBatchNormMode("train");
      CuMatrix<BaseFloat> mat_in(N, D);
      ReadCuMatrixFromString("[ 14.20613743  10.34564924  24.15385594 \n\
      10.73953935  12.54804921  19.91240559 \n\
       7.45383798   9.04181671  12.93801613 \n\
      10.35065021   6.03617694  10.97561745 ]", &mat_in);

      CuMatrix<BaseFloat> param(2, D);
      ReadCuMatrixFromString("[ -0.35882895  0.6034716  -1.66478853 \n\
      -0.70017904  1.15139101  1.85733101 ]", &param);
      Vector<BaseFloat> para(2*D);
      para.CopyRowsFromMat(param);
      c->SetParams(para);

      CuMatrix<BaseFloat> mat_out;
      c->Propagate(mat_in, &mat_out);
      KALDI_LOG << "mat_out" << mat_out;
      /* mat_out should be
      [-1.22724083  1.36975815 -0.39042692]
      [-0.707968    1.93375134  0.94131083]
      [-0.21579227  1.03587117  3.13114143]
      [-0.64971505  0.26618338  3.74729868]
      */

      CuMatrix<BaseFloat> mat_out_diff(mat_out);
      ReadCuMatrixFromString("[ -1.51117956  0.64484751 -0.98060789 \n\
      -0.85685315 -0.87187918 -0.42250793 \n\
       0.99643983  0.71242127  0.05914424 \n\
      -0.36331088  0.00328884 -0.10593044 ]", &mat_out_diff);
      // backpropagate,
      CuMatrix<BaseFloat> mat_in_diff;
      c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
      KALDI_LOG << "mat_in_diff " << mat_in_diff;
      /* mat_in_diff should be
      [-0.03290006  0.1578986   0.0370725 ]
      [ 0.06051027 -0.1683892  -0.0451365 ]
      [-0.03566548  0.13842916 -0.04340304]
      [ 0.00805527 -0.12793856  0.05146703]
      */

      Vector<BaseFloat> gradient(2*D);
      c->GetGradient(&gradient);
      KALDI_LOG << "gradient of gamma and beta " << gradient;
      /* gradient should be
      [-3.53228792 -1.03819347 -1.48146646]
      [-1.73490376  0.48867844 -1.44990201]
      */
    }
    delete c;
  }

  void UnitTestEmbedding() { /* Implemented by Kaituo Xu */
    int32 V = 10, D = 5, N = 1, T = 6;

    Component* cp = Component::Init("<Embedding> <InputDim> 1 \
    <OutputDim> 5 <VocabSize> 10");
    Embedding* c = dynamic_cast<Embedding*>(cp);

    KALDI_LOG << "Embedding Matrix";
    CuMatrix<BaseFloat> W(V, D);
    W = c->GetW();
    KALDI_LOG << W;

    CuMatrix<BaseFloat> mat_in(N, T);
    ReadCuMatrixFromString("[ 1 \n2 \n3 \n0 \n9 \n8 ]", &mat_in);
    KALDI_LOG << "mat_in" << mat_in;
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out;

  }

  void UnitTestTfLstm() { /* Implemented by Kaituo XU */
    // Note: <DiffCilp> should not be set too small
    Component* cp = Component::Init(
      "<TfLstm> <InputDim> 3 <OutputDim> 6 \
      <CellDim> 3 <F> 2 <S> 1 <DiffClip> 10"
    );
    TfLstm* c = dynamic_cast<TfLstm*>(cp);
    auto t = c->GetType();
    KALDI_LOG << c->TypeToMarker(t);
    KALDI_LOG << c->Info();

    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ -0.4 -0.34117647 -0.28235294 \n\
        0.12941176  0.18823529  0.24705882 \n\
       -0.22352941 -0.16470588 -0.10588235 \n\
        0.30588235  0.36470588  0.42352941 \n\
       -0.04705882  0.01176471  0.07058824 \n\
        0.48235294  0.54117647  0.6       ]", &mat_in);
    KALDI_LOG << mat_in.NumRows() << " " << mat_in.NumCols();
    std::vector<int32> seq_len{3, 3};
    c->SetSeqLengths(seq_len);

    CuMatrix<BaseFloat> Wx_Wr_Wk_b_pi_pf_po;
    ReadCuMatrixFromString("[ -0.2         0.37391304 \
        -0.15217391  0.42173913 \
        -0.10434783  0.46956522 \
        -0.05652174  0.5173913  \
        -0.00869565  0.56521739 \
         0.03913043  0.61304348 \
         0.08695652  0.66086957 \
         0.13478261  0.70869565 \
         0.1826087   0.75652174 \
         0.23043478  0.80434783 \
         0.27826087  0.85217391 \
         0.32608696  0.9        \
         -0.3         0.07714286  0.45428571 \
         -0.26857143  0.10857143  0.48571429 \
         -0.23714286  0.14        0.51714286 \
         -0.20571429  0.17142857  0.54857143 \
         -0.17428571  0.20285714  0.58       \
         -0.14285714  0.23428571  0.61142857 \
         -0.11142857  0.26571429  0.64285714 \
         -0.08        0.29714286  0.67428571 \
         -0.04857143  0.32857143  0.70571429 \
         -0.01714286  0.36        0.73714286 \
          0.01428571  0.39142857  0.76857143 \
          0.04571429  0.42285714  0.8        \
            -0.4        -0.02285714  0.35428571 \
            -0.36857143  0.00857143  0.38571429 \
            -0.33714286  0.04        0.41714286 \
            -0.30571429  0.07142857  0.44857143 \
            -0.27428571  0.10285714  0.48       \
            -0.24285714  0.13428571  0.51142857 \
            -0.21142857  0.16571429  0.54285714 \
            -0.18        0.19714286  0.57428571 \
            -0.14857143  0.22857143  0.60571429 \
            -0.11714286  0.26        0.63714286 \
            -0.08571429  0.29142857  0.66857143 \
            -0.05428571  0.32285714  0.7        \
           0.2         0.24545455  0.29090909  0.33636364  0.38181818  0.42727273 \
          0.47272727  0.51818182  0.56363636  0.60909091  0.65454545  0.7       \
          0.1   0.35  0.6 \
          0.2   0.45  0.7 \
          0.3   0.55  0.8 ]", &Wx_Wr_Wk_b_pi_pf_po);
    Vector<BaseFloat> para;
    int32 F = 2, H = 3;
    para.Resize(F*4*H+H*4*H*2+4*H+H*3, kSetZero);
    para.CopyRowsFromMat(Wx_Wr_Wk_b_pi_pf_po);
    KALDI_LOG << para;
    c->SetParams(para);

    std::ofstream ofs("temp.nnet");
    c->WriteData(ofs, false);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out;
    /* mat_out should be
      [[ 0.04662942  0.05057702  0.05469498  0.05276928  0.06095797  0.06962654]
       [ 0.10042051  0.13082645  0.1634004   0.11976994  0.1616913   0.2060733 ]
       [ 0.10098896  0.11988935  0.14068334  0.12327551  0.15602406  0.19243518]
       [ 0.23211234  0.31130399  0.39585347  0.30889627  0.42007649  0.52788134]
       [ 0.17568366  0.22313201  0.2768604   0.23822603  0.31793715  0.4049018 ]
       [ 0.4109416   0.54324051  0.66706117  0.56601571  0.71869179  0.82716888]]
     */

    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString("[ -0.6        -0.56571429 -0.53142857 -0.49714286 -0.46285714 -0.42857143 \n\
             0.01714286  0.05142857  0.08571429  0.12        0.15428571  0.18857143 \n\
            -0.39428571 -0.36       -0.32571429 -0.29142857 -0.25714286 -0.22285714 \n\
             0.22285714  0.25714286  0.29142857  0.32571429  0.36        0.39428571 \n\
            -0.18857143 -0.15428571 -0.12       -0.08571429 -0.05142857 -0.01714286 \n\
             0.42857143  0.46285714  0.49714286  0.53142857  0.56571429  0.6 ]", &mat_out_diff);
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff;
    /* mat_in_diff should be
      [[ 0.11938887 -0.45243866 -0.39861172]
       [-0.08808162  0.57554339  0.66244764]
       [ 0.06153257 -0.31880107 -0.21816791]
       [-0.06731306  0.52782637  0.54888115]
       [ 0.01872825 -0.11813731 -0.04066477]
       [-0.03206199  0.29631486  0.25675242]]
    */
    delete c;
  }

  void UnitTestBiCompactVfsmn() { /* Implemented by Kaituo XU */
    Component* cp = Component::Init(
      "<BiCompactVfsmn> <InputDim> 5 <OutputDim> 5 <BackOrder> 4 <AheadOrder> 3"
    );
    BiCompactVfsmn* c = dynamic_cast<BiCompactVfsmn*>(cp);
    auto t = c->GetType();
    KALDI_LOG << c->TypeToMarker(t);
    KALDI_LOG << c->Info();

    CuMatrix<BaseFloat> mat_in;
    ReadCuMatrixFromString("[ 1.   2.   3.   4.   5. \n\
          6.   7.   8.   9.  10. \n\
         11.  12.  13.  14.  15. \n\
         16.  17.  18.  19.  20. \n\
         21.  22.  23.  24.  25. \n\
         26.  27.  28.  29.  30. \n\
         31.  32.  33.  34.  35. \n\
         36.  37.  38.  39.  40. \n\
         41.  42.  43.  44.  45. \n\
         46.  47.  48.  49.  50. \n\
         51.  52.  53.  54.  55. \n\
         56.  57.  58.  59.  60. \n\
         61.  62.  63.  64.  65. \n\
         66.  67.  68.  69.  70. \n\
         71.  72.  73.  74.  75. ]", &mat_in);
    KALDI_LOG << mat_in.NumRows() << " " << mat_in.NumCols();
    KALDI_LOG << mat_in;

    CuMatrix<BaseFloat> bposition, fposition;
    ReadCuMatrixFromString("[ 0 \n 1 \n 2 \n 3 \n 4 \n 5 \n 6 \n 7 \n 0 \n 1 \n 2 \n 3 \n 4 \n 0 \n 1 ]", &bposition);
    ReadCuMatrixFromString("[ 7 \n 6 \n 5 \n 4 \n 3 \n 2 \n 1 \n 0 \n 4 \n 3 \n 2 \n 1 \n 0 \n 1 \n 0 ]", &fposition);
    KALDI_LOG << bposition.NumRows() << " " << bposition.NumCols();
    KALDI_LOG << fposition.NumRows() << " " << fposition.NumCols();
    KALDI_LOG << bposition;
    KALDI_LOG << fposition;

    // Prepare extra info
    ExtraInfo info(bposition, fposition);
    c->Prepare(info);

    CuMatrix<BaseFloat> filter;
    ReadCuMatrixFromString("[    0.1  0.2  0.3  0.4  0.5 \n\
         0.6  0.7  0.8  0.9  1.  \n\
         1.1  1.2  1.3  1.4  1.5 \n\
         1.6  1.7  1.8  1.9  2.  \n\
         2.1  2.2  2.3  2.4  2.5 \n\
         3.          3.10714286  3.21428571  3.32142857  3.42857143 \n\
         3.53571429  3.64285714  3.75        3.85714286  3.96428571 \n\
         4.07142857  4.17857143  4.28571429  4.39285714  4.5       ]", &filter);
    KALDI_LOG << filter;

    Vector<BaseFloat> para;
    int32 N1 = 4, N2 = 3, D = 5;
    para.Resize(((N1+1)+N2)*D, kSetZero);
    para.CopyRowsFromMat(filter);
    c->SetParams(para);

    // propagate,
    CuMatrix<BaseFloat> mat_out;
    c->Propagate(mat_in, &mat_out);
    KALDI_LOG << "mat_out" << mat_out;
    /* mat_out should be
       [[ 123.13571429  138.9         155.50714286  172.95714286  191.25      ]
       [ 182.27142857  200.94285714  220.65714286  241.41428571  263.21428571]
       [ 244.90714286  267.48571429  291.30714286  316.37142857  342.67857143]
       [ 313.54285714  341.02857143  369.95714286  400.32857143  432.14285714]
       [ 390.67857143  424.07142857  459.10714286  495.78571429  534.10714286]
       [ 309.28571429  338.21428571  368.57142857  400.35714286  433.57142857]
       [ 229.5         253.96428571  279.64285714  306.53571429  334.64285714]
       [ 154.          174.          195.          217.          240.        ]
       [ 591.42142857  624.04285714  657.50714286  691.81428571  726.96428571]
       [ 674.55714286  714.08571429  754.65714286  796.27142857  838.92857143]
       [ 512.47857143  548.66428571  585.87857143  624.12142857  663.39285714]
       [ 391.4         425.24285714  460.1         495.97142857  532.85714286]
       [ 316.5         349.          382.5         417.          452.5       ]
       [ 285.6         304.11428571  323.04285714  342.38571429  362.14285714]
       [ 117.7         133.3         149.3         165.7         182.5       ]]
     */

    CuMatrix<BaseFloat> mat_out_diff(mat_out);
    ReadCuMatrixFromString("[ 75.  74.  73.  72.  71. \n\
         70.  69.  68.  67.  66. \n\
         65.  64.  63.  62.  61. \n\
         60.  59.  58.  57.  56. \n\
         55.  54.  53.  52.  51. \n\
         50.  49.  48.  47.  46. \n\
         45.  44.  43.  42.  41. \n\
         40.  39.  38.  37.  36. \n\
         35.  34.  33.  32.  31. \n\
         30.  29.  28.  27.  26. \n\
         25.  24.  23.  22.  21. \n\
         20.  19.  18.  17.  16. \n\
         15.  14.  13.  12.  11. \n\
         10.   9.   8.   7.   6. \n\
          5.   4.   3.   2.   1. ]", &mat_out_diff);
    // backpropagate,
    CuMatrix<BaseFloat> mat_in_diff;
    c->Backpropagate(mat_in, mat_out, mat_out_diff, &mat_in_diff);
    KALDI_LOG << "mat_in_diff " << mat_in_diff;
    /* mat_in_diff should be
       [[  407.5          433.           457.5          481.           503.5       ]
       [  600.           627.92857143   654.64285714   680.14285714
       704.42857143]
       [  817.67857143   846.96428571   874.82142857   901.25         926.25      ]
       [ 1057.85714286  1087.42857143  1115.35714286  1141.64285714
       1166.28571429]
       [  898.82142857   922.98571429   945.70714286   966.98571429
       986.82142857]
       [  767.78571429   786.54285714   804.05714286   820.32857143
       835.35714286]
       [  662.25         675.6          687.90714286   699.17142857
       709.39285714]
       [  579.71428571   587.65714286   594.75714286   601.01428571
       606.42857143]
       [  147.5          153.           157.5          161.           163.5       ]
       [  199.           203.84285714   207.67142857   210.48571429
       212.28571429]
       [  269.75         272.86428571   274.95         276.00714286
       276.03571429]
       [  354.57142857   354.88571429   354.15714286   352.38571429
       349.57142857]
       [  287.03571429   284.44285714   281.00714286   276.72857143
       271.60714286]
       [   14.            13.6           12.8           11.6           10.        ]
       [   35.5           32.76428571    29.61428571    26.05          22.07142857]]
    */
    
    // update
    NnetTrainOptions opts;
    opts.learn_rate = -1.0;
    c->SetTrainOptions(opts);
    c->Update(mat_in, mat_out_diff);
    KALDI_LOG << c->GetBackfilter();
    /* output should be
       14600.1 14645.2 14660.3 14645.4 14600.5 
       10030.6 10126.7 10198.8 10246.9 10271 
       6526.1 6673.2 6802.3 6913.4 7006.5 
       4011.6 4147.7 4269.8 4377.9 4472 
       2107.1 2232.2 2347.3 2452.4 2547.5
    */
    KALDI_LOG << c->GetAheadfilter();
    /* output should be
       14593 14689.1 14761.2 14809.3 14833.4 
       13368.5 13515.6 13644.8 13755.9 13849 
       11994.1 12130.2 12252.3 12360.4 12454.5
    */
    delete c;
  }
}  // namespace nnet1
}  // namespace kaldi

int main() {
  using namespace kaldi;
  using namespace kaldi::nnet1;

  for (kaldi::int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      // use no GPU,
      CuDevice::Instantiate().SelectGpuId("no");
    else
      // use GPU when available,
      CuDevice::Instantiate().SelectGpuId("optional");
#endif
    // unit-tests :
    // UnitTestLengthNorm();
    // UnitTestSimpleSentenceAveragingComponent();
    // UnitTestConvolutionalComponentUnity();
    // UnitTestConvolutionalComponent3x3();
    // UnitTestMaxPoolingComponent();
    // UnitTestConvolutional2DComponent();
    // UnitTestMaxPooling2DComponent();
    // UnitTestAveragePooling2DComponent();
    // UnitTestDropoutComponent();
    // UnitTestSimpleRecurrentUnit();
    // UnitTestBatchNormComponent();
    // UnitTestEmbedding();
    // UnitTestTfLstm();
    UnitTestBiCompactVfsmn();
    // end of unit-tests,
    if (loop == 0)
        KALDI_LOG << "Tests without GPU use succeeded.";
      else
        KALDI_LOG << "Tests with GPU use (if available) succeeded.";
  }
#if HAVE_CUDA == 1
  CuDevice::Instantiate().PrintProfile();
#endif
  return 0;
}
