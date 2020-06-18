// nnet0/nnet-component.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
//           2018 Alibaba.Inc (Author: ShaoFei Xue, ShiLiang Zhang)

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

#include "nnet0/nnet-component.h"

#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-activation.h"
#include "nnet0/nnet-kl-hmm.h"
#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-time-delay-transform.h"
#include "nnet0/nnet-c-time-delay-transform.h"
#include "nnet0/nnet-batch-normalization-transform.h"
#include "nnet0/nnet-linear-transform.h"
#include "nnet0/nnet-rbm.h"
#include "nnet0/nnet-various.h"
#include "nnet0/nnet-kl-hmm.h"

#include "nnet0/nnet-convolutional-component.h"
#include "nnet0/nnet-average-pooling-component.h"
#include "nnet0/nnet-max-pooling-component.h"

#include "nnet0/nnet-convolutional-2d-component.h"
#include "nnet0/nnet-convolutional-2d-component-fast.h"
#include "nnet0/nnet-cudnn-convolutional-2d-component.h"
#include "nnet0/nnet-average-pooling-2d-component.h"
#include "nnet0/nnet-max-pooling-2d-component.h"
#include "nnet0/nnet-max-pooling-2d-component-fast.h"
#include "nnet0/nnet-cudnn-pooling-2d-component.h"

#include "nnet0/nnet-lstm-projected-streams.h"
#include "nnet0/nnet-lstm-streams.h"
#include "nnet0/nnet-lstm-standard.h"
#include "nnet0/nnet-gru-streams.h"
#include "nnet0/nnet-gru-projected-streams.h"
#include "nnet0/nnet-gru-projected-streams-fast.h"
#include "nnet0/nnet-lstm-projected-streams-fast.h"
#include "nnet0/nnet-lstm-projected-streams-fixedpoint.h"
#include "nnet0/nnet-lstm-projected-streams-simple.h"
#include "nnet0/nnet-lstm-projected-streams-residual.h"
#include "nnet0/nnet-blstm-projected-streams.h"
#include "nnet0/nnet-blstm-streams.h"

#include "nnet0/nnet-sentence-averaging-component.h"
#include "nnet0/nnet-frame-pooling-component.h"
#include "nnet0/nnet-parallel-component.h"
#include "nnet0/nnet-word-vector-transform.h"
#include "nnet0/nnet-class-affine-transform.h"
#include "nnet0/nnet-parallel-component-multitask.h"
#include "nnet0/nnet-multi-net-component.h"
#include "nnet0/nnet-rnnt-join-transform.h"
#include "nnet0/nnet-parametric-relu.h"

#include <sstream>
#include "nnet0/nnet-time-delay-transform.h"
#include "nnet0/nnet-statistics-pooling-component.h"

#include "nnet0/nnet-fsmn.h"
#include "nnet0/nnet-deep-fsmn.h"
#include "nnet0/nnet-uni-fsmn.h"
#include "nnet0/nnet-uni-deep-fsmn.h"
#include "nnet0/nnet-fsmn-streams.h"
#include "nnet0/nnet-deep-fsmn-streams.h"

namespace kaldi {
namespace nnet0 {

const struct Component::key_value Component::kMarkerMap[] = {
  { Component::kAffineTransform,"<AffineTransform>" },
  { Component::kTimeDelayTransform,"<TimeDelayTransform>" },
  { Component::kCompressedTimeDelayTransform,"<CompressedTimeDelayTransform>" },
  { Component::kStatisticsPoolingComponent,"<StatisticsPoolingComponent>" },
  { Component::kWordVectorTransform,"<WordVectorTransform>" },
  { Component::kClassAffineTransform,"<ClassAffineTransform>" },
  { Component::kCBSoftmax,"<CBSoftmax>" },
  { Component::kBatchNormalization,"<BatchNormalization>" },
  { Component::kAffinePreconditionedOnlineTransform,"<kAffinePreconditionedOnlineTransform>" },
  { Component::kLinearTransform,"<LinearTransform>" },
  { Component::kConvolutionalComponent,"<ConvolutionalComponent>"},
  { Component::kConvolutional2DComponent,"<Convolutional2DComponent>"},
  { Component::kConvolutional2DComponentFast,"<Convolutional2DComponentFast>"},
#if HAVE_CUDA == 1
  { Component::kCudnnPooling2DComponent, "<CudnnPooling2DComponent>"},
  { Component::kCudnnConvolutional2DComponent, "<CudnnConvolutional2DComponent>"},
  { Component::kCudnnRelu, "<CudnnRelu>"},
#endif
  { Component::kLstmProjectedStreams,"<LstmProjectedStreams>"},
  { Component::kLstmStreams,"<LstmStreams>"},
  { Component::kLstmStandard,"<LstmStandard>"},
  { Component::kLstmProjectedStreamsFast,"<LstmProjectedStreamsFast>"},
  { Component::kLstmProjectedStreamsFixedPoint,"<LstmProjectedStreamsFixedPoint>"},
  { Component::kLstmProjectedStreamsSimple,"<LstmProjectedStreamsSimple>"},
  { Component::kLstmProjectedStreamsResidual,"<LstmProjectedStreamsResidual>"},
  { Component::kBLstmProjectedStreams,"<BLstmProjectedStreams>"},
  { Component::kBLstmStreams,"<BLstmStreams>"},
  { Component::kGruStreams,"<GruStreams>"},
  { Component::kGruProjectedStreams, "<GruProjectedStreams>"},
  { Component::kGruProjectedStreamsFast, "<GruProjectedStreamsFast>"},
  { Component::kSoftmax,"<Softmax>" },
  { Component::kSoftmaxB,"<SoftmaxB>" },
  { Component::kSoftmaxT,"<SoftmaxT>" },
  { Component::kLogSoftmax,"<LogSoftmax>" },
  { Component::kBlockSoftmax,"<BlockSoftmax>" },
  { Component::kSigmoid,"<Sigmoid>" },
  { Component::kRelu,"<Relu>" },
  { Component::kTanh,"<Tanh>" },
  { Component::kParametricRelu,"<ParametricRelu>" },
  { Component::kDropout,"<Dropout>" },
  { Component::kLengthNormComponent,"<LengthNormComponent>" },
  { Component::kRbm,"<Rbm>" },
  { Component::kSplice,"<Splice>" },
  { Component::kSubSample,"<SubSample>" },
  { Component::kSpliceSample,"<SpliceSample>" },
  { Component::kCopy,"<Copy>" },
  { Component::kAddShift,"<AddShift>" },
  { Component::kRescale,"<Rescale>" },
  { Component::kKlHmm,"<KlHmm>" },
  { Component::kAveragePoolingComponent,"<AveragePoolingComponent>"},
  { Component::kAveragePooling2DComponent,"<AveragePooling2DComponent>"},
  { Component::kMaxPoolingComponent, "<MaxPoolingComponent>"},
  { Component::kMaxPooling2DComponent, "<MaxPooling2DComponent>"},
  { Component::kMaxPooling2DComponentFast, "<MaxPooling2DComponentFast>"},
  { Component::kSentenceAveragingComponent,"<SentenceAveragingComponent>"},
  { Component::kSimpleSentenceAveragingComponent,"<SimpleSentenceAveragingComponent>"},
  { Component::kFramePoolingComponent, "<FramePoolingComponent>"},
  { Component::kParallelComponent, "<ParallelComponent>"},
  { Component::kParallelComponentMultiTask, "<ParallelComponentMultiTask>"},
  { Component::kMultiNetComponent, "<MultiNetComponent>"},
  { Component::kRNNTJoinTransform, "<RNNTJoinTransform>"},
  { Component::kFsmn, "<Fsmn>" },
  { Component::kDeepFsmn, "<DeepFsmn>" },
  { Component::kUniFsmn, "<UniFsmn>" },
  { Component::kUniDeepFsmn, "<UniDeepFsmn>" },
  { Component::kFsmnStreams, "<FsmnStreams>" },
  { Component::kDeepFsmnStreams, "<DeepFsmnStreams>" },
};


const char* Component::TypeToMarker(ComponentType t) {
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    if (kMarkerMap[i].key == t) return kMarkerMap[i].value;
  }
  KALDI_ERR << "Unknown type" << t;
  return NULL;
}

Component::ComponentType Component::MarkerToType(const std::string &s) {
  std::string s_lowercase(s);
  std::transform(s.begin(), s.end(), s_lowercase.begin(), ::tolower); // lc
  int32 N=sizeof(kMarkerMap)/sizeof(kMarkerMap[0]);
  for(int i=0; i<N; i++) {
    std::string m(kMarkerMap[i].value);
    std::string m_lowercase(m);
    std::transform(m.begin(), m.end(), m_lowercase.begin(), ::tolower);
    if (s_lowercase == m_lowercase) return kMarkerMap[i].key;
  }
  KALDI_ERR << "Unknown marker : '" << s << "'";
  return kUnknown;
}


Component* Component::NewComponentOfType(ComponentType comp_type,
                      int32 input_dim, int32 output_dim) {
  Component *ans = NULL;
  switch (comp_type) {
    case Component::kAffineTransform :
      ans = new AffineTransform(input_dim, output_dim); 
      break;
    case Component::kTimeDelayTransform :
      ans = new TimeDelayTransform(input_dim, output_dim);
      break;
    case Component::kCompressedTimeDelayTransform :
      ans = new CompressedTimeDelayTransform(input_dim, output_dim);
      break;
    case Component::kStatisticsPoolingComponent :
      ans = new StatisticsPoolingComponent(input_dim, output_dim);
      break;
    case Component::kWordVectorTransform :
      ans = new WordVectorTransform(input_dim, output_dim);
      break;
    case Component::kClassAffineTransform :
      ans = new ClassAffineTransform(input_dim, output_dim);
      break;
    case Component::kCBSoftmax :
      ans = new CBSoftmax(input_dim, output_dim);
      break;
    case Component::kBatchNormalization :
      ans = new BatchNormalization(input_dim, output_dim);
      break;
    case Component::kLinearTransform :
      ans = new LinearTransform(input_dim, output_dim); 
      break;
    case Component::kConvolutionalComponent :
      ans = new ConvolutionalComponent(input_dim, output_dim);
      break;
    case Component::kConvolutional2DComponent :
      ans = new Convolutional2DComponent(input_dim, output_dim);
      break;
    case Component::kConvolutional2DComponentFast :
      ans = new Convolutional2DComponentFast(input_dim, output_dim);
      break;
#if HAVE_CUDA == 1
    case Component::kCudnnConvolutional2DComponent :
      ans = new CudnnConvolutional2DComponent(input_dim, output_dim);
      break;
    case Component::kCudnnPooling2DComponent:
      ans = new CudnnPooling2DComponent(input_dim, output_dim);
      break;
    case Component::kCudnnRelu :
      ans = new CudnnRelu(input_dim, output_dim);
      break;
#endif
    case Component::kLstmProjectedStreams :
      ans = new LstmProjectedStreams(input_dim, output_dim);
      break;
    case Component::kLstmStreams :
      ans = new LstmStreams(input_dim, output_dim);
      break;
    case Component::kLstmStandard :
      ans = new LstmStandard(input_dim, output_dim);
      break;
    case Component::kLstmProjectedStreamsFast :
      ans = new LstmProjectedStreamsFast(input_dim, output_dim);
      break;
    case Component::kLstmProjectedStreamsFixedPoint :
      ans = new LstmProjectedStreamsFixedPoint(input_dim, output_dim);
      break;
    case Component::kLstmProjectedStreamsSimple :
      ans = new LstmProjectedStreamsSimple(input_dim, output_dim);
      break;
    case Component::kLstmProjectedStreamsResidual :
      ans = new LstmProjectedStreamsResidual(input_dim, output_dim);
      break;
    case Component::kBLstmProjectedStreams :
      ans = new BLstmProjectedStreams(input_dim, output_dim);
      break;
    case Component::kBLstmStreams :
      ans = new BLstmStreams(input_dim, output_dim);
      break;
    case Component::kGruStreams :
      ans = new GruStreams(input_dim, output_dim);
      break;
    case Component::kGruProjectedStreams:
      ans = new GruProjectedStreams(input_dim, output_dim);
      break;
    case Component::kGruProjectedStreamsFast:
      ans = new GruProjectedStreamsFast(input_dim, output_dim);
      break;
    case Component::kSoftmax :
      ans = new Softmax(input_dim, output_dim);
      break;
    case Component::kSoftmaxB :
      ans = new SoftmaxB(input_dim, output_dim);
      break;
    case Component::kSoftmaxT :
      ans = new SoftmaxT(input_dim, output_dim);
      break;
    case Component::kLogSoftmax :
      ans = new LogSoftmax(input_dim, output_dim);
      break;
    case Component::kBlockSoftmax :
      ans = new BlockSoftmax(input_dim, output_dim);
      break;
    case Component::kSigmoid :
      ans = new Sigmoid(input_dim, output_dim);
      break;
    case Component::kRelu :
      ans = new Relu(input_dim, output_dim);
      break;
    case Component::kTanh :
      ans = new Tanh(input_dim, output_dim);
      break;
    case Component::kParametricRelu :
      ans = new ParametricRelu(input_dim, output_dim);
      break;
    case Component::kDropout :
      ans = new Dropout(input_dim, output_dim); 
      break;
    case Component::kLengthNormComponent :
      ans = new LengthNormComponent(input_dim, output_dim); 
      break;
    case Component::kRbm :
      ans = new Rbm(input_dim, output_dim);
      break;
    case Component::kSplice :
      ans = new Splice(input_dim, output_dim);
      break;
    case Component::kSubSample :
      ans = new SubSample(input_dim, output_dim);
      break;
    case Component::kSpliceSample :
      ans = new SpliceSample(input_dim, output_dim);
      break;
    case Component::kCopy :
      ans = new CopyComponent(input_dim, output_dim);
      break;
    case Component::kAddShift :
      ans = new AddShift(input_dim, output_dim);
      break;
    case Component::kRescale :
      ans = new Rescale(input_dim, output_dim);
      break;
    case Component::kKlHmm :
      ans = new KlHmm(input_dim, output_dim);
      break;
    case Component::kSentenceAveragingComponent :
      ans = new SentenceAveragingComponent(input_dim, output_dim);
      break;
    case Component::kSimpleSentenceAveragingComponent :
      ans = new SimpleSentenceAveragingComponent(input_dim, output_dim);
      break;
    case Component::kAveragePoolingComponent :
      ans = new AveragePoolingComponent(input_dim, output_dim);
      break;
    case Component::kAveragePooling2DComponent :
      ans = new AveragePooling2DComponent(input_dim, output_dim);
      break;
    case Component::kMaxPoolingComponent :
      ans = new MaxPoolingComponent(input_dim, output_dim);
      break;
    case Component::kMaxPooling2DComponent :
      ans = new MaxPooling2DComponent(input_dim, output_dim);
      break;
    case Component::kMaxPooling2DComponentFast :
      ans = new MaxPooling2DComponentFast(input_dim, output_dim);
      break;
    case Component::kFramePoolingComponent :
      ans = new FramePoolingComponent(input_dim, output_dim);
      break;
    case Component::kParallelComponent :
      ans = new ParallelComponent(input_dim, output_dim);
      break;
    case Component::kParallelComponentMultiTask :
      ans = new ParallelComponentMultiTask(input_dim, output_dim);
      break;
    case Component::kMultiNetComponent :
      ans = new MultiNetComponent(input_dim, output_dim);
      break;
    case Component::kRNNTJoinTransform :
      ans = new RNNTJoinTransform(input_dim, output_dim);
      break;
    case Component::kFsmn:
      ans = new Fsmn(input_dim, output_dim);
      break;
    case Component::kDeepFsmn:
      ans = new DeepFsmn(input_dim, output_dim);
      break;
    case Component::kUniFsmn:
      ans = new UniFsmn(input_dim, output_dim);
      break;
    case Component::kUniDeepFsmn:
      ans = new UniDeepFsmn(input_dim, output_dim);
      break;
    case Component::kFsmnStreams:
          ans = new FsmnStreams(input_dim, output_dim);
          break;
	case Component::kDeepFsmnStreams:
	  ans = new DeepFsmnStreams(input_dim, output_dim);
	  break;
    case Component::kUnknown :
    default :
      KALDI_ERR << "Missing type: " << TypeToMarker(comp_type);
  }
  return ans;
}


Component* Component::Init(const std::string &conf_line) {
  std::istringstream is(conf_line);
  std::string component_type_string;
  int32 input_dim, output_dim;

  // initialize component w/o internal data
  ReadToken(is, false, &component_type_string);
  ComponentType component_type = MarkerToType(component_type_string);
  ExpectToken(is, false, "<InputDim>");
  ReadBasicType(is, false, &input_dim); 
  ExpectToken(is, false, "<OutputDim>");
  ReadBasicType(is, false, &output_dim);
  Component *ans = NewComponentOfType(component_type, input_dim, output_dim);

  // initialize internal data with the remaining part of config line
  ans->InitData(is);

  return ans;
}


Component* Component::Read(std::istream &is, bool binary) {
  int32 dim_out, dim_in;
  std::string token;

  int first_char = Peek(is, binary);
  if (first_char == EOF) return NULL;

  ReadToken(is, binary, &token);
  // Skip optional initial token
  if(token == "<Nnet>") {
    ReadToken(is, binary, &token); // Next token is a Component
  }
  // Finish reading when optional terminal token appears
  if(token == "</Nnet>") {
    return NULL;
  }

  ReadBasicType(is, binary, &dim_out); 
  ReadBasicType(is, binary, &dim_in);

  Component *ans = NewComponentOfType(MarkerToType(token), dim_in, dim_out);
  ans->ReadData(is, binary);
  return ans;
}


void Component::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, Component::TypeToMarker(GetType()));
  WriteBasicType(os, binary, OutputDim());
  WriteBasicType(os, binary, InputDim());
  if(!binary) os << "\n";
  this->WriteData(os, binary);
}


} // namespace nnet0
} // namespace kaldi
