// nnet3/nnet-am-decodable-simple.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_MACE_MACE_COMPUTER_H_
#define KALDI_MACE_MACE_COMPUTER_H_

#include <vector>

#include "cudamatrix/cu-matrix-lib.h"
#include "itf/options-itf.h"
#include "matrix/matrix-lib.h"
#include "mace/public/mace.h"
#include "mace/port/env.h"
#include "mace/port/port.h"
#include "mace/utils/memory.h"
#include "mace/port/file_system.h"

namespace kaldi {
namespace MACE {

class MaceModelInfo {
 public:
  std::string input_name;
  std::string ivector_name;
  std::string output_name;
  int32 input_dim;
  int32 ivector_dim;
  int32 output_dim;
  int32 left_context;
  int32 right_context;
  int32 chunk_size;
  int32 modulus;
  int32 batch;

  std::string model_file;
  std::string weight_file;

  MaceModelInfo():
      input_name("input"),
      ivector_name("ivector"),
      output_name("output"),
      input_dim(0),
      ivector_dim(0),
      output_dim(0),
      left_context(0),
      right_context(0),
      chunk_size(20),
      modulus(1),
      batch(1),
      model_file(""),
      weight_file("") {}

  void Register(OptionsItf *opts) {
    opts->Register("input-name", &input_name,
                   "the input node name of the nnet, default is 'input'.");
    opts->Register("ivector-name", &ivector_name,
                   "the ivector node name of nnet, default is 'ivector'.");
    opts->Register("output-name", &output_name,
                   "the output node name of nnet, default is 'output'.");
    opts->Register("input-dim", &input_dim, "input feature's dim.");
    opts->Register("ivector-dim", &ivector_dim, "ivector's dim.");
    opts->Register("output-dim", &output_dim, "Nnet's output dim.");
    opts->Register("chunk-size", &chunk_size,
                   "Number of output frames for one computation.");
    opts->Register("modulus", &modulus,
                   "Modulus of the nnet model.");
    opts->Register("batch", &batch,
                   "Batch size for one computation.");
    opts->Register("model-file", &model_file, "Model graph file path.");
    opts->Register("weight-file", &weight_file, "Model data file path.");
  }
};


class MaceComputer {
 public:
  MaceComputer(MaceModelInfo info);

  MaceComputer(const std::string &model_file,
               const std::string &weight_file,
               const std::vector<std::string> &input_nodes,
               const std::vector<std::string> &output_nodes,
               const std::vector<std::vector<int64>> &input_shapes,
               const std::vector<std::vector<int64>> &output_shapes):
      modulus_(1),
      left_context_(0),
      right_context_(0) {
    InitEngine(model_file,
               weight_file,
               input_nodes,
               output_nodes);
    InitTensors(input_nodes,
                input_shapes,
                output_nodes,
                output_shapes);
  };

  int32 OutputDim() const { return output_dim_; }
  int32 InputDim() const { return input_dim_; }
  int32 IvectorDim() const { return ivector_dim_; }
  int32 Modulus() const { return modulus_; }
  int32 LeftContext() const { return left_context_; }
  int32 RightContext() const { return right_context_; }

  mace::MaceStatus InitEngine(const std::string &model_file,
                              const std::string &weight_file,
                              const std::vector<std::string> &input_nodes,
                              const std::vector<std::string> &output_nodes);
  void InitTensors(const std::vector<std::string> &input_names,
                   const std::vector<std::vector<int64>> &input_shapes,
                   const std::vector<std::string> &output_names,
                   const std::vector<std::vector<int64>> &output_shapes);

  void AcceptInput(const std::string &name,
                   CuMatrix<BaseFloat> *input_mat);
  void GetOutputDestructive(const std::string &name,
                            CuMatrix<BaseFloat> *output_mat);

  mace::MaceStatus Run() { return engine_->Run(inputs_, &outputs_); };

  mace::MaceStatus Run(mace::RunMetadata *run_metadata) {
    return engine_->Run(inputs_, &outputs_, run_metadata);
  };

 private:
  int32 modulus_;
  std::shared_ptr<mace::MaceEngine> engine_;
  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_weights_data_;
  int32 input_dim_;
  int32 ivector_dim_;
  int32 output_dim_;
  int32 left_context_;
  int32 right_context_;
  std::map<std::string, mace::MaceTensor> inputs_;
  std::map<std::string, mace::MaceTensor> outputs_;
};

} // namespace MACE
} // namespace kaldi

#endif  // KALDI_MACE_MACE_COMPUTER_H_
