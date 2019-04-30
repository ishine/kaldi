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
  int32 left_context;
  int32 right_context;
//  int32 extra_left_context;
//  int32 extra_right_context;
  int32 extra_left_context_initial;
  int32 extra_right_context_final;
  int32 frame_subsampling_factor;
  int32 frames_per_chunk;
  int32 modulus;
  BaseFloat acoustic_scale;
  std::vector<std::string> input_nodes;
  std::vector<std::string> output_nodes;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;

  std::string model_file;
  std::string weight_file;

  MaceModelInfo():
      left_context(0),
      right_context(0),
      extra_left_context_initial(-1),
      extra_right_context_final(-1),
      frame_subsampling_factor(1),
      frames_per_chunk(50),
      modulus(1),
      acoustic_scale(0.1),
      input_nodes({"input"}),
      output_nodes({"output"}),
      input_shapes({}),
      output_shapes({}),
      model_file(""),
      weight_file("") {}
};


class MaceComputer {
 public:
  MaceComputer(MaceModelInfo info) :
    modulus_(info.modulus),
    left_context_(info.left_context),
    right_context_(info.right_context) {
    InitEngine(info.model_file,
               info.weight_file,
               info.input_nodes,
               info.output_nodes);
    InitTensors(info.input_nodes,
                info.input_shapes,
                info.output_nodes,
                info.output_shapes);

  }

  MaceComputer(const std::string &model_file,
               const std::string &weight_file,
               const std::vector<std::string> &input_nodes,
               const std::vector<std::string> &output_nodes,
               const std::vector<std::vector<int64_t>> &input_shapes,
               const std::vector<std::vector<int64_t>> &output_shapes):
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
                   const std::vector<std::vector<int64_t>> &input_shapes,
                   const std::vector<std::string> &output_names,
                   const std::vector<std::vector<int64_t>> &output_shapes);

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
