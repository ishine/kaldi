// nnet3/nnet-am-decodable-simple.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <numeric>

#include "mace-computer.h"

namespace kaldi {
namespace MACE {

MaceComputer::MaceComputer(MaceModelInfo info) :
    modulus_(info.modulus),
    left_context_(info.left_context),
    right_context_(info.right_context) {
  std::vector<std::string> input_names;
  input_names.push_back(info.input_name);
  int64_t input_frames = info.left_context + info.right_context + info.chunk_size;
  std::vector<std::vector<int64>> input_shapes;
  std::vector<int64> input_shape = {1, input_frames, info.input_dim};
  std::vector<int64> ivector_shape = {1, 1, info.ivector_dim};
  std::vector<int64> output_shape = {1, info.chunk_size, info.output_dim};
  input_shapes.push_back(input_shape);
  if (info.ivector_dim > 0) {
    input_names.push_back(info.ivector_name);
    input_shapes.push_back(ivector_shape);
  }
  std::vector<std::string> output_names;
  output_names.push_back(info.output_name);
  KALDI_LOG << "Start init engine.";
  InitEngine(info.model_file,
             info.weight_file,
             input_names,
             output_names);
  KALDI_LOG<< "start init tensors.";
  InitTensors(input_names,
              input_shapes,
              output_names,
              {output_shape});

}


mace::MaceStatus MaceComputer::InitEngine(
    const std::string &model_file,
    const std::string &weight_file,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes) {
  int32 omp_num_threads = -1;
  int32 cpu_affinity_policy = 1;
  mace::MaceEngineConfig config(mace::DeviceType::CPU);
  mace::MaceStatus status = config.SetCPUThreadPolicy(
      omp_num_threads,
      static_cast<mace::CPUAffinityPolicy>(cpu_affinity_policy));
  if (status != mace::MaceStatus::MACE_SUCCESS) {
    KALDI_ERR << "Set openmp or cpu affinity failed.";
  }

  KALDI_ASSERT(!model_file.empty());
  KALDI_ASSERT(!weight_file.empty());


//  unsigned char *model_graph_data;
//  size_t graph_data_len = 0;
//
//  int fd = open(model_file.c_str(), O_RDONLY);
//  if (fd >= 0) {
//    struct stat st;
//    int r = fstat(fd, &st);
//    if (r < 0) {
//      KALDI_ERR << "Failed to stat file: " << model_file;
//    }
//
//    KALDI_LOG << "graph data len:" << st.st_size;
//
//    graph_data_len = static_cast<size_t>(st.st_size);
//    model_graph_data = reinterpret_cast<unsigned char *>(malloc(graph_data_len));
//
//    ssize_t len = read(fd, model_graph_data, 100);
//    KALDI_LOG << "graph data read len:" << len;
//    close(fd);
//    if (len < 0) {
//      KALDI_ERR << "Failed to read file: " << model_file;
//    }
//  } else {
//    KALDI_ERR << "Failed to open file: " << model_file;
//  }
//
//  fd = open(weight_file.c_str(), O_RDONLY);
//  size_t weights_data_len = 0;
//  if (fd >= 0) {
//    struct stat st;
//    fstat(fd, &st);
//    weights_data_len = static_cast<size_t>(st.st_size);
//    model_weights_data_ = reinterpret_cast<unsigned char *>(malloc(graph_data_len));
//
//    ssize_t len = read(fd, model_weights_data_, weights_data_len);
//    close(fd);
//    if (len < 0) {
//      KALDI_ERR << "Failed to read file: " << weight_file;
//    }
//
//  } else {
//    KALDI_ERR << "Failed to open file: " << weight_file;
//  }



  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_graph_data =
      mace::make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  if (!model_file.empty()) {
    auto fs = mace::GetFileSystem();
    auto status_t = fs->NewReadOnlyMemoryRegionFromFile(model_file.c_str(),
                                                 &model_graph_data);
    if (status_t != mace::MaceStatus::MACE_SUCCESS) {
      KALDI_ERR << "Failed to read file: " << model_file;
    }
  }

  model_weights_data_ =
      mace::make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  KALDI_ASSERT(!weight_file.empty());
  if (!weight_file.empty()) {
    auto fs = mace::GetFileSystem();
    auto status_t = fs->NewReadOnlyMemoryRegionFromFile(
        weight_file.c_str(),
        &model_weights_data_);
    if (status_t != mace::MaceStatus::MACE_SUCCESS) {
      KALDI_ERR << "Failed to read file: " << weight_file;
    }
  }
  KALDI_ASSERT(model_weights_data_->length() > 0);

  // Only choose one of the two type based on the `model_graph_format`
  // in model deployment file(.yml).

  KALDI_LOG << "Start init engine.";

  mace::MaceStatus create_engine_status;
  create_engine_status = mace::CreateMaceEngineFromProto(
      reinterpret_cast<const unsigned char *>(model_graph_data->data()),
      model_graph_data->length(),
      reinterpret_cast<const unsigned char *>(model_weights_data_->data()),
      model_weights_data_->length(),
      input_nodes,
      output_nodes,
      config,
      &engine_);


  if (create_engine_status != mace::MaceStatus::MACE_SUCCESS) {
    KALDI_ERR << "Create engine error, please check the arguments first, "
              << "if correct, the device may not run the model, "
              << "please fall back to other strategy.";
    exit(1);
  }

  return create_engine_status;
}

void MaceComputer::InitTensors(
    const std::vector<std::string> &input_names,
    const std::vector<std::vector<int64>> &input_shapes,
    const std::vector<std::string> &output_names,
    const std::vector<std::vector<int64>> &output_shapes) {
  const size_t input_count = input_names.size();
  const size_t output_count = output_names.size();

  std::map<std::string, int64> inputs_size;
  for (size_t i = 0; i < input_count; ++i) {
    int64 input_size =
        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 1,
                        std::multiplies<int64>());
    inputs_size[input_names[i]] = input_size;
    auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                            std::default_delete<float[]>());
    inputs_[input_names[i]] = mace::MaceTensor(input_shapes[i],
                                               buffer_in,
                                               mace::DataFormat::NONE);
  }

  size_t dim_size = input_shapes[0].size();
  input_dim_ = input_shapes[0][dim_size - 1];
  if (input_count > 1) {
    dim_size = input_shapes[1].size();
    ivector_dim_ = input_shapes[1][dim_size - 1];
  } else {
    ivector_dim_ = -1;
  }
  KALDI_ASSERT(output_count == 1);
  for (size_t i = 0; i < output_count; ++i) {
    int64 output_size =
        std::accumulate(output_shapes[i].begin(),
                        output_shapes[i].end(), 1,
                        std::multiplies<int64>());
    auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                             std::default_delete<float[]>());
    outputs_[output_names[i]] = mace::MaceTensor(output_shapes[i],
                                                 buffer_out,
                                                 mace::DataFormat::NONE);
  }

  dim_size = output_shapes[0].size();
  output_dim_ = output_shapes[0][dim_size - 1];

}

void MaceComputer::AcceptInput(const std::string &input_name,
                               CuMatrix<BaseFloat> *input_feats_cu) {
  auto iter = inputs_.find(input_name);
  KALDI_ASSERT(iter != inputs_.end());
  mace::MaceTensor tensor = inputs_[input_name];

  int64 input_size = std::accumulate(tensor.shape().begin(), tensor.shape().end(), 1,
                                     std::multiplies<int64_t>());
  std::copy_n(input_feats_cu->Data(), input_size, tensor.data().get());
}

void MaceComputer::GetOutputDestructive(const std::string &name,
                                        CuMatrix<BaseFloat> *output_mat) {
  auto iter = outputs_.find(name);
  KALDI_ASSERT(iter != outputs_.end());
  mace::MaceTensor tensor = outputs_[name];
  const std::vector<int64> &output_shape = tensor.shape();
  int64 rows = std::accumulate(output_shape.begin(),
                               output_shape.end() - 1, 1,
                               std::multiplies<int64>());
  output_mat->Resize(rows, output_dim_);
  int64 output_size = rows * output_dim_;
  std::copy_n(tensor.data().get(), output_size, output_mat->Data());
}


} // namespace MACE
} // namespace kaldi
