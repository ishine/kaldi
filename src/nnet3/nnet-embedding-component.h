// nnet3/nnet-embedding-component.h

// Copyright      2019  Jarvan Wang
//           2011-2013  Karel Vesely
//           2012-2017  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang
//           2014-2016  Vijayaditya Peddinti
//           2014-2015  Guoguo Chen
//                2015  Daniel Galvez
//                2015  Tom Ko

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

#ifndef KALDI_NNET3_NNET_EMBEDDING_COMPONENT_H_
#define KALDI_NNET3_NNET_EMBEDDING_COMPONENT_H_

#include "nnet3/nnet-common.h"
#include "nnet3/nnet-component-itf.h"
#include "nnet3/natural-gradient-online.h"
#include "nnet3/convolution.h"
#include <iostream>

namespace kaldi {
namespace nnet3 {

class PositionEmbeddingComponent: public Component {
 public:
  explicit PositionEmbeddingComponent(const PositionEmbeddingComponent &other);
  PositionEmbeddingComponent() { }
  virtual std::string Type() const { return "PositionEmbeddingComponent"; }
  virtual int32 Properties() const {
    return kPropagateAdds|kBackpropAdds;
  }
  virtual void InitFromConfig(ConfigLine *cfl);
  virtual int32 InputDim() const { return input_dim_; }
  virtual int32 OutputDim() const { return output_dim_; }
  virtual void Read(std::istream &is, bool binary);
  virtual void Write(std::ostream &os, bool binary) const;
  virtual std::string Info() const;
  virtual Component* Copy() const { return new PositionEmbeddingComponent(*this); }
  virtual void ReorderIndexes(std::vector<Index> *input_indexes,
                              std::vector<Index> *output_indexes) const;
  virtual ComponentPrecomputedIndexes* PrecomputeIndexes(
      const MiscComputationInfo &misc_info,
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      bool need_backprop) const;

  class PrecomputedIndexes: public ComponentPrecomputedIndexes {
   public:
    PrecomputedIndexes() { }
    PrecomputedIndexes(const PrecomputedIndexes &other):
        io(other.io) { }
    virtual PrecomputedIndexes *Copy() const;
    virtual void Write(std::ostream &os, bool binary) const;
    virtual void Read(std::istream &os, bool binary);
    virtual std::string Type() const {
      return "PositionEmbeddingComponentPrecomputedIndexes";
    }
    virtual ~PrecomputedIndexes() { }

    time_height_convolution::ConvolutionComputationIo io;
  };
  virtual void* Propagate(const ComponentPrecomputedIndexes *indexes,
                          const CuMatrixBase<BaseFloat> &in,
                          CuMatrixBase<BaseFloat> *out) const;
  virtual void Backprop(const std::string &debug_info,
                        const ComponentPrecomputedIndexes *indexes,
                        const CuMatrixBase<BaseFloat> &, //in_value
                        const CuMatrixBase<BaseFloat> &, // out_value,
                        const CuMatrixBase<BaseFloat> &out_deriv,
                        void *memo,
                        Component *to_update,
                        CuMatrixBase<BaseFloat> *in_deriv) const;
 private:
  void GetIndexes(
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      time_height_convolution::ConvolutionComputationIo &io,
      std::vector<Index> *new_input_indexes,
      std::vector<Index> *new_output_indexes) const;
  static void CreateIndexesVector(
      const std::vector<std::pair<int32, int32> > &n_x_pairs,
      int32 t_start, int32 t_step, int32 num_t_values,
      const std::unordered_set<Index, IndexHasher> &index_set,
      std::vector<Index> *output_indexes);
  int32 input_dim_;
  int32 output_dim_;
  PositionEmbeddingComponent &operator = (const PositionEmbeddingComponent &other); // Disallow.
};

} // namespace nnet3
} // namespace kaldi


#endif
