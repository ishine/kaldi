// nnet3/nnet-embedding-component.cc

// Copyright      2019  Jarvan Wang
//           2011-2013  Karel Vesely
//           2012-2017  Johns Hopkins University (author: Daniel Povey)
//                2013  Xiaohui Zhang
//           2014-2016  Vijayaditya Peddinti
//           2014-2015  Guoguo Chen
//                2015  Daniel Galvez
//                2015  Tom Ko

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
#include <algorithm>
#include <iomanip>
#include "nnet3/nnet-embedding-component.h"
#include "nnet3/nnet-parse.h"
#include "nnet3/nnet-compile-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet3 {

PositionEmbeddingComponent::PositionEmbeddingComponent(const PositionEmbeddingComponent &other):
    input_dim_(other.input_dim_), output_dim_(other.output_dim_)
    { }

void PositionEmbeddingComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = cfl->GetValue("input-dim", &input_dim_) &&
      cfl->GetValue("output-dim", &output_dim_);
  if (!ok)
    KALDI_ERR << "input-dim and output-dim must both be provided.";
  if (input_dim_ <= 0 || input_dim_ % output_dim_ != 0)
    KALDI_ERR << "Invalid values input-dim=" << input_dim_
              << " output-dim=" << output_dim_;
}

void PositionEmbeddingComponent::Read(std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary, "<PositionEmbeddingComponent>", "<InputDim>");
  ReadBasicType(is, binary, &input_dim_);
  ExpectToken(is, binary, "<OutputDim>");
  ReadBasicType(is, binary, &output_dim_);
  ExpectToken(is, binary, "</PositionEmbeddingComponent>");
}

void PositionEmbeddingComponent::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PositionEmbeddingComponent>");
  WriteToken(os, binary, "<InputDim>");
  WriteBasicType(os, binary, input_dim_);
  WriteToken(os, binary, "<OutputDim>");
  WriteBasicType(os, binary, output_dim_);
  WriteToken(os, binary, "</PositionEmbeddingComponent>");
}

std::string PositionEmbeddingComponent::Info() const {
  std::ostringstream stream;
  stream << Type() << ", input-dim=" << input_dim_
         << ", output-dim=" << output_dim_;
  return stream.str();
}

void* PositionEmbeddingComponent::Propagate(const ComponentPrecomputedIndexes *indexes_in,
                                   const CuMatrixBase<BaseFloat> &in,
                                   CuMatrixBase<BaseFloat> *out) const {
  const PrecomputedIndexes *indexes = dynamic_cast<const PrecomputedIndexes*>(
      indexes_in);
  KALDI_ASSERT(out->NumRows() == in.NumRows() &&
               out->NumCols() == in.NumCols() &&
               out->NumCols() == output_dim_ &&
               in.NumCols() == input_dim_ &&
               indexes != NULL &&
               in.NumRows() == indexes->io.num_t_in * indexes->io.num_images &&
               out->NumRows() == indexes->io.num_t_out * indexes->io.num_images);
  out->CopyFromMat(in);

  int32 batch_size = indexes->io.num_images,
	    sentence_len = indexes->io.num_t_in,
		d_model = in.NumCols();
     
  CuMatrix<BaseFloat> position_embedding(sentence_len, d_model, kUndefined);
  position_embedding.PosEmbStandard(10000.0);

  for (int32 i = 0; i < sentence_len; i++) {
	CuSubVector<BaseFloat> pos_vec(position_embedding, i);
	CuSubMatrix<BaseFloat> pos_mat(*out, i * batch_size, batch_size, 0, d_model);
	pos_mat.AddVecToRows(1.0, pos_vec, 1.0);
  }
/*
  Matrix<BaseFloat> position_embedding_cpu(sentence_len, d_model, kUndefined);
  Matrix<BaseFloat> position_embedding_gpu(position_embedding);

  for(int t=0;t<position_embedding_cpu.NumRows();t++){
	SubVector<BaseFloat> embedding(position_embedding_cpu, t);
    for(int i=0;i < d_model;i++){
      if(i%2 == 0){
        //odd
		embedding.Data()[i] = std::sin(t/std::pow(10000.0,1.0*i/d_model));
      }
      else{
        //even
		if(t==1 && i == 1)
		{
		   printf("t:%d, i:%d, d_model:%d, pow(10000.0,2.0*i/d_model):%f, t/std::pow(10000.0,2.0*i/d_model):%f, cos:%f\n", t, i, d_model, pow(10000.0,1.0*i/d_model), t/std::pow(10000.0,1.0*i/d_model), std::cos(t/std::pow(10000.0,1.0*i/d_model)));
		}
		embedding.Data()[i] = std::cos(t/std::pow(10000.0,1.0*i/d_model));
      }
    }
  }

  printf("=============GPU Matrix===============\n");
  for (int i = 0; i < position_embedding.NumRows(); i++)
  {
	for (int j = 0; j < position_embedding.NumCols(); j++) {
	  printf("%f  ", position_embedding_gpu(i, j));
	}
	printf("\n");
  }

  printf("=============CPU Matrix===============\n");
  for (int i = 0; i < sentence_len; i++)
  {
	for (int j = 0; j < d_model; j++) {
	  printf("%f  ", position_embedding_cpu(i, j));
	}
	printf("\n");
  }

/*
 // CPU version to generate position-embedding matrix;
  int d_model = in.NumCols();
  Vector<BaseFloat> embedding(d_model);
  CuVector<BaseFloat> embedding_(d_model);
  int stride = indexes->io.num_images;
  for(int t=0;t<in.NumRows();t+=stride){
    embedding.SetZero();
    embedding_.SetZero();
	int row = t/stride;
    for(int i=0;i < d_model;i++){
      if(i%2){
        //odd
        embedding.Data()[i] = std::cos(row/std::pow(10000.0,2.0*i/d_model));
      }
      else{
        //even
        embedding.Data()[i] = std::sin(row/std::pow(10000.0,2.0*i/d_model));
      }
    }
    for(int j=t;j<t+stride;j++){
	  embedding_.CopyFromVec(embedding);
      out->Row(j).AddVec(1.0, embedding_, 1.0);
	}
  }
*/
  return NULL;
}

void PositionEmbeddingComponent::Backprop(
    const std::string &debug_info,
    const ComponentPrecomputedIndexes *indexes,
    const CuMatrixBase<BaseFloat> &, //in_value
    const CuMatrixBase<BaseFloat> &, // out_value,
    const CuMatrixBase<BaseFloat> &out_deriv,
    void *memo,
    Component *to_update,
    CuMatrixBase<BaseFloat> *in_deriv) const {
  if (in_deriv) {
    in_deriv->CopyFromMat(out_deriv);
  }
}
void PositionEmbeddingComponent::ReorderIndexes(
    std::vector<Index> *input_indexes,
    std::vector<Index> *output_indexes) const {
  using namespace time_height_convolution;
  ConvolutionComputationIo io;
  GetComputationIo(*input_indexes, *output_indexes, &io);
  if (io.t_step_out == 0) io.t_step_out = 1;
  if (io.t_step_in == 0) io.t_step_in = 1;
  std::vector<Index> new_input_indexes, new_output_indexes;
  GetIndexes(*input_indexes, *output_indexes, io,
             &new_input_indexes, &new_output_indexes);
  input_indexes->swap(new_input_indexes);
  output_indexes->swap(new_output_indexes);
}
void PositionEmbeddingComponent::GetIndexes(
      const std::vector<Index> &input_indexes,
      const std::vector<Index> &output_indexes,
      time_height_convolution::ConvolutionComputationIo &io,
      std::vector<Index> *new_input_indexes,
      std::vector<Index> *new_output_indexes) const {

  std::unordered_set<Index, IndexHasher> input_set, output_set;
  for (std::vector<Index>::const_iterator iter = input_indexes.begin();
       iter != input_indexes.end(); ++iter)
    input_set.insert(*iter);
  for (std::vector<Index>::const_iterator iter = output_indexes.begin();
       iter != output_indexes.end(); ++iter)
    output_set.insert(*iter);

  std::vector<std::pair<int32, int32> > n_x_pairs;
  GetNxList(input_indexes, &n_x_pairs);  // the n,x pairs at the output will be
                                         // identical.
  KALDI_ASSERT(n_x_pairs.size() == io.num_images);
  CreateIndexesVector(n_x_pairs, io.start_t_in, io.t_step_in, io.num_t_in,
                      input_set, new_input_indexes);
  CreateIndexesVector(n_x_pairs, io.start_t_out, io.t_step_out, io.num_t_out,
                      output_set, new_output_indexes);
}
// static
void PositionEmbeddingComponent::CreateIndexesVector(
    const std::vector<std::pair<int32, int32> > &n_x_pairs,
    int32 t_start, int32 t_step, int32 num_t_values,
    const std::unordered_set<Index, IndexHasher> &index_set,
    std::vector<Index> *output_indexes) {
  output_indexes->resize(static_cast<size_t>(num_t_values) * n_x_pairs.size());
  std::vector<Index>::iterator out_iter = output_indexes->begin();
  for (int32 t = t_start; t < t_start + (t_step * num_t_values); t += t_step) {
    std::vector<std::pair<int32, int32> >::const_iterator
        iter = n_x_pairs.begin(), end = n_x_pairs.end();
    for (; iter != end; ++iter) {
      out_iter->n = iter->first;
      out_iter->t = t;
      out_iter->x = iter->second;
      if (index_set.count(*out_iter) == 0)
        out_iter->t = kNoTime;
      ++out_iter;
    }
  }
  KALDI_ASSERT(out_iter == output_indexes->end());
}
ComponentPrecomputedIndexes* PositionEmbeddingComponent::PrecomputeIndexes(
    const MiscComputationInfo &,  // misc_info
    const std::vector<Index> &input_indexes,
    const std::vector<Index> &output_indexes,
    bool) // need_backprop
    const {
  PrecomputedIndexes *ans = new PrecomputedIndexes();
  GetComputationIo(input_indexes, output_indexes, &(ans->io));
  if (ans->io.t_step_out == 0) ans->io.t_step_out = 1;
  if (ans->io.t_step_in == 0) ans->io.t_step_in = 1;
  if (GetVerboseLevel() >= 2) {
    // what goes next is just a check.
    std::vector<Index> new_input_indexes, new_output_indexes;
    GetIndexes(input_indexes, output_indexes, ans->io,
               &new_input_indexes, &new_output_indexes);
    // input_indexes and output_indexes should be the ones that were
    // output by ReorderIndexes(), so they should already
    // have gone through the GetComputationStructure()->GetIndexes()
    // procedure.  Applying the same procedure twice is supposed to
    // give an unchanged results.
    KALDI_ASSERT(input_indexes == new_input_indexes &&
                 output_indexes == new_output_indexes);
  }
  return ans;
}
PositionEmbeddingComponent::PrecomputedIndexes*
PositionEmbeddingComponent::PrecomputedIndexes::Copy() const {
  return new PrecomputedIndexes(*this);
}

void PositionEmbeddingComponent::PrecomputedIndexes::Write(
    std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<PositionEmbeddingComponentPrecomputedIndexes>");
  WriteToken(os, binary, "<Io>");
  io.Write(os, binary);
  WriteToken(os, binary, "</PositionEmbeddingComponentPrecomputedIndexes>");
}

void PositionEmbeddingComponent::PrecomputedIndexes::Read(
    std::istream &is, bool binary) {
  ExpectOneOrTwoTokens(is, binary,
                       "<PositionEmbeddingComponentPrecomputedIndexes>",
                       "<Io>");
  io.Read(is, binary);
  ExpectToken(is, binary, "</PositionEmbeddingComponentPrecomputedIndexes>");
}

} // namespace nnet3
} // namespace kaldi
