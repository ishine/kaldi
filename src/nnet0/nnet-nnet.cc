// nnet0/nnet-nnet.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

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

#include "nnet0/nnet-nnet.h"
#include "nnet0/nnet-component.h"
#include "nnet0/nnet-parallel-component.h"
#include "nnet0/nnet-activation.h"
#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-various.h"
#include "nnet0/nnet-lstm-projected-streams.h"
#include "nnet0/nnet-blstm-projected-streams.h"
#include "nnet0/nnet-blstm-streams.h"

#include "nnet0/nnet-affine-preconditioned-transform.h"
#include "nnet0/nnet-lstm-projected-streams-fast.h"
#include "nnet0/nnet-lstm-projected-streams-fixedpoint.h"
#include "nnet0/nnet-lstm-projected-streams-simple.h"
#include "nnet0/nnet-lstm-streams.h"
#include "nnet0/nnet-gru-streams.h"
#include "nnet0/nnet-gru-projected-streams.h"
#include "nnet0/nnet-gru-projected-streams-fast.h"
#include "nnet0/nnet-class-affine-transform.h"
#include "nnet0/nnet-parallel-component-multitask.h"

namespace kaldi {
namespace nnet0 {


Nnet::Nnet(const Nnet& other) {
  // copy the components
  for(int32 i = 0; i < other.NumComponents(); i++) {
    components_.push_back(other.GetComponent(i).Copy());
  }
  // create empty buffers
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()+1);
  // copy train opts
  SetTrainOptions(other.opts_);
  Check();
}

Nnet & Nnet::operator = (const Nnet& other) {
  Destroy();
  // copy the components
  for(int32 i = 0; i < other.NumComponents(); i++) {
    components_.push_back(other.GetComponent(i).Copy());
  }
  // create empty buffers
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()+1);
  // copy train opts
  SetTrainOptions(other.opts_); 
  Check();
  return *this;
}


Nnet::~Nnet() {
  Destroy();
}


void Nnet::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (NumComponents() == 0) {
    // (*out) = in; // copy
	if (out->NumRows() != in.NumRows() || out->NumCols() != in.NumCols())
		(static_cast<CuMatrix<BaseFloat>*>(out))->Resize(in.NumRows(), in.NumCols(), kUndefined);
	out->CopyFromMat(in);
    return; 
  }

  // we need at least L+1 input buffers
  KALDI_ASSERT((int32)propagate_buf_.size() >= NumComponents()+1);
  
  propagate_buf_[0].Resize(in.NumRows(), in.NumCols(), kUndefined, kStrideEqualNumCols);
  propagate_buf_[0].CopyFromMat(in);

  for(int32 i=0; i<(int32)components_.size(); i++) {
    components_[i]->Propagate(propagate_buf_[i], &propagate_buf_[i+1]);
  }
  
  // (*out) = propagate_buf_[components_.size()];
  CuMatrix<BaseFloat> &mat = propagate_buf_[components_.size()];
  if (out->NumRows() != mat.NumRows() || out->NumCols() != mat.NumCols())
  		(static_cast<CuMatrix<BaseFloat>*>(out))->Resize(mat.NumRows(), mat.NumCols(), kUndefined);
  out->CopyFromMat(mat);
}

/*
void Nnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff) {

  //////////////////////////////////////
  // Backpropagation
  //

  // 0 layers
  if (NumComponents() == 0) { (*in_diff) = out_diff; return; }

  KALDI_ASSERT((int32)propagate_buf_.size() == NumComponents()+1);
  KALDI_ASSERT((int32)backpropagate_buf_.size() == NumComponents()+1);

  // copy out_diff to last buffer
  backpropagate_buf_[NumComponents()] = out_diff;
  // backpropagate using buffers
  for (int32 i = NumComponents()-1; i >= 0; i--) {
    components_[i]->Backpropagate(propagate_buf_[i], propagate_buf_[i+1],
                            backpropagate_buf_[i+1], &backpropagate_buf_[i]);
    if (components_[i]->IsUpdatable()) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
      uc->Update(propagate_buf_[i], backpropagate_buf_[i+1]);
    }
  }
  // eventually export the derivative
  if (NULL != in_diff) (*in_diff) = backpropagate_buf_[0];

  //
  // End of Backpropagation
  //////////////////////////////////////
}
*/
void Nnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff, bool update) {

  //////////////////////////////////////
  // Backpropagation
  //

  // 0 layers
  if (NumComponents() == 0) {
	// (*in_diff) = out_diff;
	if (in_diff->NumRows() != out_diff.NumRows() || in_diff->NumCols() != out_diff.NumCols())
			(static_cast<CuMatrix<BaseFloat>*>(in_diff))->Resize(out_diff.NumRows(), out_diff.NumCols(), kUndefined);
	in_diff->CopyFromMat(out_diff);
	return;
  }

  KALDI_ASSERT((int32)propagate_buf_.size() == NumComponents()+1);
  KALDI_ASSERT((int32)backpropagate_buf_.size() == NumComponents()+1);

  // copy out_diff to last buffer
  //backpropagate_buf_[NumComponents()] = out_diff;
  backpropagate_buf_[NumComponents()].Resize(out_diff.NumRows(), out_diff.NumCols(), kUndefined, kStrideEqualNumCols);
  backpropagate_buf_[NumComponents()].CopyFromMat(out_diff);
  // backpropagate using buffers
  for (int32 i = NumComponents()-1; i >= 0; i--) {
    components_[i]->Backpropagate(propagate_buf_[i], propagate_buf_[i+1],
                            backpropagate_buf_[i+1], &backpropagate_buf_[i]);

    if (components_[i]->IsUpdatable() && update) {
        UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
        uc->Gradient(propagate_buf_[i], backpropagate_buf_[i+1]);
        uc->UpdateGradient();
        //uc->Update(propagate_buf_[i], backpropagate_buf_[i+1]);
    }
  }
  // eventually export the derivative
  // if (NULL != in_diff) (*in_diff) = backpropagate_buf_[0];

  if (NULL != in_diff) {
	  CuMatrix<BaseFloat> &mat = backpropagate_buf_[0];
	  if (in_diff->NumRows() != mat.NumRows() || in_diff->NumCols() != mat.NumCols())
	  		(static_cast<CuMatrix<BaseFloat>*>(in_diff))->Resize(mat.NumRows(), mat.NumCols(), kUndefined);
	  in_diff->CopyFromMat(mat);
  }

  //
  // End of Backpropagation
  //////////////////////////////////////
}

void Nnet::ResetGradient()
{

    for (int32 i = NumComponents()-1; i >= 0; i--) {
            if (components_[i]->IsUpdatable()) {
              UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
              uc->ResetGradient();
            }
          }
}

void Nnet::Gradient()
{

	for (int32 i = NumComponents()-1; i >= 0; i--) {
		    if (components_[i]->IsUpdatable()) {
		      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
		      uc->Gradient(propagate_buf_[i], backpropagate_buf_[i+1]);
		    }
		  }
}

void Nnet::UpdateGradient()
{
	for (int32 i = NumComponents()-1; i >= 0; i--) {
		    if (components_[i]->IsUpdatable()) {
		      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
		      uc->UpdateGradient();
		    }
		  }
}

/// Perform update gradient pass through the network
void Nnet::Update()
{

	  //////////////////////////////////////
	  // Update gradient
	  //

	  // update using buffers
	  for (int32 i = NumComponents()-1; i >= 0; i--) {
	    if (components_[i]->IsUpdatable()) {
	      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
	      uc->Update(propagate_buf_[i], backpropagate_buf_[i+1]);
	    }
	  }

	  //
	  // End of Update
	  //////////////////////////////////////
}

int Nnet::WeightCopy(void *buffer, int direction, int copykind)
{
	int pos = 0;
	for (int32 i = 0; i < components_.size(); i++) {
		if (components_[i]->IsUpdatable()) {
	        UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
			pos += uc->WeightCopy((void*)((char *)buffer+pos) , direction, copykind);
		}
	}
	return pos;
}

int Nnet::GetDim() const
{
	int pos = 0;
	for (int32 i = 0; i < components_.size(); i++) {
		if (components_[i]->IsUpdatable()) {
	        UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(components_[i]);
			pos += uc->GetDim();
		}
	}
	return pos;
}

void Nnet::Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
  KALDI_ASSERT(NULL != out);

  if (NumComponents() == 0) { 
    out->Resize(in.NumRows(), in.NumCols());
    out->CopyFromMat(in); 
    return; 
  }

  // we need at least 2 input buffers
  KALDI_ASSERT(propagate_buf_.size() >= 2);

  // propagate by using exactly 2 auxiliary buffers
  propagate_buf_[0].Resize(in.NumRows(), in.NumCols(), kUndefined, kStrideEqualNumCols);
  propagate_buf_[0].CopyFromMat(in);

  int32 L = 0;
  for(L=0; L<NumComponents()-1; L++) {
    components_[L]->Propagate(propagate_buf_[L%2], &propagate_buf_[(L+1)%2]);
  }
  components_[L]->Propagate(propagate_buf_[L%2], out);
  // release the buffers we don't need anymore
  propagate_buf_[0].Resize(0,0);
  propagate_buf_[1].Resize(0,0);
}


int32 Nnet::OutputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.back()->OutputDim();
}

int32 Nnet::InputDim() const {
  KALDI_ASSERT(!components_.empty());
  return components_.front()->InputDim();
}

const Component& Nnet::GetComponent(int32 component) const {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

Component& Nnet::GetComponent(int32 component) {
  KALDI_ASSERT(static_cast<size_t>(component) < components_.size());
  return *(components_[component]);
}

void Nnet::SetComponent(int32 c, Component *component) {
  KALDI_ASSERT(static_cast<size_t>(c) < components_.size());
  delete components_[c];
  components_[c] = component;
  Check(); // Check that all the dimensions still match up.
}

void Nnet::AppendComponent(Component* dynamically_allocated_comp) {
  // append,
  components_.push_back(dynamically_allocated_comp);
  // create training buffers,
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()+1);
  //
  Check();
}

void Nnet::AppendNnet(const Nnet& nnet_to_append) {
  // append,
  for(int32 i=0; i<nnet_to_append.NumComponents(); i++) {
    AppendComponent(nnet_to_append.GetComponent(i).Copy());
  }
  // create training buffers,
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()+1);
  //
  Check();
}

void Nnet::RemoveComponent(int32 component) {
  KALDI_ASSERT(component < NumComponents());
  // remove,
  Component* ptr = components_[component];
  components_.erase(components_.begin()+component);
  delete ptr;
  // create training buffers,
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()+1);
  // 
  Check();
}


void Nnet::GetParams(Vector<BaseFloat>* wei_copy) const {
  wei_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 i=0; i<components_.size(); i++) {
    if(components_[i]->IsUpdatable()) {
      UpdatableComponent& c = dynamic_cast<UpdatableComponent&>(*components_[i]);
      Vector<BaseFloat> c_params; 
      c.GetParams(&c_params);
      wei_copy->Range(pos,c_params.Dim()).CopyFromVec(c_params);
      pos += c_params.Dim();
    }
  }
  KALDI_ASSERT(pos == NumParams());
}

void Nnet::GetWeights(Vector<BaseFloat>* wei_copy) const {
  wei_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      switch(components_[n]->GetType()) {
        case Component::kAffineTransform : {
          // copy weight matrix row-by-row to the vector
          Matrix<BaseFloat> mat(dynamic_cast<AffineTransform*>(components_[n])->GetLinearity());
          int32 mat_size = mat.NumRows()*mat.NumCols();
          wei_copy->Range(pos,mat_size).CopyRowsFromMat(mat);
          pos += mat_size;
          // append biases
          Vector<BaseFloat> vec(dynamic_cast<AffineTransform*>(components_[n])->GetBias());
          wei_copy->Range(pos,vec.Dim()).CopyFromVec(vec);
          pos += vec.Dim();
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(components_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


void Nnet::SetWeights(const Vector<BaseFloat>& wei_src) {
  KALDI_ASSERT(wei_src.Dim() == NumParams());
  int32 pos = 0;
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      switch(components_[n]->GetType()) {
        case Component::kAffineTransform : {
          // get the component
          AffineTransform* aff_t = dynamic_cast<AffineTransform*>(components_[n]);
          // we need weight matrix with original dimensions
          Matrix<BaseFloat> mat(aff_t->GetLinearity());
          int32 mat_size = mat.NumRows()*mat.NumCols();
          mat.CopyRowsFromVec(wei_src.Range(pos,mat_size));
          pos += mat_size;
          // get the bias vector
          Vector<BaseFloat> vec(aff_t->GetBias());
          vec.CopyFromVec(wei_src.Range(pos,vec.Dim()));
          pos += vec.Dim();
          // assign to the component
          aff_t->SetLinearity(CuMatrix<BaseFloat>(mat));
          aff_t->SetBias(CuVector<BaseFloat>(vec));
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(components_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}

 
void Nnet::GetGradient(Vector<BaseFloat>* grad_copy) const {
  grad_copy->Resize(NumParams());
  int32 pos = 0;
  // copy the params
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      switch(components_[n]->GetType()) {
        case Component::kAffineTransform : {
          // get the weights from CuMatrix to Matrix
          const CuMatrixBase<BaseFloat>& cu_mat = 
            dynamic_cast<AffineTransform*>(components_[n])->GetLinearityCorr();
          Matrix<BaseFloat> mat(cu_mat.NumRows(),cu_mat.NumCols());
          cu_mat.CopyToMat(&mat);
          // copy the the matrix row-by-row to the vector
          int32 mat_size = mat.NumRows()*mat.NumCols();
          grad_copy->Range(pos,mat_size).CopyRowsFromMat(mat);
          pos += mat_size;
          // get the biases from CuVector to Vector
          const CuVector<BaseFloat>& cu_vec = 
            dynamic_cast<AffineTransform*>(components_[n])->GetBiasCorr();
          Vector<BaseFloat> vec(cu_vec.Dim());
          cu_vec.CopyToVec(&vec);
          // append biases to the supervector
          grad_copy->Range(pos,vec.Dim()).CopyFromVec(vec);
          pos += vec.Dim();
        } break;
        default :
          KALDI_ERR << "Unimplemented access to parameters "
                    << "of updatable component " 
                    << Component::TypeToMarker(components_[n]->GetType());
      }
    }
  }
  KALDI_ASSERT(pos == NumParams());
}


int32 Nnet::NumParams() const {
  int32 n_params = 0;
  for(int32 n=0; n<components_.size(); n++) {
    if(components_[n]->IsUpdatable()) {
      n_params += dynamic_cast<UpdatableComponent*>(components_[n])->NumParams();
    }
  }
  return n_params;
}


void Nnet::SetDropoutRetention(BaseFloat r)  {
  for (int32 c=0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kDropout) {
      Dropout& comp = dynamic_cast<Dropout&>(GetComponent(c));
      BaseFloat r_old = comp.GetDropoutRetention();
      comp.SetDropoutRetention(r);
      KALDI_LOG << "Setting dropout-retention in component " << c 
                << " from " << r_old << " to " << r;
    }
  }
}


void Nnet::ResetLstmStreams(const std::vector<int32> &stream_reset_flag, int32 ntruncated_bptt_size) {
  for (int32 c=0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kLstmProjectedStreams) {
      LstmProjectedStreams& comp = dynamic_cast<LstmProjectedStreams&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag, ntruncated_bptt_size);
    }    
    else if (GetComponent(c).GetType() == Component::kLstmProjectedStreamsFast) {
      LstmProjectedStreamsFast& comp = dynamic_cast<LstmProjectedStreamsFast&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag, ntruncated_bptt_size);
    }
    else if (GetComponent(c).GetType() == Component::kLstmProjectedStreamsFixedPoint) {
      LstmProjectedStreamsFixedPoint& comp = dynamic_cast<LstmProjectedStreamsFixedPoint&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag, ntruncated_bptt_size);
    }
    else if (GetComponent(c).GetType() == Component::kLstmProjectedStreamsSimple) {
      LstmProjectedStreamsSimple& comp = dynamic_cast<LstmProjectedStreamsSimple&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag, ntruncated_bptt_size);
    }
    else if (GetComponent(c).GetType() == Component::kLstmStreams) {
      LstmStreams& comp = dynamic_cast<LstmStreams&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag, ntruncated_bptt_size);
    }
    else if (GetComponent(c).GetType() == Component::kGruStreams) {
    	GruStreams& comp = dynamic_cast<GruStreams&>(GetComponent(c));
      comp.ResetGRUStreams(stream_reset_flag, ntruncated_bptt_size);
    }else if(GetComponent(c).GetType() == Component::kGruProjectedStreams){
        GruProjectedStreams& comp = dynamic_cast<GruProjectedStreams&>(GetComponent(c));
        comp.ResetGRUProjectedStreams(stream_reset_flag, ntruncated_bptt_size);
    }else if (GetComponent(c).GetType() == Component::kGruProjectedStreamsFast) {
        GruProjectedStreamsFast& comp = dynamic_cast<GruProjectedStreamsFast&>(GetComponent(c));
        comp.ResetGRUProjectedStreamsFast(stream_reset_flag, ntruncated_bptt_size);
    }   
    else if (GetComponent(c).GetType() == Component::kParallelComponentMultiTask) {
      ParallelComponentMultiTask& comp = dynamic_cast<ParallelComponentMultiTask&>(GetComponent(c));
      comp.ResetLstmStreams(stream_reset_flag, ntruncated_bptt_size);
    }
  }
}

void Nnet::UpdateLstmStreamsState(const std::vector<int32> &stream_update_flag) {
  for (int32 c=0; c < NumComponents(); c++) {
	if (GetComponent(c).GetType() == Component::kLstmProjectedStreamsFast) {
      LstmProjectedStreamsFast& comp = dynamic_cast<LstmProjectedStreamsFast&>(GetComponent(c));
      comp.UpdateLstmStreamsState(stream_update_flag);
    }
	if (GetComponent(c).GetType() == Component::kLstmProjectedStreamsFixedPoint) {
      LstmProjectedStreamsFixedPoint& comp = dynamic_cast<LstmProjectedStreamsFixedPoint&>(GetComponent(c));
      comp.UpdateLstmStreamsState(stream_update_flag);
    }
	else if (GetComponent(c).GetType() == Component::kParallelComponentMultiTask) {
	  ParallelComponentMultiTask& comp = dynamic_cast<ParallelComponentMultiTask&>(GetComponent(c));
	  comp.UpdateLstmStreamsState(stream_update_flag);
	}
  }
}

void Nnet::SetSeqLengths(const std::vector<int32> &sequence_lengths) {
  for (int32 c=0; c < NumComponents(); c++) {
    if (GetComponent(c).GetType() == Component::kBLstmProjectedStreams) {
      BLstmProjectedStreams& comp = dynamic_cast<BLstmProjectedStreams&>(GetComponent(c));
      comp.SetSeqLengths(sequence_lengths);
    }
    else if (GetComponent(c).GetType() == Component::kBLstmStreams) {
        BLstmStreams& comp = dynamic_cast<BLstmStreams&>(GetComponent(c));
        comp.SetSeqLengths(sequence_lengths);
      }
  }
}

BaseFloat Nnet::ComputeConditionalLogprob(int32 current_word,
		    const std::vector<int32>  &history_words,
		    const std::vector<CuMatrixBase<BaseFloat> >  &context_in,
		    std::vector<CuMatrix<BaseFloat> >  &context_out,
            int32 unk_sym,
            std::unordered_map<std::string, float> &unk_penalty)
{
    static ClassAffineTransform *class_affine = NULL;
    static std::vector<int32> word2class;
    static std::vector<int32> class_boundary;
    static std::vector<int32> sortedclass_target_index(1,0), sortedclass_target_reindex(1,0);
    static int32 num_class;

    if (class_affine == NULL)
    {
        for (int32 c = 0; c < this->NumComponents(); c++)
        {
        	if (this->GetComponent(c).GetType() == Component::kClassAffineTransform)
        		class_affine = &(dynamic_cast<ClassAffineTransform&>(this->GetComponent(c)));
        }

    	int i,j = 0;
    	class_boundary = class_affine->GetClassBoundary();
    	num_class = class_boundary.size()-1;
    	word2class.resize(class_boundary[num_class]);
    	for (i = 0; i < class_boundary[num_class]; i++)
    	{
    		if (i>=class_boundary[j] && i<class_boundary[j+1])
    			word2class[i] = j;
    		else
    			word2class[i] = ++j;
    	}
    }

    int32 cid = word2class[current_word];
    int32 vocab_size = this->OutputDim() - num_class;
    std::vector<int32> sortedclass_target(1, cid);

    class_affine->SetUpdateClassId(sortedclass_target, sortedclass_target_index, sortedclass_target_reindex);

    // forward pass
    Matrix<BaseFloat> featmat(1, 1, kUndefined);
    Vector<BaseFloat> feat(1, kUndefined);
    CuMatrix<BaseFloat> words(1, 1, kUndefined);
    CuMatrix<BaseFloat> nnet_out(1, this->OutputDim(), kUndefined);
    Matrix<BaseFloat> host_nnet_out(1, this->OutputDim(), kUndefined);
    BaseFloat logprob = 0;

    feat(0) = history_words.back();
    featmat.CopyColFromVec(feat, 0);
    words.CopyFromMat(featmat);

    this->RestoreContext(context_in);
    this->Propagate(words, &nnet_out);
    host_nnet_out.CopyFromMat(nnet_out);
    logprob = log(host_nnet_out(0, current_word) * host_nnet_out(0, vocab_size+cid));
    this->DumpContext(context_out);

    return logprob;
}

void Nnet::RestoreContext(const std::vector<CuMatrixBase<BaseFloat> >  &context)
{
	int idx = 0;
	for (int32 c=0; c < NumComponents(); c++) {
	    if (GetComponent(c).GetType() == Component::kLstmProjectedStreamsFast) {
	      LstmProjectedStreamsFast& comp = dynamic_cast<LstmProjectedStreamsFast&>(GetComponent(c));
	      KALDI_ASSERT(idx < context.size());
	      comp.SetLstmContext(context[idx]);
	      idx++;
	    }
	}
}

void Nnet::DumpContext(std::vector<CuMatrix<BaseFloat> >  &context)
{
	int idx = 0;
	for (int32 c=0; c < NumComponents(); c++) {
	    if (GetComponent(c).GetType() == Component::kLstmProjectedStreamsFast) {
	      LstmProjectedStreamsFast& comp = dynamic_cast<LstmProjectedStreamsFast&>(GetComponent(c));
	      KALDI_ASSERT(idx < context.size());
	      comp.GetLstmContext(&context[idx]);
	      idx++;
	    }
	}
}

int32 Nnet::GetLMNumHiddenLayer()
{
	int num = 0;
	for (int32 c=0; c < NumComponents(); c++) {
	    if (GetComponent(c).GetType() == Component::kLstmProjectedStreamsFast) {
	      num++;
	    }
	}
	return num;
}

void Nnet::SetClassBoundary(std::string classboundary_file)
{
	if (classboundary_file == "")
	{
		KALDI_WARN << "The lm class boundary file '" << classboundary_file << "' is empty.";
		return;
	}

    Input in;
    Vector<BaseFloat> classinfo;
    in.OpenTextMode(classboundary_file);
    classinfo.Read(in.Stream(), false);
    in.Close();

    std::vector<int32> class_boundary;
	class_boundary.resize(classinfo.Dim());
    for (int i = 0; i < classinfo.Dim(); i++)
    	class_boundary[i] = classinfo(i);

    ClassAffineTransform *class_affine = NULL;
    //WordVectorTransform *word_transf = NULL;
    CBSoftmax *cb_softmax = NULL;
    for (int32 c = 0; c < this->NumComponents(); c++)
    {
    	if (this->GetComponent(c).GetType() == Component::kClassAffineTransform)
    		class_affine = &(dynamic_cast<ClassAffineTransform&>(this->GetComponent(c)));
    	else if (this->GetComponent(c).GetType() == Component::kCBSoftmax)
    		cb_softmax = &(dynamic_cast<CBSoftmax&>(this->GetComponent(c)));
    }

    if (NULL != class_affine && NULL != cb_softmax)
    {
	    class_affine->SetClassBoundary(class_boundary);
        cb_softmax->SetClassBoundary(class_boundary);
    }
}

void Nnet::Init(const std::string &file) {
  Input in(file);
  std::istream &is = in.Stream();
  // do the initialization with config lines,
  std::string conf_line, token;
  while (!is.eof()) {
    KALDI_ASSERT(is.good());
    std::getline(is, conf_line); // get a line from config file,
    if (conf_line == "") continue;
    KALDI_VLOG(1) << conf_line; 
    std::istringstream(conf_line) >> std::ws >> token; // get 1st token,
    if (token == "<NnetProto>" || token == "</NnetProto>") continue; // ignored tokens,
    AppendComponent(Component::Init(conf_line+"\n"));
    is >> std::ws;
  }
  // cleanup
  in.Close();
  Check();
}


void Nnet::Read(const std::string &file) {
  bool binary;
  Input in(file, &binary);
  Read(in.Stream(), binary);
  in.Close();
  // Warn if the NN is empty
  if(NumComponents() == 0) {
    KALDI_WARN << "The network '" << file << "' is empty.";
  }
}


void Nnet::Read(std::istream &is, bool binary) {
  // get the network layers from a factory
  Component *comp;
  while (NULL != (comp = Component::Read(is, binary))) {
    if (NumComponents() > 0 && components_.back()->OutputDim() != comp->InputDim()) {
      KALDI_ERR << "Dimensionality mismatch!"
                << " Previous layer output:" << components_.back()->OutputDim()
                << " Current layer input:" << comp->InputDim();
    }
    components_.push_back(comp);
  }
  // create empty buffers
  propagate_buf_.resize(NumComponents()+1);
  backpropagate_buf_.resize(NumComponents()+1);
  // reset learn rate
  opts_.learn_rate = 0.0;
  
  Check(); //check consistency (dims...)
}


void Nnet::Write(const std::string &file, bool binary) const {
  Output out(file, binary, true);
  Write(out.Stream(), binary);
  out.Close();
}


void Nnet::Write(std::ostream &os, bool binary) const {
  Check();
  WriteToken(os, binary, "<Nnet>");
  if(binary == false) os << std::endl;
  for(int32 i=0; i<NumComponents(); i++) {
    components_[i]->Write(os, binary);
  }
  WriteToken(os, binary, "</Nnet>");  
  if(binary == false) os << std::endl;
}


std::string Nnet::Info() const {
  // global info
  std::ostringstream ostr;
  ostr << "num-components " << NumComponents() << std::endl;
  ostr << "input-dim " << InputDim() << std::endl;
  ostr << "output-dim " << OutputDim() << std::endl;
  ostr << "number-of-parameters " << static_cast<float>(NumParams())/1e6 
       << " millions" << std::endl;
  // topology & weight stats
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "component " << i+1 << " : " 
         << Component::TypeToMarker(components_[i]->GetType()) 
         << ", input-dim " << components_[i]->InputDim()
         << ", output-dim " << components_[i]->OutputDim()
         << ", " << components_[i]->Info() << std::endl;
  }
  return ostr.str();
}

std::string Nnet::InfoGradient() const {
  std::ostringstream ostr;
  // gradient stats
  ostr << "### Gradient stats :\n";
  for (int32 i = 0; i < NumComponents(); i++) {
    ostr << "Component " << i+1 << " : " 
         << Component::TypeToMarker(components_[i]->GetType()) 
         << ", " << components_[i]->InfoGradient() << std::endl;
  }
  return ostr.str();
}

std::string Nnet::InfoPropagate() const {
  std::ostringstream ostr;
  // forward-pass buffer stats
  ostr << "### Forward propagation buffer content :\n";
  ostr << "[0] output of <Input> " << MomentStatistics(propagate_buf_[0]) << std::endl;
  for (int32 i=0; i<NumComponents(); i++) {
    ostr << "["<<1+i<< "] output of " 
         << Component::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(propagate_buf_[i+1]) << std::endl;
    // nested networks too...
    if (Component::kParallelComponent == components_[i]->GetType()) {
      ostr << dynamic_cast<ParallelComponent*>(components_[i])->InfoPropagate();
    }
  }
  return ostr.str();
}

std::string Nnet::InfoBackPropagate() const {
  std::ostringstream ostr;
  // forward-pass buffer stats
  ostr << "### Backward propagation buffer content :\n";
  ostr << "[0] diff of <Input> " << MomentStatistics(backpropagate_buf_[0]) << std::endl;
  for (int32 i=0; i<NumComponents(); i++) {
    ostr << "["<<1+i<< "] diff-output of " 
         << Component::TypeToMarker(components_[i]->GetType())
         << MomentStatistics(backpropagate_buf_[i+1]) << std::endl;
    // nested networks too...
    if (Component::kParallelComponent == components_[i]->GetType()) {
      ostr << dynamic_cast<ParallelComponent*>(components_[i])->InfoBackPropagate();
    }
  }
  return ostr.str();
}



void Nnet::Check() const {
  // check we have correct number of buffers,
  KALDI_ASSERT(propagate_buf_.size() == NumComponents()+1);
  KALDI_ASSERT(backpropagate_buf_.size() == NumComponents()+1);
  // check dims,
  for (size_t i = 0; i + 1 < components_.size(); i++) {
    KALDI_ASSERT(components_[i] != NULL);
    int32 output_dim = components_[i]->OutputDim(),
      next_input_dim = components_[i+1]->InputDim();
    KALDI_ASSERT(output_dim == next_input_dim);
  }
  // check for nan/inf in network weights,
  Vector<BaseFloat> weights;
  GetParams(&weights);
  BaseFloat sum = weights.Sum();
  if(KALDI_ISINF(sum)) {
    KALDI_ERR << "'inf' in network parameters (weight explosion, try lower learning rate?)";
  }
  if(KALDI_ISNAN(sum)) {
    KALDI_ERR << "'nan' in network parameters (try lower learning rate?)";
  }
}


void Nnet::Destroy() {
  for(int32 i=0; i<NumComponents(); i++) {
    delete components_[i];
  }
  components_.resize(0);
  propagate_buf_.resize(0);
  backpropagate_buf_.resize(0);
}


void Nnet::SetTrainOptions(const NnetTrainOptions& opts) {
  opts_ = opts;
  //set values to individual components
  for (int32 l=0; l<NumComponents(); l++) {
    if(GetComponent(l).IsUpdatable()) {
      dynamic_cast<UpdatableComponent&>(GetComponent(l)).SetTrainOptions(opts_);
    }
  }
}

void Nnet::SwitchToOnlinePreconditioning(int32 rank_in, int32 rank_out,
                                         int32 update_period,
                                         BaseFloat num_samples_history,
                                         BaseFloat alpha) {
  int32 switched = 0;
  for (size_t i = 0; i < components_.size(); i++) {
    if (dynamic_cast<AffineTransform*>(components_[i]) != NULL) {
    	AffinePreconditionedOnlineTransform *ac =
          new AffinePreconditionedOnlineTransform(
              *(dynamic_cast<AffineTransform*>(components_[i])),
              rank_in, rank_out, update_period, num_samples_history, alpha);
      delete components_[i];
      components_[i] = ac;
      switched++;
    }
  }
  KALDI_LOG << "Switched " << switched << " components to use online "
            << "preconditioning, with (input, output) rank = "
            << rank_in << ", " << rank_out << " and num_samples_history = "
            << num_samples_history;
  //SetIndexes();
  Check();
}

 
} // namespace nnet0
} // namespace kaldi
