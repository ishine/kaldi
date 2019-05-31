// nnet0/nnet-multi-net-component.h

// Copyright 2014  Brno University of Technology (Author: Karel Vesely)
// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

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


#ifndef KALDI_NNET_NNET_MULTI_NET_COMPONENT_H_
#define KALDI_NNET_NNET_MULTI_NET_COMPONENT_H_


#include "nnet0/nnet-component.h"
#include "nnet0/nnet-utils.h"
#include "nnet0/nnet-nnet.h"
#include "cudamatrix/cu-math.h"

#include <sstream>

namespace kaldi {
namespace nnet0 {

class MultiNetComponent : public UpdatableComponent {
 public:
	MultiNetComponent(int32 dim_in, int32 dim_out)
    : UpdatableComponent(dim_in, dim_out)
  { }
  ~MultiNetComponent()
  { }

  Component* Copy() const { return new MultiNetComponent(*this); }
  ComponentType GetType() const { return kMultiNetComponent; }

  void InitData(std::istream &is) {
    // define options
    // std::vector<std::string> nested_nnet_proto;
    // std::vector<std::string> nested_nnet_filename;
    // parse config
    std::string token, name;
	int32 offset, len = 0; 
    BaseFloat scale, escale;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token); 
      if (token == "<NestedNnet>" || token == "<NestedNnetFilename>") {
    	  ExpectToken(is, false, "<Name>");
    	  ReadToken(is, false, &name);

          std::string file_or_end;
          ReadToken(is, false, &file_or_end);

          // read nnets from files
          Nnet nnet;
          nnet.Read(file_or_end);
          nnet_[name] = nnet;
          KALDI_LOG << "Loaded nested <Nnet> from file : " << file_or_end;

          ReadToken(is, false, &file_or_end);
          KALDI_ASSERT(file_or_end == "</NestedNnet>" || file_or_end == "</NestedNnetFilename>");

      } else if (token == "<NestedNnetProto>") {
    	  ExpectToken(is, false, "<Name>");
    	  ReadToken(is, false, &name);

          std::string file_or_end;
          ReadToken(is, false, &file_or_end);

          // initialize nnets from prototypes
          Nnet nnet;
          nnet.Init(file_or_end);
          nnet_[name] = nnet;
          KALDI_LOG << "Initialized nested <Nnet> from prototype : " << file_or_end;

          ReadToken(is, false, &file_or_end);
          KALDI_ASSERT(file_or_end == "</NestedNnetProto>");

      } else KALDI_ERR << "Unknown token " << token << ", typo in config?"
                       << " (NestedNnet|NestedNnetFilename|NestedNnetProto)";
      is >> std::ws; // eat-up whitespace
    }
    // initialize
    // KALDI_ASSERT((nested_nnet_proto.size() > 0) ^ (nested_nnet_filename.size() > 0)); //xor
    KALDI_ASSERT(nnet_.size() > 0);
  }

  void ReadData(std::istream &is, bool binary) {
    // read
    ExpectToken(is, binary, "<NestedNnetCount>");
    std::string name;
    int32 nnet_count;
    ReadBasicType(is, binary, &nnet_count);
    for (int32 i=0; i<nnet_count; i++) {
      ExpectToken(is, binary, "<NestedNnet>");
      int32 dummy;
      ReadBasicType(is, binary, &dummy);

      ExpectToken(is, false, "<Name>");
      ReadToken(is, false, &name);

      Nnet nnet;
      nnet.Read(is, binary);
      nnet_[name] = nnet;
    }
    ExpectToken(is, binary, "</MultiNetComponent>");
  }

  void WriteData(std::ostream &os, bool binary) const {
    // useful dims
    int32 nnet_count = nnet_.size();
    std::string name;
    //unordered_map<std::string, std::pair<int32, int32> >::iterator it;
    int32 i = 0;
    //
    WriteToken(os, binary, "<NestedNnetCount>");
    WriteBasicType(os, binary, nnet_count);

    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
    	name = it->first;
        WriteToken(os, binary, "<NestedNnet>");
        WriteBasicType(os, binary, i+1);

        WriteToken(os, binary, "<Name>");
        WriteToken(os, binary, name);
        if(binary == false) os << std::endl;

        nnet_.find(name)->second.Write(os, binary);
        if(binary == false) os << std::endl;
    }
    WriteToken(os, binary, "</MultiNetComponent>");
  }

  int32 NumParams() const { 
    int32 num_params_sum = 0;
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
      num_params_sum += it->second.NumParams();
    return num_params_sum;
  }

  void GetParams(Vector<BaseFloat>* wei_copy) const { 
    wei_copy->Resize(NumParams());
    int32 offset = 0;
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      Vector<BaseFloat> wei_aux;
      it->second.GetParams(&wei_aux);
      wei_copy->Range(offset, wei_aux.Dim()).CopyFromVec(wei_aux);
      offset += wei_aux.Dim();
    }
    KALDI_ASSERT(offset == NumParams());
  }
    
  std::string Info() const { 
    std::ostringstream os;
    os << "\n";
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      os << "nested_network #" << it->first << "{\n" << it->second.Info() << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1); // removing last '\n'
    return s;
  }
                       
  std::string InfoGradient() const {
    std::ostringstream os;
    os << "\n";
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      os << "nested_gradient #" << it->first << "{\n"
         << it->second.InfoGradient() << "}\n";
    }
    std::string s(os.str());
    s.erase(s.end() -1); // removing last '\n'
    return s;
  }

  std::string InfoPropagate() const {
    std::ostringstream os;
    os << "\n";
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      os << "nested_propagate #" << it->first << "{\n"
         << it->second.InfoPropagate() << "}\n";
    }
    return os.str();
  }

  std::string InfoBackPropagate() const {
    std::ostringstream os;
    os << "\n";
    for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
      os << "nested_backpropagate #" << it->first << "{\n"
    	 <<  it->second.InfoBackPropagate() << "}\n";
    }
    return os.str();
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
	  // unimplemented
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
	  // unimplemented
  }

  void Gradient(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
	  // unimplemented
  }

  void UpdateGradient() {
	  // unimplemented
  }

  void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
	  // do nothing
  }
 
  void SetTrainOptions(const NnetTrainOptions &opts) {
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
		  it->second.SetTrainOptions(opts);
    }
  }

  int32 GetDim() const {
	  int32 dim = 0;
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
		  dim += it->second.GetDim();
	  return dim;
  }

  int WeightCopy(void *host, int direction, int copykind) {
	  int pos = 0;
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it)
		  pos += it->second.WeightCopy((void*)((char *)host+pos), direction, copykind);
	  return pos;
  }

  Component* GetComponent(Component::ComponentType type) {
	  Component *com = NULL;
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
		  Nnet &nnet = it->second;
		  for (int32 c = 0; c < nnet.NumComponents(); c++) {

			  if (nnet.GetComponent(c).GetType() == type) {
				  com = &nnet.GetComponent(c);
				  return com;
			  } else if (nnet.GetComponent(c).GetType() == Component::kMultiNetComponent) {
				  com = (dynamic_cast<MultiNetComponent&>(nnet.GetComponent(c))).GetComponent(type);
				  if (com != NULL) return com;
			  }
		  }
	  }
	  return com;
  }

  void ResetGradient() {
	  for (auto it = nnet_.begin(); it != nnet_.end(); ++it) {
		  Nnet &nnet = it->second;
		  nnet.ResetGradient();
	  }
  }

  std::unordered_map<std::string, Nnet> &GetNnet() {
	  return nnet_;
  }

  Nnet &GetNestNnet(std::string name) {
	  return nnet_[name];
  }

 private:
	std::unordered_map<std::string, Nnet> nnet_;

};

} // namespace nnet0
} // namespace kaldi

#endif
