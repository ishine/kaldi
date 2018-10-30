// online0/online-nnet3-forward.h
// Copyright 2018   Alibaba Inc (author: Wei Deng)

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

#ifndef ONLINE0_ONLINE_NNET3_FORWARD_H_
#define ONLINE0_ONLINE_NNET3_FORWARD_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {

struct OnlineNnet3ForwardOptions {
	typedef nnet3::NnetSimpleComputationOptions NnetSimpleComputationOptions;
	typedef nnet3::CachingOptimizingCompilerOptions CachingOptimizingCompilerOptions;

    std::string use_gpu;
    std::string network_model;

    NnetSimpleComputationOptions compute_opts;
    CachingOptimizingCompilerOptions compiler_config;

    OnlineNnet3ForwardOptions():use_gpu("no"){}

    void Register(OptionsItf *po) {
        compiler_config.Register(po);
        compute_opts.Register(po);

        po->Register("use-gpu", &use_gpu,
          "yes|no|optional|wait, only has effect if compiled with CUDA");
    	po->Register("network-model", &network_model, "Main neural network model (in nnet0 format)");
    }
};

class OnlineNnet3Forward {
public:
	OnlineNnet3Forward(OnlineNnet3ForwardOptions *opts):
		opts_(opts),compiler_(NULL) {
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(opts->use_gpu);
#endif

    	using namespace kaldi::nnet3;

        opts->compute_opts.acoustic_scale = 1.0;
    	ReadKaldiObject(opts->network_model, &nnet_);
		SetBatchnormTestMode(true, &nnet_);
		SetDropoutTestMode(true, &nnet_);
		CollapseModel(CollapseModelConfig(), &nnet_);
        //compiler_ = new CachingOptimizingCompiler(nnet_, opts->compute_opts.optimize_config);
        compiler_ = new CachingOptimizingCompiler(nnet_, opts->compute_opts.optimize_config, opts->compiler_config);
	}

    virtual ~OnlineNnet3Forward() {
        delete compiler_;
        compiler_ = NULL;
    }

	void Forward(const MatrixBase<BaseFloat> &in, Matrix<BaseFloat> *out) {
		using namespace kaldi::nnet3;

        feat_.Resize(in.NumRows(), in.NumCols(), kUndefined);
		feat_.CopyFromMat(in);

		ComputationRequest request;
		request.need_model_derivative = false;
		request.store_component_stats = false;
		request.inputs.push_back(IoSpecification("input", 0, in.NumRows()));
		IoSpecification output_spec;
		output_spec.name = "output";
		output_spec.has_deriv = false;
		output_spec.indexes.resize(1);
		request.outputs.resize(1);
		request.outputs[0].Swap(&output_spec);

		std::shared_ptr<const NnetComputation> computation(std::move(compiler_->Compile(request)));
        //const NnetComputation *computation = compiler_->Compile(request);
		Nnet *nnet_to_update = NULL;  // we're not doing any update.
		NnetComputer computer(NnetComputeOptions(), *computation, nnet_, nnet_to_update);

		computer.AcceptInput("input", &feat_);
		computer.Run();
		computer.GetOutputDestructive("output", &feat_out_);

		out->Resize(feat_out_.NumRows(), feat_out_.NumCols(), kUndefined);
		out->CopyFromMat(feat_out_);
	}

	int OutputDim() {
		return nnet_.OutputDim("output");
	}
private:
	OnlineNnet3ForwardOptions *opts_;
	kaldi::nnet3::Nnet nnet_;
	kaldi::nnet3::CachingOptimizingCompiler *compiler_;
	CuMatrix<BaseFloat> feat_;
	CuMatrix<BaseFloat> feat_out_;
};

}

#endif /* ONLINE0_ONLINE_NNET3_FORWARD_H_ */
