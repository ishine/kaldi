// online0/online-nnet-ipc-forward.h

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

#ifndef ONLINE0_ONLINE_NNET_IPC_FORWAR_H_
#define ONLINE0_ONLINE_NNET_IPC_FORWAR_H_

#include "util/circular-queue.h"
#include "util/kaldi-mutex.h"
#include "nnet0/nnet-trnopts.h"
#include "nnet0/nnet-pdf-prior.h"
#include "online0/online-util.h"

#include "online0/online-ipc-message.h"
#include "online0/kaldi-unix-domain-socket.h"

namespace kaldi {

struct OnlineNnetIpcForwardOptions {
    typedef nnet0::PdfPriorOptions PdfPriorOptions;
    std::string feature_transform;
    std::string network_model;
    std::string socket_path;
    bool no_softmax;
    bool apply_log;
    bool copy_posterior;
    std::string use_gpu;
    int32 gpuid;
    int32 num_threads;
    float blank_posterior_scale;
    std::string network_type;

    int32 batch_size;
    int32 num_stream;
    int32 skip_frames;
    int32 input_dim;
    int32 output_dim;
    bool skip_inner;

    const PdfPriorOptions *prior_opts;

    OnlineNnetIpcForwardOptions(const PdfPriorOptions *prior_opts)
    	:feature_transform(""),network_model(""),socket_path(""),
		no_softmax(false),apply_log(false),copy_posterior(false),
								 use_gpu("no"),gpuid(-1),num_threads(1),
								 blank_posterior_scale(-1.0),network_type("lstm"),
		 	 	 	 	 	 	 batch_size(18),num_stream(10),
								 skip_frames(1),skip_inner(false),
								 prior_opts(prior_opts) {
    }

    void Register(OptionsItf *po) {
    	po->Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet0 format)");
    	po->Register("network-model", &network_model, "Main neural network model (in nnet0 format)");
    	po->Register("socket-path", &socket_path, "Unix domain socket file path");
    	po->Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    	po->Register("apply-log", &apply_log, "Transform MLP output to logscale");
    	po->Register("copy-posterior", &copy_posterior, "Copy posterior for skip frames output");
    	po->Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
        po->Register("gpuid", &gpuid, "gpuid < 0 for automatic select gpu, gpuid >= 0 for select specified gpu, only has effect if compiled with CUDA");
    	po->Register("num-threads", &num_threads, "Number of threads(GPUs) to use");
        po->Register("blank-posterior-scale", &blank_posterior_scale, "For CTC decoding, scale blank label posterior by a constant value(e.g. 0.11), other label posteriors are directly used in decoding.");
        po->Register("network-type", &network_type, "multi-stream forward neural network type, (lstm|fsmn)");

        po->Register("batch-size", &batch_size, "---LSTM--- BPTT batch size");
        po->Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training");
        po->Register("skip-frames", &skip_frames, "LSTM model skip frames for next input");
        po->Register("skip-inner", &skip_inner, "skip frame in neural network inner or input");
        po->Register("input-dim", &input_dim, "neural network model input dim");
        po->Register("output-dim", &output_dim, "neural network model output dim");
    }

};

class IpcForwardSync {
public:
    IpcForwardSync(): time_now_(0), time_send_(0),
						time_received_(0), time_forward_(0) {
    	repo_in_ = new Repository(1);
    	repo_out_ = new Repository(1);
    }

    virtual ~IpcForwardSync() {
    	delete repo_in_;
    	delete repo_out_;
    }

    void LockGpu() {
        gpu_mutex_.Lock();
    }

    void UnlockGpu() {
        gpu_mutex_.Unlock();
    }

    std::vector<int> new_utt_flags_;
    std::vector<int> update_state_flags_;
    double time_now_;
    double time_send_;
    double time_received_;
    double time_forward_;

private:
    Mutex gpu_mutex_;
    Repository *repo_in_;
    Repository *repo_out_;
};

class OnlineNnetIpcForwardClass : public MultiThreadable {
private:
	const static int MAX_BUFFER_SIZE = 10;
	const static int MATRIX_INC_STEP = 1024;
	const OnlineNnetIpcForwardOptions &opts_;
    IpcForwardSync &forward_sync_;
	std::string model_filename_;
        
public:
	OnlineNnetIpcForwardClass(const OnlineNnetIpcForwardOptions &opts,
			IpcForwardSync &forward_sync, std::string model_filename):
				opts_(opts), forward_sync_(forward_sync), model_filename_(model_filename) {

	}

	~OnlineNnetIpcForwardClass() {}

	void operator () ()
	{
        forward_sync_.LockGpu();
#if HAVE_CUDA==1
		if (opts_.use_gpu == "yes") {
			if (opts_.gpuid < 0)
				CuDevice::Instantiate().SelectGpu();
			else
				CuDevice::Instantiate().SelectPreferGpu(opts_.gpuid);
		}
#endif
        forward_sync_.UnlockGpu();

        using namespace kaldi::nnet0;

		bool no_softmax = opts_.no_softmax;
		std::string feature_transform = opts_.feature_transform;
		bool apply_log = opts_.apply_log;
		int32 num_stream = opts_.num_stream;
		int32 batch_size = opts_.batch_size;
		int32 skip_frames = opts_.skip_frames;
		const PdfPriorOptions *prior_opts = opts_.prior_opts;

		Nnet nnet_transf;

	    if (feature_transform != "")
	    	nnet_transf.Read(feature_transform);

	    Nnet nnet;
	    nnet.Read(model_filename_);

	    // optionally remove softmax,
	    Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
	    if (no_softmax) {
	      if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
	        KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename_;
	        nnet.RemoveComponent(nnet.NumComponents()-1);
	      } else {
	        KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
	      }
	    }

	    // avoid some bad option combinations,
	    if (apply_log && no_softmax) {
	    	KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
	    }

	    // we will subtract log-priors later,
	    PdfPrior pdf_prior(*prior_opts);

	    CuMatrix<BaseFloat>  cufeat, feats_transf, nnet_out;
	    Matrix<BaseFloat> *nnet_in_host, nnet_out_host;
	    std::vector<int> &new_utt_flags = forward_sync_.new_utt_flags_;
	    std::vector<int> &update_state_flags = forward_sync_.update_state_flags_;

        Timer gap_time;

	    while (true) {

	    	nnet_in_host = forward_sync_.repo_in_->Provide();
	    	if (nnet_in_host == NULL) {
	    		forward_sync_.repo_out_->Done();
	    		break;
	    	}

	    	gap_time.Reset();
	    	// apply optional feature transform
	    	cufeat.Resize(nnet_in_host->NumRows(), nnet_in_host->NumCols(), kUndefined);
            cufeat.CopyFromMat(*nnet_in_host);
	    	nnet_transf.Propagate(cufeat, &feats_transf); // Feedforward

			// for streams with new utterance, history states need to be reset
			nnet.ResetLstmStreams(new_utt_flags);
			nnet.SetSeqLengths(new_utt_flags);
	    	// for streams with new data, history states need to be update
	    	nnet.UpdateLstmStreamsState(update_state_flags);

			// forward pass
			nnet.Propagate(feats_transf, &nnet_out);

	    	// convert posteriors to log-posteriors,
	    	if (apply_log) {
				nnet_out.Add(1e-20); // avoid log(0),
				nnet_out.ApplyLog();
	    	}

	    	// subtract log-priors from log-posteriors or pre-softmax,
	    	if (prior_opts->class_frame_counts != "") {
	    		pdf_prior.SubtractOnLogpost(&nnet_out);
	    	}

	    	if (nnet_out_host.NumRows() != nnet_out.NumRows())
	    		nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols(), kSetZero, kStrideEqualNumCols);

			nnet_out.CopyToMat(&nnet_out_host);
			forward_sync_.time_forward_ += gap_time.Elapsed();

			forward_sync_.repo_out_->Accept(&nnet_out_host);

	    } // while loop
	}
};


}// namespace kaldi

#endif /* ONLINE0_ONLINE_NNET_IPC_FORWARD_H_ */
