// nnet0/nnet-compute-chain-parallel.cc

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

#include <deque>
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "thread/kaldi-semaphore.h"
#include "thread/kaldi-mutex.h"
#include "thread/kaldi-thread.h"

#include "lat/kaldi-lattice.h"

#include "cudamatrix/cu-device.h"
#include "base/kaldi-types.h"


#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-affine-preconditioned-transform.h"
#include "nnet0/nnet-model-merge-function.h"
#include "nnet0/nnet-activation.h"
#include "nnet0/nnet-example.h"

#include "nnet0/nnet-compute-chain-parallel.h"

namespace kaldi {
namespace nnet0 {

class TrainChainParallelClass: public MultiThreadable {

private:
    const NnetChainUpdateOptions *opts;
    NnetModelSync *model_sync;
    fst::StdVectorFst *den_fst;

	std::string feature_transform,
				model_filename,
				si_model_filename;

	ExamplesRepository *repository_;
    NnetChainStats *stats_;

    const NnetTrainOptions *trn_opts;
    const NnetDataRandomizerOptions *rnd_opts;
    const NnetParallelOptions *parallel_opts;

    BaseFloat 	kld_scale;

    std::string use_gpu;
    std::string objective_function;
    int32 num_threads;
    bool crossvalidate;

    unordered_map<std::string, ChainInfo, StringHasher> objf_info_;

 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
    TrainChainParallelClass(const NnetChainUpdateOptions *opts,
			NnetModelSync *model_sync,
		    fst::StdVectorFst *den_fst,
			std::string	model_filename,
			ExamplesRepository *repository,
			NnetChainStats *stats):
				opts(opts),
				model_sync(model_sync),
				den_fst(den_fst),
				model_filename(model_filename),
				repository_(repository),
				stats_(stats)
 	 		{
				trn_opts = opts->trn_opts;
				rnd_opts = opts->rnd_opts;
				parallel_opts = opts->parallel_opts;

				kld_scale = opts->kld_scale;
				objective_function = opts->objective_function;
				use_gpu = opts->use_gpu;
				feature_transform = opts->feature_transform;
				si_model_filename = opts->si_model_filename;

				num_threads = parallel_opts->num_threads;
				crossvalidate = opts->crossvalidate;
 	 		}

	void monitor(Nnet *nnet, kaldi::int64 total_frames, int32 num_frames)
	{
        // 1st minibatch : show what happens in network
        if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
          KALDI_VLOG(1) << "### After " << total_frames << " frames,";
          KALDI_VLOG(1) << nnet->InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(1) << nnet->InfoBackPropagate();
            KALDI_VLOG(1) << nnet->InfoGradient();
          }
        }

        // monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
          if ((total_frames/25000) != ((total_frames+num_frames)/25000)) { // print every 25k frames
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            KALDI_VLOG(2) << nnet->InfoPropagate();
            if (!crossvalidate) {
              KALDI_VLOG(2) << nnet->InfoGradient();
            }
          }
        }
	}

	  // This does the main function of the class.
	void operator ()()
	{
		int thread_idx = this->thread_id_;

		model_sync->LockModel();

	    // Select the GPU
	#if HAVE_CUDA == 1
	    if (parallel_opts->num_procs > 1)
	    {
	    	//thread_idx = model_sync->GetThreadIdx();
	    	KALDI_LOG << "MyId: " << parallel_opts->myid << "  ThreadId: " << thread_idx;
	    	CuDevice::Instantiate().MPISelectGpu(model_sync->gpuinfo_, model_sync->win, thread_idx, this->num_threads);
	    	for (int i = 0; i< this->num_threads*parallel_opts->num_procs; i++)
	    	{
	    		KALDI_LOG << model_sync->gpuinfo_[i].hostname << "  myid: " << model_sync->gpuinfo_[i].myid
	    					<< "  gpuid: " << model_sync->gpuinfo_[i].gpuid;
	    	}
	    }
	    else
	    	CuDevice::Instantiate().SelectGpu();

	    //CuDevice::Instantiate().DisableCaching();
	#endif

	    model_sync->UnlockModel();

		Nnet nnet_transf;
	    if (feature_transform != "") {
	      nnet_transf.Read(feature_transform);
	    }

	    Nnet nnet;
	    nnet.Read(model_filename);

	    nnet.SetTrainOptions(*trn_opts);

	    int32 rank_in = 20, rank_out = 80, update_period = 4;
	   	    BaseFloat num_samples_history = 2000.0;
	   	    BaseFloat alpha = 4.0;

	    if (opts->use_psgd)
	    	nnet.SwitchToOnlinePreconditioning(rank_in, rank_out, update_period, num_samples_history, alpha);

	    if (opts->dropout_retention > 0.0) {
	      nnet_transf.SetDropoutRetention(opts->dropout_retention);
	      nnet.SetDropoutRetention(opts->dropout_retention);
	    }
	    if (crossvalidate) {
	      nnet_transf.SetDropoutRetention(1.0);
	      nnet.SetDropoutRetention(1.0);
	    }

	    Nnet si_nnet;
	    if (this->kld_scale > 0)
	    	si_nnet.Read(si_model_filename);

	    // multi-thread initialization
	    model_sync->Initialize(&nnet);

	    RandomizerMask randomizer_mask(*rnd_opts);
	    MatrixRandomizer feature_randomizer(*rnd_opts);
	    PosteriorRandomizer targets_randomizer(*rnd_opts);
	    VectorRandomizer weights_randomizer(*rnd_opts);

		ModelMergeFunction *p_merge_func = model_sync->GetModelMergeFunction();

		chain::DenominatorGraph den_graph(*den_fst, nnet.OutputDim());

        bool use_xent = (opts->chain_config.xent_regularize != 0.0);
        std::vector<int32> sweep_frames;
        if (!kaldi::SplitStringToIntegers(opts->sweep_frames_str, ":", false, &sweep_frames))
        	KALDI_ERR << "Invalid sweep-frames string " << opts->sweep_frames_str;

		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0, num_dump = 0, num_minibatch = 0;
		kaldi::int64 total_frames = 0;

        int32 num_stream = opts->num_stream;
        int32 targets_delay = opts->targets_delay;
        //int32 batch_size= opts->batch_size;
        int32 skip_frames = opts->skip_frames;
        int32 context_left = opts->context_left;

        std::vector<int> new_utt_flags(num_stream, 1);
	    CuMatrix<BaseFloat> cu_feat_mat, cu_feat_utts;
		CuMatrix<BaseFloat> feats_transf, nnet_out, nnet_diff, xent_logsoftmax;
		CuSubMatrix<BaseFloat> *mmi_nnet_out = NULL, *xent_nnet_out = NULL,
								*mmi_deriv = NULL, *xent_deriv = NULL;
        Matrix<BaseFloat> feat_mat_host, feat_utts;

	    ChainNnetExample *chain_example = NULL;
	    NnetExample		*example = NULL;
	    Timer time;
	    double time_now = 0;

		int32 feat_dim = nnet.InputDim();
		int32 out_dim = nnet.OutputDim();


		while((example = repository_->ProvideExample()) != NULL) {

			int size = 0, minibatch = 0, utt_frame_num = 0,
					utt_len, offset, ctx_left, reset = false, nbptt_truncated;
            if (chain_example != NULL) 
               delete chain_example;
			chain_example = dynamic_cast<ChainNnetExample*>(example);
			const kaldi::nnet3::NnetIo &io = chain_example->chain_eg.inputs[0];
	        const kaldi::nnet3::NnetChainSupervision &sup = chain_example->chain_eg.outputs[0];

			size = io.indexes.size();
			minibatch = io.indexes[size-1].n + 1;
			reset = num_stream == minibatch ? false : true;
			num_stream = minibatch;
            new_utt_flags.resize(num_stream, 1);
            //KALDI_WARN << "utterances per example not equal with number of stream, force set stream " << minibatch; 
			utt_frame_num = size/minibatch;
			utt_len = sup.supervision.frames_per_sequence;
			offset = -io.indexes[0].t;
			ctx_left = offset/skip_frames;
			if (context_left > ctx_left)
				KALDI_WARN << "egs left context is " << ctx_left << ", it will be force use the small one";
			ctx_left = (ctx_left > context_left && context_left != -1) ? context_left : ctx_left;

			nbptt_truncated = utt_len;
            KALDI_ASSERT(utt_frame_num-offset >=  (utt_len+targets_delay)*skip_frames);

            cu_feat_utts.Resize(io.features.NumRows(), io.features.NumCols(), kUndefined);
            //feat_utts.Resize(io.features.NumRows(), io.features.NumCols(), kUndefined);
			io.features.CopyToMat(&cu_feat_utts);
			//io.features.CopyToMat(&feat_utts);
            //io.features.SwapFullMatrix(&feat_utts);

			// Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
			int in_frames = (ctx_left+targets_delay+utt_len)*num_stream;
			int row_start = (ctx_left+targets_delay)*num_stream, chunk_frames = in_frames-row_start;

			if (reset) {
                KALDI_LOG << "egs left context: " << offset << ", actually: " << row_start/num_stream << " frames.";
				cu_feat_mat.Resize(in_frames, feat_dim, kUndefined);
				nnet_out.Resize(in_frames, out_dim, kUndefined);
				nnet_diff.Resize(in_frames, out_dim, kSetZero);
				if (mmi_nnet_out) delete mmi_nnet_out;
				if (xent_nnet_out) delete xent_nnet_out;
				if (mmi_deriv) delete mmi_deriv;
				if (xent_deriv) delete xent_deriv;

				if (use_xent) {
					// except row_start histroy
					mmi_nnet_out = new CuSubMatrix<BaseFloat>(nnet_out.Range(row_start, chunk_frames, 0, out_dim/2));
					xent_nnet_out = new CuSubMatrix<BaseFloat>(nnet_out.Range(row_start, chunk_frames, out_dim/2, out_dim/2));
					mmi_deriv = new CuSubMatrix<BaseFloat>(nnet_diff.Range(row_start, chunk_frames, 0, out_dim/2));
					xent_deriv = new CuSubMatrix<BaseFloat>(nnet_diff.Range(row_start, chunk_frames, out_dim/2, out_dim/2));
					xent_logsoftmax.Resize(chunk_frames, out_dim/2, kUndefined);
				} else {
					mmi_nnet_out = new CuSubMatrix<BaseFloat>(nnet_out.Range(row_start, chunk_frames, 0, out_dim));
					mmi_deriv = new CuSubMatrix<BaseFloat>(nnet_diff.Range(row_start, chunk_frames, 0, out_dim));
					xent_nnet_out = NULL;
					xent_deriv = NULL;
				}
			}

			num_frames = chunk_frames;
            // report the speed
            if ((num_done/minibatch) % 300 == 0) {
              time_now = time.Elapsed();
              KALDI_VLOG(1) << "After " << num_done << " utterances chunk: time elapsed = "
                            << time_now/60 << " min; processed " << total_frames/time_now
                            << " frames per second.";
            }

	// sweep each frame
	for (int n = 0; n < sweep_frames.size(); n++) {

			num_done += minibatch; // number stream of short utterances for a minibatch
			num_minibatch++; // num_minibatches_processed_

			// rearrange utterance
			int s, t, len = in_frames/num_stream; //his_len = ctx_left+targets_delay;
			// sweep each frame
			int cur_offset = (offset-ctx_left*skip_frames+sweep_frames[n]);
			std::vector<int32> indexes(in_frames);
			for (t = 0; t < len; t++) {
				for (s = 0; s < num_stream; s++) {
					indexes[t*num_stream+s] = s*utt_frame_num + cur_offset + t*skip_frames;
				}
			}

			CuArray<int32> idx(indexes);
			cu_feat_mat.CopyRows(cu_feat_utts, idx);

			// apply optional feature transform
			nnet_transf.Feedforward(cu_feat_mat, &feats_transf);

	        // for streams with new utterance, history states need to be reset
	        nnet.ResetLstmStreams(new_utt_flags);
	        //nnet.ResetLstmStreams(new_utt_flags, nbptt_truncated);

	        // forward pass
	        nnet.Propagate(feats_transf, &nnet_out);
            //mmi_nnet_out->ApplyLog();

	        BaseFloat tot_objf, tot_l2_term, tot_weight;
	        // get mmi objective function derivative, and xent soft supervision label
			kaldi::chain::ComputeChainObjfAndDeriv(opts->chain_config, den_graph,
									 sup.supervision, *mmi_nnet_out,
									 &tot_objf, &tot_l2_term, &tot_weight,
									 mmi_deriv,
									 (use_xent ? xent_deriv : NULL));
			objf_info_["mmi"].UpdateStats("mmi", opts->print_interval, num_minibatch,
													tot_weight, tot_objf, tot_l2_term);
            mmi_deriv->Scale(-1.0);

			// this block computes the cross-entropy objective.
			if (use_xent) {
				// log softmax
                xent_logsoftmax.CopyFromMat(*xent_nnet_out);
				//xent_logsoftmax.Add(1e-20); // avoid log(0)
				xent_logsoftmax.ApplyLog(); // log(y)

				// at this point, xent_deriv is posteriors derived from the numerator
				// computation.  note, xent_objf has a factor of '.supervision.weight'
				BaseFloat xent_objf = TraceMatMat(xent_logsoftmax, *xent_deriv, kTrans); // sum(t*log(y))
				objf_info_["xent"].UpdateStats("xent", opts->print_interval, num_minibatch,
				                                        			tot_weight, xent_objf);

				// xent derivative
				xent_deriv->AddMat(-1.0, *xent_nnet_out);
				// cross entropy regularization,
				xent_deriv->Scale(-1.0 * opts->chain_config.xent_regularize);
			}

			// weighting ...
			if (opts->apply_deriv_weights && sup.deriv_weights.Dim() != 0) {
			      CuVector<BaseFloat> cu_deriv_weights(sup.deriv_weights);
			      mmi_nnet_out->MulRowsVec(cu_deriv_weights);
			      if (use_xent)
			        xent_deriv->MulRowsVec(cu_deriv_weights);
			}

		        // backward pass
				if (!crossvalidate) {
					// backpropagate

                    if (model_sync->reset_gradient_[thread_idx] && parallel_opts->merge_func == "globalgradient") {
                        nnet.ResetGradient();
                        model_sync->reset_gradient_[thread_idx] = false;
                        //KALDI_VLOG(1) << "Reset Gradient";
                    }

					if (parallel_opts->num_threads > 1 && update_frames >= opts->update_frames) {
						nnet.Backpropagate(nnet_diff, NULL, false);
						nnet.Gradient();

						//t2 = time.Elapsed();
						//time.Reset();

						if (parallel_opts->asgd_lock)
							model_sync->LockModel();

						model_sync->SetWeight(&nnet);
						nnet.UpdateGradient();
						model_sync->GetWeight(&nnet);

						if (parallel_opts->asgd_lock)
							model_sync->UnlockModel();

						update_frames = 0;

					} else {
						nnet.Backpropagate(nnet_diff, NULL, true);
					}

					//multi-machine
					if (parallel_opts->num_procs > 1)
					{
						model_sync->LockModel();

						if (p_merge_func->CurrentMergeCache() + num_frames > parallel_opts->merge_size)
						{
							if (p_merge_func->leftMerge() <= 1 && !p_merge_func->isLastMerge())
							{
								p_merge_func->MergeStatus(1);
							}

							if (p_merge_func->leftMerge() > 1 || !p_merge_func->isLastMerge())
							{
								model_sync->GetWeight(&nnet);

							    p_merge_func->Merge(0);
							    KALDI_VLOG(1) << "Model merge NO." << parallel_opts->num_merge - p_merge_func->leftMerge()
							    				<< " Current mergesize = " << p_merge_func->CurrentMergeCache() << " frames.";
							    p_merge_func->MergeCacheReset();

							    model_sync->SetWeight(&nnet);
                                model_sync->ResetGradient();
							}
						}

						p_merge_func->AddMergeCache((int) num_frames);

						model_sync->UnlockModel();

					}
				}

				monitor(&nnet, total_frames, num_frames);

				// increase time counter
		        update_frames += num_frames;
		        total_frames += num_frames;

		        // track training process
			    if (!crossvalidate && this->thread_id_ == 0 && parallel_opts->myid == 0 && opts->dump_time > 0)
				{
                    int num_procs = parallel_opts->num_procs > 1 ? parallel_opts->num_procs : 1;
					if ((total_frames*parallel_opts->num_threads*num_procs)/(3600*100*opts->dump_time) > num_dump)
					{
						char name[512];
						num_dump++;
						sprintf(name, "%s_%d_%ld", model_filename.c_str(), num_dump, total_frames);
						nnet.Write(std::string(name), true);
					}
				}

		        fflush(stderr);
		        fsync(fileno(stderr));
			} //sweep frames
		}

		model_sync->LockStates();

		stats_->total_frames += total_frames;
		stats_->num_done += num_done;
		stats_->Add(objf_info_);

		model_sync->UnlockStates();

		//last merge
		if (!crossvalidate){
			model_sync->LockModel();

			bool last_thread = true;
			for (int i = 0; i < parallel_opts->num_threads; i++)
			{
				if (i != thread_idx && !model_sync->isfinished_[i]){
						last_thread = false;
						break;
				}
			}

			if (parallel_opts->num_procs > 1)
			{
				if (last_thread)
				{
					if (!p_merge_func->isLastMerge())
						p_merge_func->MergeStatus(0);

					model_sync->GetWeight(&nnet);

					p_merge_func->Merge(0);
						KALDI_VLOG(1) << "Model merge NO." << parallel_opts->num_merge-p_merge_func->leftMerge()
									   << " Current mergesize = " << p_merge_func->CurrentMergeCache() << " frames.";
						model_sync->SetWeight(&nnet);
				}
			}

			if (last_thread)
			{
				KALDI_VLOG(1) << "Last thread upload model to host.";
					model_sync->CopyToHost(&nnet);
			}

			model_sync->isfinished_[thread_idx] = true;
			model_sync->UnlockModel();
		}
	}

};


void NnetChainUpdateParallel(const NnetChainUpdateOptions *opts,
		fst::StdVectorFst *den_fst,
		std::string	model_filename,
		std::string feature_rspecifier,
		Nnet *nnet,
		NnetChainStats *stats) {
		ExamplesRepository repository;
		NnetModelSync model_sync(nnet, opts->parallel_opts);

		TrainChainParallelClass c(opts, &model_sync,
								den_fst,
								model_filename,
								&repository, stats); {

			kaldi::nnet3::SequentialNnetChainExampleReader example_reader(feature_rspecifier);

			// The initialization of the following class spawns the threads that
			// process the examples.  They get re-joined in its destructor.
			MultiThreader<TrainChainParallelClass> mc(opts->parallel_opts->num_threads, c);
			NnetExample *example;
			std::vector<NnetExample*> examples;
			for (; !example_reader.Done(); example_reader.Next()) {
					example = new ChainNnetExample(&example_reader);
					if (example->PrepareData(examples)) {
						for (int i = 0; i < examples.size(); i++)
							repository.AcceptExample(examples[i]);
						if (examples[0] != example)
							delete example;
					}
					else
						delete example;
			}
			repository.ExamplesDone();
	  }
}

} // namespace nnet0
} // namespace kaldi


