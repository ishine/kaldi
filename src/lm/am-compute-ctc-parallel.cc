// lm/am-compute-ctc-parallel.cc

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
#include "util/kaldi-semaphore.h"
#include "util/kaldi-mutex.h"
#include "util/kaldi-thread.h"
#include "lat/kaldi-lattice.h"
#include "cudamatrix/cu-device.h"
#include "base/kaldi-types.h"


#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-activation.h"

#include "lm/example.h"
#include "lm/am-compute-ctc-parallel.h"

namespace kaldi {
namespace lm {

typedef nnet0::NnetTrainOptions NnetTrainOptions;
typedef nnet0::NnetDataRandomizerOptions NnetDataRandomizerOptions;
typedef nnet0::NnetParallelOptions NnetParallelOptions;
typedef nnet0::Component Component;
typedef nnet0::Softmax Softmax;
typedef nnet0::LogSoftmax LogSoftmax;
typedef nnet0::RandomizerMask RandomizerMask;
typedef nnet0::MatrixRandomizer MatrixRandomizer;
typedef nnet0::VectorRandomizer VectorRandomizer;
typedef nnet0::PosteriorRandomizer PosteriorRandomizer;
typedef nnet0::Xent Xent;
typedef nnet0::Mse	Mse;
typedef nnet0::MultiTaskLoss MultiTaskLoss;
typedef nnet0::CtcItf CtcItf;
typedef nnet0::Ctc Ctc;
typedef nnet0::WarpCtc WarpCtc;

class TrainCtcParallelClass: public MultiThreadable {

private:
    const NnetCtcUpdateOptions *opts;
    LmModelSync *model_sync;

	std::string feature_transform,
				model_filename,
				target_model_filename,
				si_model_filename,
				targets_rspecifier;

	ExamplesRepository *repository_;
	Nnet *host_nnet_;
    NnetStats *stats_;

    const NnetTrainOptions *trn_opts;
    const NnetDataRandomizerOptions *rnd_opts;
    const NnetParallelOptions *parallel_opts;

    BaseFloat kld_scale;

    std::string use_gpu;
    std::string objective_function;
    int32 num_threads;
    bool crossvalidate;



 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
    TrainCtcParallelClass(const NnetCtcUpdateOptions *opts,
    		LmModelSync *model_sync,
			std::string	model_filename,
			std::string	target_model_filename,
			std::string targets_rspecifier,
			ExamplesRepository *repository,
			Nnet *nnet,
			NnetStats *stats):
				opts(opts),
				model_sync(model_sync),
				model_filename(model_filename),
				target_model_filename(target_model_filename),
				targets_rspecifier(targets_rspecifier),
				repository_(repository),
                host_nnet_(nnet),
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
        if (opts->use_gpu == "yes") {
	    if (parallel_opts->num_procs > 1) {
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
	    	//CuDevice::Instantiate().SelectGpuId(opts->use_gpu);
        CuDevice::Instantiate().SetCuAllocatorOptions(*opts->cuallocator_opts);
        }

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


	    if (opts->dropout_retention > 0.0) {
	    	nnet_transf.SetDropoutRetention(opts->dropout_retention);
	    	nnet.SetDropoutRetention(opts->dropout_retention);
	    }

	    if (crossvalidate) {
	      nnet_transf.SetDropoutRetention(1.0);
	      nnet.SetDropoutRetention(1.0);
	    }

	    Nnet si_nnet, softmax;
	    bool use_kld = (this->kld_scale > 0 && si_model_filename != "") ? true : false;
	    if (use_kld) {
	    	si_nnet.Read(si_model_filename);
	    }

	    Nnet frozen_nnet;
        bool use_frozen = opts->frozen_model_filename != "" ? true : false;
	    if (use_frozen) {
	    	frozen_nnet.Read(opts->frozen_model_filename);
	    }

	    RandomizerMask randomizer_mask(*rnd_opts);
	    MatrixRandomizer feature_randomizer(*rnd_opts);
	    PosteriorRandomizer targets_randomizer(*rnd_opts);
	    VectorRandomizer weights_randomizer(*rnd_opts);

	    Xent xent(*opts->loss_opts);
	    Mse mse(*opts->loss_opts);

	    CtcItf *ctc;
	    // Initialize CTC optimizer
	    if (opts->ctc_imp == "eesen")
	    	ctc = new Ctc;
	    else if (opts->ctc_imp == "warp") {
			ctc = new WarpCtc(opts->blank_label);
            // using activations directly: remove softmax, if present
            if (nnet.GetComponent(nnet.NumComponents()-1).GetType() == kaldi::nnet0::Component::kSoftmax) {
                //KALDI_LOG << "Removing softmax from the nnet " << model_filename;
                KALDI_LOG << "Removing softmax from the nnet " << model_filename << ", Appending logsoftmax";
                nnet.RemoveComponent(nnet.NumComponents()-1);
                nnet.AppendComponent(new LogSoftmax(nnet.OutputDim(),nnet.OutputDim()));
            } else {
                KALDI_LOG << "The nnet was without softmax " << model_filename;
            }    
        } else {
                KALDI_ERR << opts->ctc_imp << " ctc loss not implemented yet.";
        }

	    model_sync->Initialize(&nnet, this->thread_id_);

	    if (use_kld && opts->ctc_imp == "warp") {
            KALDI_LOG << "KLD model Appending the softmax ...";
	        softmax.AppendComponent(new Softmax(nnet.OutputDim(),nnet.OutputDim()));
        }

		CuMatrix<BaseFloat> feats_transf, nnet_in, nnet_out, nnet_diff, frozen_nnet_out,
							nnet_out_rearrange, nnet_diff_rearrange,
							*p_nnet_in, *p_nnet_out, *p_nnet_diff;
		CuMatrix<BaseFloat> si_nnet_out, soft_nnet_out;

		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;
		Matrix<BaseFloat> nnet_out_host, si_nnet_out_host, soft_nnet_out_host;


		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0, num_dump = 0;
		kaldi::int64 total_frames = 0;

		int32 num_stream = opts->num_stream;
		int32 frame_limit = opts->max_frames;
		int32 targets_delay = opts->targets_delay;
		int32 batch_size = opts->batch_size;
        int32 skip_frames = opts->skip_frames;

	    std::vector< Matrix<BaseFloat> > feats_utt(num_stream);  // Feature matrix of every utterance
	    std::vector< std::vector<int> > labels_utt(num_stream);  // Label vector of every utterance
	    std::vector<int> num_utt_frame_in, num_utt_frame_out;
        std::vector<int> new_utt_flags;
		Vector<BaseFloat> utt_flags;
		std::vector<int> idx, reidx;
		CuArray<MatrixIndexT> indexes;

	    Matrix<BaseFloat> feat_mat_host;
	    Vector<BaseFloat> frame_mask_host;
	    Posterior target;
	    std::vector<Posterior> targets_utt(num_stream);
        std::string utt;

	    CTCExample *ctc_example = NULL;
	    DNNExample *dnn_example = NULL;
	    Example		*example = NULL;
	    Timer time;
	    double time_now = 0;

		int32 cur_stream_num = 0, num_skip, in_rows, out_rows, 
              in_frames, out_frames, in_frames_pad, out_frames_pad;
		int32 feat_dim = use_frozen ? frozen_nnet.InputDim() : nnet.InputDim();
		//BaseFloat l2_term;
	    num_skip = opts->skip_inner ? skip_frames : 1;
        frame_limit *= num_skip;

	    while (num_stream) {

			int32 s = 0, max_frame_num = 0, cur_frames = 0;
			cur_stream_num = 0; num_frames = 0;
			num_utt_frame_in.clear();
			num_utt_frame_out.clear();

			if (NULL == example)
				example = repository_->ProvideExample();

			if (NULL == example)
				break;

			while (s < num_stream && cur_frames < frame_limit && NULL != example) {

				utt = example->utt;
				Matrix<BaseFloat> &mat = example->input_frames;

				if (objective_function == "xent"){
					dnn_example = dynamic_cast<DNNExample*>(example);
					targets_utt[s] = dnn_example->targets;
				} else if (objective_function == "ctc"){
					ctc_example = dynamic_cast<CTCExample*>(example);
					labels_utt[s] = ctc_example->targets;
				}

				if ((s+1)*mat.NumRows() > frame_limit || (s+1)*max_frame_num > frame_limit) break;
				if (max_frame_num < mat.NumRows()) max_frame_num = mat.NumRows();

				// forward the features through a feature-transform,
				nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

				feats_utt[s].Resize(feats_transf.NumRows(), feats_transf.NumCols(), kUndefined);
				feats_transf.CopyToMat(&feats_utt[s]);

		        	//feats_utt[s] = mat;
				in_rows = mat.NumRows();
				num_utt_frame_in.push_back(in_rows);
				num_frames += in_rows;

		        // inner skip frames
		        out_rows = in_rows/num_skip;
		        out_rows += in_rows%num_skip > 0 ? 1:0;
		        num_utt_frame_out.push_back(out_rows);

				s++;
				num_done++;
				cur_frames = max_frame_num * s;

				delete example;
				example = repository_->ProvideExample();
			}

			cur_stream_num = s;
            in_frames_pad = cur_stream_num * max_frame_num;
            out_frames_pad = cur_stream_num * ((max_frame_num+num_skip-1)/num_skip);
			new_utt_flags.resize(cur_stream_num, 1);

			if (this->objective_function == "xent") {
				target.resize(out_frames_pad);
				frame_mask_host.Resize(out_frames_pad, kSetZero);
			}

			if (opts->network_type == "lstm") {
				// Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
				feat_mat_host.Resize(in_frames_pad, feat_dim, kSetZero);

				for (int s = 0; s < cur_stream_num; s++) {
				  for (int r = 0; r < num_utt_frame_in[s]; r++) {
					  feat_mat_host.Row(r*cur_stream_num + s).CopyFromVec(feats_utt[s].Row(r));
				  }

				  //ce label
				  if (this->objective_function == "xent") {
					  for (int r = 0; r < num_utt_frame_out[s]; r++) {
						  if (r < targets_delay) {
							  frame_mask_host(r*cur_stream_num + s) = 0;
							  target[r*cur_stream_num + s] = targets_utt[s][r];
						  } else if (r < num_utt_frame_out[s] + targets_delay) {
							  frame_mask_host(r*cur_stream_num + s) = 1;
							  target[r*cur_stream_num + s] = targets_utt[s][r-targets_delay];
						  } else {
							  frame_mask_host(r*cur_stream_num + s) = 0;
							  int last = targets_utt[s].size()-1;
							  target[r*cur_stream_num + s] = targets_utt[s][last];
						  }
					  }
				  }
				}
			} else if (opts->network_type == "fsmn") {
				in_frames = 0, out_frames = 0;
				for (int s = 0; s < cur_stream_num; s++) {
					in_frames += num_utt_frame_in[s];
					out_frames += num_utt_frame_out[s];
				}

				feat_mat_host.Resize(in_frames, feat_dim, kUndefined);

				int k = 0, offset = 0;
				utt_flags.Resize(in_frames);
				for (int s = 0; s < cur_stream_num; s++) {
					for (int r = 0; r < num_utt_frame_in[s]; r++) {
                        utt_flags(k++) = num_done-(cur_stream_num-s);
                    }
                }

                k = 0, offset = 0;
                idx.resize(0);
				idx.resize(out_frames_pad, -1);
				reidx.resize(out_frames);
				for (int s = 0; s < cur_stream_num; s++) {
					for (int r = 0; r < num_utt_frame_out[s]; r++) {
						//utt_flags(k) = num_done-(cur_stream_num-s);
						idx[r*cur_stream_num + s] = k;
						reidx[k] = r*cur_stream_num + s;
						k++;
					}

					feat_mat_host.RowRange(offset, num_utt_frame_in[s]).CopyFromMat(feats_utt[s]);
					offset += num_utt_frame_in[s];
				}
			}

			// Set the original lengths of utterances before padding
	        if (opts->network_type == "lstm") {
			    // lstm
			    nnet.ResetLstmStreams(new_utt_flags, batch_size);
			    // bilstm
			    nnet.SetSeqLengths(num_utt_frame_out, batch_size);
			    if (use_frozen) {
			    	frozen_nnet.ResetLstmStreams(new_utt_flags, batch_size);
			    	frozen_nnet.SetSeqLengths(num_utt_frame_out, batch_size);
			    }
            } else if (opts->network_type == "fsmn") {
			    // fsmn
			    nnet.SetFlags(utt_flags);
			    if (use_frozen) {
			    	frozen_nnet.SetFlags(utt_flags);
			    }
            }

	        // report the speed
	        if (num_done % 5000 == 0) {
				time_now = time.Elapsed();
				KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
							<< time_now/60 << " min; processed " << total_frames/time_now
							<< " frames per second.";
	        }

	        nnet_in = feat_mat_host;
	        p_nnet_in = &nnet_in;
	        if (use_frozen) {
				frozen_nnet.Propagate(nnet_in, &frozen_nnet_out);
				p_nnet_in = &frozen_nnet_out;
	        }

	        // Propagation and CTC training
	        nnet.Propagate(*p_nnet_in, &nnet_out);
	        p_nnet_out = &nnet_out;

			if (use_kld) {
				// for streams with new utterance, history states need to be reset
			    if (opts->network_type == "lstm")
				    si_nnet.ResetLstmStreams(new_utt_flags, batch_size);
                else if (opts->network_type == "fsmn")
				    si_nnet.SetFlags(utt_flags);
				si_nnet.Propagate(*p_nnet_in, &si_nnet_out);
				CuMatrix<BaseFloat> *p_soft_nnet_out = &soft_nnet_out;
				if (opts->ctc_imp == "warp")
					softmax.Propagate(*p_nnet_out, &soft_nnet_out);
				else
					p_soft_nnet_out = p_nnet_out;

				// si nnet error
				si_nnet_out.AddMat(-1.0, *p_soft_nnet_out);
			}

            /*
			// check there's no nan/inf,
			if (!KALDI_ISFINITE(nnet_out.Sum())) {
			    KALDI_LOG << "NaN or inf found in final output nn-output for " << utt;
                KALDI_VLOG(1) << nnet.Info();
				monitor(&nnet, 0, num_frames);
                break;
			}
            */

	        if (opts->network_type == "fsmn") {
				indexes = idx;
				nnet_out_rearrange.Resize(out_frames_pad, nnet.OutputDim(), kSetZero, kStrideEqualNumCols);
				nnet_out_rearrange.CopyRows(nnet_out, indexes);
				p_nnet_out = &nnet_out_rearrange;
	        }

	        if (objective_function == "xent") {
	        	xent.Eval(frame_mask_host, *p_nnet_out, target, &nnet_diff);
	        } else if (objective_function == "ctc") {
				//ctc error
				ctc->EvalParallel(num_utt_frame_out, *p_nnet_out, labels_utt, &nnet_diff);
				// Error rates
				ctc->ErrorRateMSeq(num_utt_frame_out, *p_nnet_out, labels_utt);
	        } else
	        	KALDI_ERR<< "Unknown objective function code : " << objective_function;

            /*
			// check there's no nan/inf,
			if (!KALDI_ISFINITE(nnet_diff.Sum())) {
			    KALDI_LOG << "NaN or inf found in final nnet diff for " << utt;
                KALDI_VLOG(1) << nnet.Info();
				monitor(&nnet, 0, num_frames);
                break;
			}
            */

	        p_nnet_diff = &nnet_diff;
	        if (opts->network_type == "fsmn") {
				indexes = reidx;
				nnet_diff_rearrange.Resize(out_frames, nnet.OutputDim(), kUndefined);
				nnet_diff_rearrange.CopyRows(nnet_diff, indexes);
				p_nnet_diff = &nnet_diff_rearrange;

				//l2_term = 0;
				if (opts->l2_regularize > 0.0) {
					//l2_term += -0.5 * opts->l2_regularize * TraceMatMat(nnet_out, nnet_out, kTrans);
					//p_nnet_diff->AddMat(opts->l2_regularize, nnet_out);
				}
	        }

	        if (use_kld) {
	        	p_nnet_diff->Scale(1.0-kld_scale);
	        	p_nnet_diff->AddMat(-kld_scale, si_nnet_out);
	        }

			// backward pass
			if (!crossvalidate) {
				// backpropagate
				nnet.Backpropagate(*p_nnet_diff, NULL, true);
				update_frames += num_frames;
				if ((parallel_opts->num_threads > 1 || parallel_opts->num_procs > 1) &&
						update_frames + num_frames > parallel_opts->merge_size && !model_sync->isLastMerge())
				{
					// upload current model
					model_sync->GetWeight(&nnet, this->thread_id_, this->thread_id_);

					// model merge
					model_sync->ThreadSync(this->thread_id_, 1);

					// download last model
					if (!model_sync->isLastMerge())
					{
						model_sync->SetWeight(&nnet, this->thread_id_);

						nnet.ResetGradient();

						KALDI_VLOG(1) << "Thread " << thread_id_ << " merge NO."
										<< parallel_opts->num_merge - model_sync->leftMerge()
											<< " Current mergesize = " << update_frames << " frames.";
						update_frames = 0;
					}
				}
			}
			monitor(&nnet, total_frames, num_frames);
			// increase time counter
			total_frames += num_frames;

			// track training process
			if (!crossvalidate && this->thread_id_ == 0 && parallel_opts->myid == 0 && opts->dump_time > 0) {
				int num_procs = parallel_opts->num_procs > 1 ? parallel_opts->num_procs : 1;
				if ((total_frames*parallel_opts->num_threads*num_procs)/(3600*100*opts->dump_time) > num_dump) {
					char name[512];
					num_dump++;
					sprintf(name, "%s_%d_%ld", model_filename.c_str(), num_dump, total_frames);
					nnet.Write(string(name), true);
				}
			}

			fflush(stderr);
			fsync(fileno(stderr));
		}

		model_sync->LockStates();

		stats_->total_frames += total_frames;
		stats_->num_done += num_done;
		if (objective_function == "xent")
			stats_->xent.Add(&xent);
		else if (objective_function == "ctc")
			dynamic_cast<NnetCtcStats*>(stats_)->ctc.Add(ctc);
		else
			KALDI_ERR<< "Unknown objective function code : " << objective_function;

		model_sync->UnlockStates();

		//last merge
		if (!crossvalidate) {
			if (parallel_opts->num_threads > 1 || parallel_opts->num_procs > 1) {
				// upload current model
				model_sync->GetWeight(&nnet, this->thread_id_, this->thread_id_);

				// last model merge
				model_sync->ThreadSync(this->thread_id_, 0);

				// download last model
				model_sync->SetWeight(&nnet, this->thread_id_);

				KALDI_VLOG(1) << "Thread " << thread_id_ << " merge NO."
								<< parallel_opts->num_merge - model_sync->leftMerge()
									<< " Current mergesize = " << update_frames << " frames.";
			} else if (parallel_opts->num_threads == 1) {
                // upload current model
                model_sync->GetWeight(&nnet, this->thread_id_);
            }

			if (this->thread_id_ == 0 && parallel_opts->myid == 0) {
				KALDI_VLOG(1) << "Last thread upload model to host.";
                if (opts->ctc_imp == "warp") {
                    //add back the softmax
                    KALDI_LOG << "Removing logsoftmax, Appending the softmax " << target_model_filename;
                    nnet.RemoveComponent(nnet.NumComponents()-1);
                    nnet.AppendComponent(new Softmax(nnet.OutputDim(),nnet.OutputDim()));
                    //KALDI_LOG << "Appending the softmax " << target_model_filename;
                    //nnet.AppendComponent(new Softmax(nnet.OutputDim(),nnet.OutputDim()));
                }
				nnet.Write(target_model_filename, opts->binary);
			}
		}

	}

};


void AmCtcUpdateParallel(const NnetCtcUpdateOptions *opts,
		std::string	model_filename,
		std::string target_model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetCtcStats *stats)
{
		ExamplesRepository repository;
		LmModelSync model_sync(nnet, opts->parallel_opts);

		TrainCtcParallelClass c(opts, &model_sync,
								model_filename, target_model_filename, targets_rspecifier,
								&repository, nnet, stats);


	  {

		    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		    RandomAccessInt32VectorReader targets_reader(targets_rspecifier);
		    RandomAccessBaseFloatMatrixReader si_feature_reader(opts->si_feature_rspecifier);
	    	RandomAccessTokenReader *spec_aug_reader = NULL;
			std::string spec_aug_rspecifier = "";
    		if (opts->spec_aug_filename != "") {
    			std::stringstream ss;
    			ss << "ark,t:" << opts->spec_aug_filename;
    			spec_aug_rspecifier = ss.str();
				spec_aug_reader = new RandomAccessTokenReader(spec_aug_rspecifier);
    		}

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainCtcParallelClass> mc(opts->parallel_opts->num_threads, c);

		// prepare sample
	    Example *example;
	    std::vector<Example*> examples;
	    std::vector<int> sweep_frames, loop_frames;
		if (!kaldi::SplitStringToIntegers(opts->sweep_frames_str, ":", false, &sweep_frames))
			KALDI_ERR << "Invalid sweep-frames string " << opts->sweep_frames_str;
		for (int i = 0; i < sweep_frames.size(); i++) {
			if (sweep_frames[i] >= opts->skip_frames)
				KALDI_ERR << "invalid sweep frames indexes";
		}

		int nframes = sweep_frames.size();
		int idx = 0;
		loop_frames = sweep_frames;
		// loop sweep skip frames
	    for (; !feature_reader.Done(); feature_reader.Next()) {
	    	if (!opts->sweep_loop) {
	    		loop_frames.resize(1);
	    		loop_frames[0] = sweep_frames[idx];
	    		idx = (idx+1)%nframes;
	    	}

	    	example = new CTCExample(&feature_reader, &si_feature_reader, spec_aug_reader,
					&targets_reader, &model_sync, stats, opts);
            example->SetSweepFrames(loop_frames, opts->skip_inner);
	    	if (example->PrepareData(examples)) {
	    		for (int i = 0; i < examples.size(); i++) {
	    			repository.AcceptExample(examples[i]);
	    		}
	    		if (examples[0] != example)
	    			delete example;
	    	}
	    	else
	    		delete example;
	    }
	    repository.ExamplesDone();
	  }

}

void AmCEUpdateParallel(const NnetCtcUpdateOptions *opts,
		std::string	model_filename,
		std::string target_model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetStats *stats)
{
		ExamplesRepository repository;
		LmModelSync model_sync(nnet, opts->parallel_opts);

		TrainCtcParallelClass c(opts, &model_sync,
								model_filename, target_model_filename, targets_rspecifier,
								&repository, nnet, stats);


	  {

		    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
			RandomAccessBaseFloatMatrixReader si_feature_reader(opts->si_feature_rspecifier);
		    RandomAccessBaseFloatVectorReader weights_reader;
			RandomAccessPosteriorReader targets_reader(targets_rspecifier);
			std::string spec_aug_rspecifier = "";
	    	RandomAccessTokenReader *spec_aug_reader = NULL;
    		if (opts->spec_aug_filename != "") {
    			std::stringstream ss;
    			ss << "ark,t:" << opts->spec_aug_filename;
    			spec_aug_rspecifier = ss.str();
				spec_aug_reader = new RandomAccessTokenReader(spec_aug_rspecifier);
    		}

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainCtcParallelClass> mc(opts->parallel_opts->num_threads, c);


		// prepare sample
	    Example *example;
	    std::vector<Example*> examples;
	    std::vector<int> sweep_frames, loop_frames;
		if (!kaldi::SplitStringToIntegers(opts->sweep_frames_str, ":", false, &sweep_frames))
			KALDI_ERR << "Invalid sweep-frames string " << opts->sweep_frames_str;
		for (int i = 0; i < sweep_frames.size(); i++) {
			if (sweep_frames[i] >= opts->skip_frames)
				KALDI_ERR << "invalid sweep frames indexes";
		}

		int nframes = sweep_frames.size();
		int idx = 0;
		loop_frames = sweep_frames;
		// loop sweep skip frames
	    for (; !feature_reader.Done(); feature_reader.Next()) {
	    	if (!opts->sweep_loop) {
	    		loop_frames.resize(1);
	    		loop_frames[0] = sweep_frames[idx];
	    		idx = (idx+1)%nframes;
	    	}

	    	example = new DNNExample(&feature_reader, &si_feature_reader, spec_aug_reader,
					&targets_reader, &weights_reader, &model_sync, stats, opts);
            example->SetSweepFrames(loop_frames, opts->skip_inner);
	    	if (example->PrepareData(examples)) {
	    		for (int i = 0; i < examples.size(); i++) {
	    			repository.AcceptExample(examples[i]);
	    		}
	    		if (examples[0] != example)
	    			delete example;
	    	}
	    	else
	    		delete example;
	    }
	    repository.ExamplesDone();
	  }

}

} // namespace lm
} // namespace kaldi


