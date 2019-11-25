// lm/am-compute-parallel.cc

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
#include "nnet0/nnet-example.h"
#include "nnet0/nnet-utils.h"

#include "lm/am-compute-parallel.h"

namespace kaldi {
namespace lm {

typedef nnet0::NnetTrainOptions NnetTrainOptions;
typedef nnet0::NnetDataRandomizerOptions NnetDataRandomizerOptions;
typedef nnet0::NnetParallelOptions NnetParallelOptions;
typedef nnet0::ExamplesRepository  ExamplesRepository;
typedef nnet0::NnetExample NnetExample;
typedef nnet0::DNNNnetExample DNNNnetExample;
typedef nnet0::Component Component;
typedef nnet0::RandomizerMask RandomizerMask;
typedef nnet0::MatrixRandomizer MatrixRandomizer;
typedef nnet0::VectorRandomizer VectorRandomizer;
typedef nnet0::PosteriorRandomizer PosteriorRandomizer;
typedef nnet0::Xent Xent;
typedef nnet0::Mse	Mse;
typedef nnet0::MultiTaskLoss MultiTaskLoss;

class TrainParallelClass: public MultiThreadable {

private:
    const NnetUpdateOptions *opts;
    LmModelSync *model_sync;

	std::string feature_transform,
				model_filename,
                target_model_filename,
				si_model_filename,
				targets_rspecifier;

	ExamplesRepository *repository_;
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
    TrainParallelClass(const NnetUpdateOptions *opts,
    		LmModelSync *model_sync,
			std::string	model_filename,
            std::string target_model_filename,
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
	void operator ()() {

		int thread_idx = this->thread_id_;
		model_sync->LockModel();
	    // Select the GPU
	#if HAVE_CUDA == 1
	    if (parallel_opts->num_procs > 1) {
	    	//int32 thread_idx = model_sync->GetThreadIdx();
	    	KALDI_LOG << "MyId: " << parallel_opts->myid << "  ThreadId: " << thread_idx;
	    	CuDevice::Instantiate().MPISelectGpu(model_sync->gpuinfo_, model_sync->win, thread_idx, this->num_threads);
	    	for (int i = 0; i< this->num_threads*parallel_opts->num_procs; i++) {
	    		KALDI_LOG << model_sync->gpuinfo_[i].hostname << "  myid: " << model_sync->gpuinfo_[i].myid
	    					<< "  gpuid: " << model_sync->gpuinfo_[i].gpuid;
	    	}
	    } else {
	    	CuDevice::Instantiate().SelectGpu();
	    }
        CuDevice::Instantiate().SetCuAllocatorOptions(*opts->cuallocator_opts);

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

        /*
	    int32 rank_in = 20, rank_out = 80, update_period = 4;
	   	    BaseFloat num_samples_history = 2000.0;
	   	    BaseFloat alpha = 4.0;
	    if (opts->use_psgd)
	     	nnet.SwitchToOnlinePreconditioning(rank_in, rank_out, update_period, num_samples_history, alpha);
        */

	    if (opts->dropout_retention > 0.0) {
	    	nnet_transf.SetDropoutRetention(opts->dropout_retention);
	    	nnet.SetDropoutRetention(opts->dropout_retention);
	    }

	    if (crossvalidate) {
	    	nnet_transf.SetDropoutRetention(1.0);
	    	nnet.SetDropoutRetention(1.0);
	    }

	    Nnet si_nnet;
	    bool use_kld = (this->kld_scale > 0 && si_model_filename != "") ? true : false;
	    if (use_kld)
	    	si_nnet.Read(si_model_filename);

	    Nnet frozen_nnet;
	    if (opts->frozen_model_filename != "") {
	    	frozen_nnet.Read(opts->frozen_model_filename);
	    }

	    model_sync->Initialize(&nnet, this->thread_id_);

	    // skip frames
	    int32 skip_frames = opts->skip_frames;
        //int in_skip = opts->skip_inner ? 1 : skip_frames;
        int out_skip = opts->skip_inner ? skip_frames : 1;
        int num_skip = opts->skip_inner ? skip_frames : 1;

	    NnetDataRandomizerOptions skip_rand_opts = *rnd_opts;
	    skip_rand_opts.minibatch_size = rnd_opts->minibatch_size*out_skip;

	    RandomizerMask randomizer_mask(*rnd_opts);
	    MatrixRandomizer feature_randomizer(skip_rand_opts);
	    PosteriorRandomizer targets_randomizer(*rnd_opts);
	    VectorRandomizer weights_randomizer(*rnd_opts);
	    VectorRandomizer flags_randomizer(skip_rand_opts);

	    Xent xent(*opts->loss_opts);
	    Mse mse(*opts->loss_opts);
	    MultiTaskLoss multitask(*opts->loss_opts);
	    if (0 == objective_function.compare(0, 9, "multitask")) {
			// objective_function contains something like :
			// 'multitask,xent,2456,1.0,mse,440,0.001'
			//
			// the meaning is following:
			// 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
			multitask.InitFromString(objective_function);
			stats_->multitask.InitFromString(objective_function);
		}

	    Timer time, mpi_time;

		CuMatrix<BaseFloat> feats, feats_transf, nnet_out, nnet_diff, frozen_nnet_out;
		CuMatrix<BaseFloat> si_nnet_out; // *p_si_nnet_out = NULL;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;
		Vector<BaseFloat> utt_flags;

		DNNNnetExample *example;

		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0, num_dump = 0;
		kaldi::int64 total_frames = 0;

		while (true) {
			while (!feature_randomizer.IsFull() && (example = dynamic_cast<DNNNnetExample*>(repository_->ProvideExample())) != NULL) {
				//time.Reset();
				std::string utt = example->utt;
				const Matrix<BaseFloat> &mat = example->input_frames;
				Posterior &targets = example->targets;
				Vector<BaseFloat> &weights = example->frames_weights;
				//t1 = time.Elapsed();
				//time.Reset();

		        // apply optional feature transform
		        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

		        // fsmn target delay
		        if (opts->targets_delay > 0) {
		        		int targets_delay = opts->targets_delay;
		        		CuMatrix<BaseFloat> tmp(feats_transf);
		        		int offset = opts->targets_delay*num_skip;
		        		int rows = feats_transf.NumRows();
		        		feats_transf.Resize(rows+offset, tmp.NumCols(), kUndefined);
		        		feats_transf.RowRange(0, rows).CopyFromMat(tmp);

		        		for (int i = 0; i < offset; i++)
		        			feats_transf.Row(rows+i).CopyFromVec(tmp.Row(rows-1));


		        		int tg_size = targets.size();
		        		Posterior tgt = targets;
		        		targets.resize(tg_size+targets_delay);
		        		Vector<BaseFloat> wt = weights;
		        		weights.Resize(tg_size+targets_delay, kUndefined);
		        		for (int i = 0; i < targets_delay; i++) {
		        			targets[i] = tgt[0];
		        			weights(i) = 0.0;
		        		}
		        		for (int i = targets_delay; i < tg_size+targets_delay; i++) {
		        			targets[i] = tgt[i-targets_delay];
		        			weights(i) = wt(i-targets_delay);
		        		}
		        }

		        // pass data to randomizers
		        KALDI_ASSERT(feats_transf.NumRows() == targets.size()*out_skip);
		        feature_randomizer.AddData(feats_transf);
		        targets_randomizer.AddData(targets);
		        weights_randomizer.AddData(weights);
		        //utt_flags.Resize(targets.size(), kSetZero);
		        utt_flags.Resize(feats_transf.NumRows(), kSetZero);
		        utt_flags.Set(BaseFloat(num_done));
		        flags_randomizer.AddData(utt_flags);
		        num_done++;

		        // report the speed
		        if (num_done % 5000 == 0) {
		          double time_now = time.Elapsed();
		          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
		                        << time_now/60 << " min; processed " << total_frames/time_now
		                        << " frames per second.";
		        }

		        // release the buffers we don't need anymore
		       	delete example;
			}

	        if (feature_randomizer.Done())
	        	break;

		      // randomize
      		if (!crossvalidate && opts->randomize) {
				const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
				feature_randomizer.Randomize(mask);
				targets_randomizer.Randomize(mask);
       			weights_randomizer.Randomize(mask);
      		}

	        // train with data from randomizers (using mini-batches)
			for (; !feature_randomizer.Done(); feature_randomizer.Next(),
											  targets_randomizer.Next(),
											  weights_randomizer.Next(),
											  flags_randomizer.Next()) {

				// get block of feature/target pairs
				const CuMatrixBase<BaseFloat>& nnet_in = feature_randomizer.Value();
				const Posterior& nnet_tgt = targets_randomizer.Value();
				const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();
				const Vector<BaseFloat>& flags = flags_randomizer.Value();
				num_frames = nnet_in.NumRows();

				const CuMatrixBase<BaseFloat> *p_nnet_in = &nnet_in;
				if (opts->frozen_model_filename != "") {
					frozen_nnet.SetFlags(flags);
					frozen_nnet.Propagate(nnet_in, &frozen_nnet_out);
					p_nnet_in = &frozen_nnet_out;
				}

				// fsmn
				nnet.SetFlags(flags);
				// forward pass
				nnet.Propagate(*p_nnet_in, &nnet_out);

				CuMatrix<BaseFloat> tgt_mat;
			    if (use_kld) {
				    si_nnet.SetFlags(flags);
			      	si_nnet.Propagate(*p_nnet_in, &si_nnet_out);
			      	//p_si_nnet_out = &si_nnet_out;
			        // convert posterior to matrix,
					PosteriorToMatrix(nnet_tgt, nnet.OutputDim(), &tgt_mat);
					tgt_mat.Scale(1-this->kld_scale);
					tgt_mat.AddMat(this->kld_scale, si_nnet_out);
			    }

				// evaluate objective function we've chosen
				if (objective_function == "xent") {
					// gradients re-scaled by weights in Eval,
					if (use_kld)
						xent.Eval(frm_weights, nnet_out, tgt_mat, &nnet_diff);
					else
						xent.Eval(frm_weights, nnet_out, nnet_tgt, &nnet_diff);
				} else if (objective_function == "mse") {
					// gradients re-scaled by weights in Eval,
					if (use_kld)
						mse.Eval(frm_weights, nnet_out, tgt_mat, &nnet_diff);
					else
						mse.Eval(frm_weights, nnet_out, nnet_tgt, &nnet_diff);
				} else if (0 == objective_function.compare(0, 9, "multitask")) {
			          // gradients re-scaled by weights in Eval,
					if (use_kld)
						multitask.Eval(frm_weights, nnet_out, tgt_mat, &nnet_diff);
					else
						multitask.Eval(frm_weights, nnet_out, nnet_tgt, &nnet_diff);
			    } else {
					KALDI_ERR<< "Unknown objective function code : " << objective_function;
				}

				// backward pass
				if (!crossvalidate) {
					// backpropagate
					nnet.Backpropagate(nnet_diff, NULL, true);
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

							// nnet.ResetGradient();

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
		}

		model_sync->LockStates();
		stats_->total_frames += total_frames;
		stats_->num_done += num_done;
		if (objective_function == "xent"){
			//KALDI_LOG << xent.Report();
			stats_->xent.Add(&xent);
		} else if (objective_function == "mse"){
			//KALDI_LOG << mse.Report();
			stats_->mse.Add(&mse);
		} else if (0 == objective_function.compare(0, 9, "multitask")) {
			stats_->multitask.Add(&multitask);
		} else {
			KALDI_ERR<< "Unknown objective function code : " << objective_function;
		}
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

			if (this->thread_id_ == 0) {
				KALDI_VLOG(1) << "Last thread upload model to host.";
				if(parallel_opts->myid == 0)
					nnet.Write(target_model_filename, opts->binary);
			}
		}
	}

};


void AmUpdateParallel(const NnetUpdateOptions *opts,
		std::string	model_filename,
        std::string target_model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		NnetStats *stats)
{
		ExamplesRepository repository(128*10);
		LmModelSync model_sync(nnet, opts->parallel_opts);

		TrainParallelClass c(opts, &model_sync,
								model_filename, 
                                target_model_filename,
                                targets_rspecifier,
								&repository, nnet, stats);

	  {

		    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		    RandomAccessBaseFloatMatrixReader si_feature_reader(opts->si_feature_rspecifier);
		    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
		    RandomAccessBaseFloatVectorReader weights_reader;

	    	RandomAccessTokenReader *spec_aug_reader = NULL;
			std::string spec_aug_rspecifier = "";
    		if (opts->spec_aug_filename != "") {
    			std::stringstream ss;
    			ss << "ark,t:" << opts->spec_aug_filename;
    			spec_aug_rspecifier = ss.str();
				spec_aug_reader = new RandomAccessTokenReader(spec_aug_rspecifier);
    		}

		    if (opts->frame_weights != "") 
		        weights_reader.Open(opts->frame_weights);

            if (opts->objective_function.compare(0, 9, "multitask") == 0)
                stats->multitask.InitFromString(opts->objective_function);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainParallelClass> m(opts->parallel_opts->num_threads, c);

	    // prepare sample
	    NnetExample *example;
		std::vector<NnetExample*> examples;
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

			example = new DNNNnetExample(&feature_reader, &si_feature_reader, spec_aug_reader,
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


