// lm/slu-compute-lstm-parallel.cc

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
#include <algorithm>
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-semaphore.h"
#include "util/kaldi-mutex.h"
#include "util/kaldi-thread.h"

#include "lat/kaldi-lattice.h"

#include "cudamatrix/cu-device.h"
#include "base/kaldi-types.h"


#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-affine-preconditioned-transform.h"
#include "nnet0/nnet-model-merge-function.h"
#include "nnet0/nnet-activation.h"
#include "nnet0/nnet-example.h"
#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-class-affine-transform.h"
#include "nnet0/nnet-word-vector-transform.h"
#include "nnet0/nnet-parallel-component-multitask.h"

#include "lm/slu-compute-lstm-parallel.h"
#include "lm/lm-compute-lstm-parallel.h"

namespace kaldi {
namespace lm {

class TrainLstmSluParallelClass: public MultiThreadable {

	typedef nnet0::NnetTrainOptions NnetTrainOptions;
	typedef nnet0::NnetDataRandomizerOptions NnetDataRandomizerOptions;
	typedef nnet0::NnetParallelOptions NnetParallelOptions;
	typedef nnet0::ExamplesRepository  ExamplesRepository;
	typedef nnet0::ClassAffineTransform ClassAffineTransform;
	typedef nnet0::WordVectorTransform WordVectorTransform;
	typedef nnet0::ParallelComponentMultiTask ParallelComponentMultiTask;
	typedef nnet0::CBSoftmax CBSoftmax;
	typedef nnet0::Nnet nnet;
	typedef nnet0::Component Component;
	typedef nnet0::SluNnetExample SluNnetExample;


private:
    const SluLstmUpdateOptions *opts;
    LmModelSync *model_sync;

	std::string feature_transform,
				model_filename,
				classboundary_file,
				si_model_filename;

	ExamplesRepository *repository_;
    SluStats *stats_;

    const NnetTrainOptions *trn_opts;
    const NnetDataRandomizerOptions *rnd_opts;
    const NnetParallelOptions *parallel_opts;

    BaseFloat 	kld_scale;

    std::string use_gpu;
    std::string objective_function;
    int32 num_threads;
    bool crossvalidate;

    std::vector<int32> class_boundary_, word2class_;

 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
    TrainLstmSluParallelClass(const SluLstmUpdateOptions *opts,
    		LmModelSync *model_sync,
			std::string	model_filename,
			ExamplesRepository *repository,
			Nnet *nnet,
			SluStats *stats):
				opts(opts),
				model_sync(model_sync),
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
				classboundary_file = opts->class_boundary;
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

		TrainlmUtil util;
		std::unordered_map<std::string, std::pair<int32, int32> > output_offset;
	    ClassAffineTransform *class_affine = NULL;
	    WordVectorTransform *word_transf = NULL;
	    CBSoftmax *cb_softmax = NULL;
	    ParallelComponentMultiTask *parallel = NULL;
	    for (int32 c = 0; c < nnet.NumComponents(); c++)
	    {
	    	if (nnet.GetComponent(c).GetType() == Component::kParallelComponentMultiTask)
	    	{
	    		parallel = &(dynamic_cast<ParallelComponentMultiTask&>(nnet.GetComponent(c)));
	    		Component *com = parallel->GetComponent(Component::kClassAffineTransform);
	    		if (com != NULL) class_affine = dynamic_cast<ClassAffineTransform*>(com);
	    		com = parallel->GetComponent(Component::kCBSoftmax);
	    		if (com != NULL) cb_softmax = dynamic_cast<CBSoftmax*>(com);
	    		output_offset = parallel->GetOutputOffset();
	    	}
	    	else if (nnet.GetComponent(c).GetType() == Component::kClassAffineTransform)
	    		class_affine = &(dynamic_cast<ClassAffineTransform&>(nnet.GetComponent(c)));
	    	else if (nnet.GetComponent(c).GetType() == Component::kWordVectorTransform)
	    		word_transf = &(dynamic_cast<WordVectorTransform&>(nnet.GetComponent(c)));
	    	else if (nnet.GetComponent(c).GetType() == Component::kCBSoftmax)
	    		cb_softmax = &(dynamic_cast<CBSoftmax&>(nnet.GetComponent(c)));
	    }

	    if (classboundary_file != "")
	    {
		    Input in;
		    Vector<BaseFloat> classinfo;
		    in.OpenTextMode(classboundary_file);
		    classinfo.Read(in.Stream(), false);
		    in.Close();
		    util.SetClassBoundary(classinfo, class_boundary_, word2class_);
	    }

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

	    model_sync->Initialize(&nnet, this->thread_id_);

	    nnet0::CBXent cbxent;
        nnet0::Xent xent(*opts->loss_opts), slot_xent(*opts->loss_opts), intent_xent(*opts->loss_opts);
	    nnet0::Mse mse(*opts->loss_opts);

        if (NULL != class_affine)
        {
	        class_affine->SetClassBoundary(class_boundary_);
	        cb_softmax->SetClassBoundary(class_boundary_);
	        cbxent.SetClassBoundary(class_boundary_);
        }

		CuMatrix<BaseFloat> feats_transf, nnet_out, nnet_diff;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0;
		kaldi::int64 total_frames = 0;

		int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;
		int32 targets_delay = opts->targets_delay;
		int32 slot_delay = opts->slot_delay;
		int32 intent_delay = opts->intent_delay;

	    //  book-keeping for multi-streams
	    std::vector<std::string> keys(num_stream);
	    std::vector<std::vector<int32> > feats(num_stream);
	    std::vector<std::vector<int32> > slot_targets(num_stream);
	    std::vector<std::vector<int32> > intent_targets(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 0);

	    // bptt batch buffer
	    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
	    Vector<BaseFloat> sorted_frame_mask(batch_size * num_stream, kSetZero);
	    Vector<BaseFloat> feat(batch_size * num_stream, kSetZero);
        Matrix<BaseFloat> featmat(batch_size * num_stream, 1, kSetZero);
        CuMatrix<BaseFloat> words(batch_size * num_stream, 1, kSetZero);
	    std::vector<int32> target(batch_size * num_stream, kSetZero);
	    std::vector<int32> sorted_target(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedclass_target(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedclass_target_index(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedclass_target_reindex(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedword_id(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedword_id_index(batch_size * num_stream, kSetZero);

	    // slot and intent target and mask
	    std::vector<int32> slot_target(batch_size * num_stream, kSetZero);
	    std::vector<int32> intent_target(batch_size * num_stream, kSetZero);
	    Vector<BaseFloat> slot_mask(batch_size * num_stream, kSetZero);
	    Vector<BaseFloat> intent_mask(batch_size * num_stream, kSetZero);

	    // parallel network output and diff of output
	    nnet_diff.Resize(batch_size * num_stream, nnet.OutputDim(), kSetZero);
	    nnet_out.Resize(batch_size * num_stream, nnet.OutputDim(), kSetZero);

	    CuSubMatrix<BaseFloat> *lm_nnet_out = NULL, *slot_nnet_out = NULL, *intent_nnet_out = NULL;
	    CuSubMatrix<BaseFloat> *lm_nnet_diff = NULL, *slot_nnet_diff = NULL, *intent_nnet_diff = NULL;

	    std::pair<int32, int32> offset;
        std::unordered_map<std::string, BaseFloat> error_scale;
	    if (output_offset.find("lm") != output_offset.end()) {
	    	offset = output_offset["lm"];
	    	lm_nnet_out = new CuSubMatrix<BaseFloat>(nnet_out.ColRange(offset.first, offset.second));
	    	lm_nnet_diff = new CuSubMatrix<BaseFloat>(nnet_diff.ColRange(offset.first, offset.second));
            error_scale["lm"] = opts->lm_escale;
	    }

	    if (output_offset.find("slot") != output_offset.end()) {
	    	offset = output_offset["slot"];
	    	slot_nnet_out = new CuSubMatrix<BaseFloat>(nnet_out.ColRange(offset.first, offset.second));
	    	slot_nnet_diff = new CuSubMatrix<BaseFloat>(nnet_diff.ColRange(offset.first, offset.second));
            error_scale["slot"] = opts->slot_escale;
            if (opts->slot_rspecifier == "")
                KALDI_ERR << "Unspecified slot label reader.";
	    }

	    if (output_offset.find("intent") != output_offset.end()) {
	    	offset = output_offset["intent"];
			intent_nnet_out = new CuSubMatrix<BaseFloat>(nnet_out.ColRange(offset.first, offset.second));
			intent_nnet_diff = new CuSubMatrix<BaseFloat>(nnet_diff.ColRange(offset.first, offset.second));
            error_scale["intent"] = opts->intent_escale;
            if (opts->intent_rspecifier == "")
                KALDI_ERR << "Unspecified intent label reader.";
	    }

        if (parallel != NULL)
            parallel->SetErrorScale(error_scale);

	    SluNnetExample *example;
	    Timer time;
	    double time_now = 0;

	    while (num_stream) {
	        // loop over all streams, check if any stream reaches the end of its utterance,
	        // if any, feed the exhausted stream with a new utterance, update book-keeping infos
	        for (int s = 0; s < num_stream; s++) {
	            // this stream still has valid frames
	            if (curt[s] < lent[s] + targets_delay && curt[s] > 0) {
	                new_utt_flags[s] = 0;
	                continue;
	            }
			
	            // else, this stream exhausted, need new utterance
	            while ((example = dynamic_cast<SluNnetExample*>(repository_->ProvideExample())) != NULL)
	            {
	            	// checks ok, put the data in the buffers,
	            	keys[s] = example->utt;
	            	feats[s] = example->input_wordids;
	            	slot_targets[s] = example->input_slotids;
	            	intent_targets[s] = example->input_intentids;

	                num_done++;

	                curt[s] = 0;
	                lent[s] = feats[s].size() - 1;
	                new_utt_flags[s] = 1;  // a new utterance feeded to this stream
	                delete example;
	                break;
	            }
	        }

	        // we are done if all streams are exhausted
	        int done = 1;
	        for (int s = 0; s < num_stream; s++) {
	            if (curt[s]  < lent[s] + targets_delay) done = 0;  // this stream still contains valid data, not exhausted
	        }

	        if (done) break;

	        // fill a multi-stream bptt batch
	        // * frame_mask: 0 indicates padded frames, 1 indicates valid frames
	        // * target: padded to batch_size
	        // * feat: first shifted to achieve targets delay; then padded to batch_size
	        for (int t = 0; t < batch_size; t++) {
	            for (int s = 0; s < num_stream; s++) {
	                // frame_mask & targets padding
	                if (curt[s] < targets_delay) {
	                	frame_mask(t * num_stream + s) = 0;
	                	target[t * num_stream + s] = feats[s][0];
	                }
	                else if (curt[s] < lent[s] + targets_delay) {
	                    frame_mask(t * num_stream + s) = 1;
	                    target[t * num_stream + s] = feats[s][curt[s]-targets_delay+1];
	                } else {
	                    frame_mask(t * num_stream + s) = 0;
	                    target[t * num_stream + s] = feats[s][lent[s]-1];
	                }

	                // feat shifting & padding
	                if (curt[s] < lent[s]) {
	                    feat(t * num_stream + s) = feats[s][curt[s]];
	                } else {
	                    feat(t * num_stream + s) = feats[s][lent[s]-1];
	                }

	                // slot label
	                if (slot_nnet_out != NULL) {
						if (curt[s] < slot_delay) {
							slot_mask(t * num_stream + s) = 0;
							slot_target[t * num_stream + s] = slot_targets[s][0];
						}
						else if (curt[s] < lent[s] + slot_delay) {
							slot_mask(t * num_stream + s) = 1;
							slot_target[t * num_stream + s] = slot_targets[s][curt[s]-slot_delay];
						} else {
							slot_mask(t * num_stream + s) = 0;
							slot_target[t * num_stream + s] = slot_targets[s][lent[s]-1];
						}
	                }

	                // intent label
	                if (intent_nnet_out != NULL) {
						if (curt[s] < intent_delay) {
							intent_mask(t * num_stream + s) = 0;
							intent_target[t * num_stream + s] = intent_targets[s][0];
						}
						else if (curt[s] < lent[s] + intent_delay) {
							intent_mask(t * num_stream + s) = 1;
							intent_target[t * num_stream + s] = intent_targets[s][curt[s]-intent_delay];
						} else {
							intent_mask(t * num_stream + s) = 0;
							intent_target[t * num_stream + s] = intent_targets[s][lent[s]-1];
						}
	                }

	                curt[s]++;
	            }
	        }


			num_frames = feat.Dim();
			// report the speed
			if (num_done % 5000 == 0)
			{
			  time_now = time.Elapsed();
			  KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
							<< time_now/60 << " min; processed " << total_frames/time_now
							<< " frames per second.";
			}
    
	        // for streams with new utterance, history states need to be reset
	        nnet.ResetLstmStreams(new_utt_flags);

			if (NULL != class_affine)
			{
				// sort output class id
				util.SortUpdateClass(target, sorted_target, sortedclass_target,
						sortedclass_target_index, sortedclass_target_reindex, frame_mask, sorted_frame_mask, word2class_);
				class_affine->SetUpdateClassId(sortedclass_target, sortedclass_target_index, sortedclass_target_reindex);
				cb_softmax->SetUpdateClassId(sortedclass_target);
			}

	        // sort input word id
	        util.SortUpdateWord(feat, sortedword_id, sortedword_id_index);
	        word_transf->SetUpdateWordId(sortedword_id, sortedword_id_index);

	        // forward pass
	        featmat.CopyColFromVec(feat, 0);
            words.CopyFromMat(featmat);

	        nnet.Propagate(words, &nnet_out);

	        // evaluate objective function we've chosen
	        if (lm_nnet_out != NULL) {
	        if (objective_function == "xent") {
	        	xent.Eval(frame_mask, *lm_nnet_out, target, lm_nnet_diff);
	        } else if (objective_function == "cbxent") {
	        	cbxent.Eval(sorted_frame_mask, *lm_nnet_out, sorted_target, lm_nnet_diff);
	        } else {
	            KALDI_ERR << "Unknown objective function code : " << objective_function;
	        }
            }

	        if (slot_nnet_out != NULL)
	        	slot_xent.Eval(slot_mask, *slot_nnet_out, slot_target, slot_nnet_diff);
	        if (intent_nnet_out != NULL)
	        	intent_xent.Eval(intent_mask, *intent_nnet_out, intent_target, intent_nnet_diff);

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
		        fflush(stderr);
		        fsync(fileno(stderr));
		}

		model_sync->LockStates();

		stats_->total_frames += total_frames;
		stats_->num_done += num_done;

		if (objective_function == "xent"){
			//KALDI_LOG << xent.Report();
			stats_->xent.Add(&xent);
		 }else if (objective_function == "cbxent"){
			//KALDI_LOG << xent.Report();
			stats_->cbxent.Add(&cbxent);
		 }else if (objective_function == "mse"){
			//KALDI_LOG << mse.Report();
			stats_->mse.Add(&mse);
		 }else {
			 KALDI_ERR<< "Unknown objective function code : " << objective_function;
		 }

		 stats_->slot_xent.Add(&slot_xent);
		 stats_->intent_xent.Add(&intent_xent);

		 model_sync->UnlockStates();

		//last merge
		if (!crossvalidate){
			if (parallel_opts->num_threads > 1 || parallel_opts->num_procs > 1)
			{
				// upload current model
				model_sync->GetWeight(&nnet, this->thread_id_, this->thread_id_);

				// last model merge
				model_sync->ThreadSync(this->thread_id_, 0);

				// download last model
				model_sync->SetWeight(&nnet, this->thread_id_);

				KALDI_VLOG(1) << "Thread " << thread_id_ << " merge NO."
								<< parallel_opts->num_merge - model_sync->leftMerge()
									<< " Current mergesize = " << update_frames << " frames.";
			}

			if (this->thread_id_ == 0)
			{
				model_sync->CopyToHost(&nnet);
				KALDI_VLOG(1) << "Last thread upload model to host.";
			}
		}
	}

};

void SluLstmUpdateParallel(const SluLstmUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		Nnet *nnet,
		SluStats *stats)
{
		nnet0::ExamplesRepository repository(128*30);
		LmModelSync model_sync(nnet, opts->parallel_opts);

		TrainLstmSluParallelClass c(opts, &model_sync,
								model_filename,
								&repository, nnet, stats);


	  {

		SequentialInt32VectorReader feature_reader(feature_rspecifier);
		RandomAccessInt32VectorReader *slot_reader = NULL;
		RandomAccessInt32VectorReader *intent_reader = NULL;
		if (opts->slot_rspecifier != "")
			slot_reader = new RandomAccessInt32VectorReader(opts->slot_rspecifier);
		if (opts->intent_rspecifier != "")
			intent_reader = new RandomAccessInt32VectorReader(opts->intent_rspecifier);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainLstmSluParallelClass> mc(opts->parallel_opts->num_threads, c);
	    nnet0::NnetExample *example;
	    std::vector<nnet0::NnetExample*> examples;
	    for (; !feature_reader.Done(); feature_reader.Next()) {
	    	example = new nnet0::SluNnetExample(opts, &feature_reader, slot_reader, intent_reader);
	    	if (example->PrepareData(examples))
	    	{
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


} // namespace lm
} // namespace kaldi


