// lm/seqlabel-compute-lstm-parallel.cc

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
#include "nnet0/nnet-word-vector-transform.h"

#include "lm/seqlabel-compute-lstm-parallel.h"
#include "lm/lm-compute-lstm-parallel.h"

namespace kaldi {
namespace lm {

class SeqLabelLstmParallelClass: public MultiThreadable {

	typedef nnet0::NnetTrainOptions NnetTrainOptions;
	typedef nnet0::NnetDataRandomizerOptions NnetDataRandomizerOptions;
	typedef nnet0::NnetParallelOptions NnetParallelOptions;
	typedef nnet0::ExamplesRepository  ExamplesRepository;
	typedef nnet0::WordVectorTransform WordVectorTransform;
	typedef nnet0::Nnet nnet;
	typedef nnet0::Component Component;
	typedef nnet0::SeqLabelNnetExample SeqLabelNnetExample;


private:
    const SeqLabelLstmUpdateOptions *opts;
    LmModelSync *model_sync;

	std::string feature_transform,
				model_filename,
				classboundary_file,
				si_model_filename;

	ExamplesRepository *repository_;
    SeqLabelStats *stats_;

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
    SeqLabelLstmParallelClass(const SeqLabelLstmUpdateOptions *opts,
    		LmModelSync *model_sync,
			std::string	model_filename,
			ExamplesRepository *repository,
			Nnet *nnet,
			SeqLabelStats *stats):
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
	    WordVectorTransform *word_transf = NULL;
	    for (int32 c = 0; c < nnet.NumComponents(); c++)
	    {
	    	if (nnet.GetComponent(c).GetType() == Component::kWordVectorTransform)
	    		word_transf = &(dynamic_cast<WordVectorTransform&>(nnet.GetComponent(c)));
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
	    {
	    	si_nnet.Read(si_model_filename);
	    }

	    model_sync->Initialize(&nnet, this->thread_id_);

	    nnet0::Xent xent(*opts->loss_opts);

		CuMatrix<BaseFloat> feats_transf, nnet_out, nnet_diff;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0;
		kaldi::int64 total_frames = 0;

		int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;
		int32 targets_delay = opts->targets_delay;

	    //  book-keeping for multi-streams
	    std::vector<std::string> keys(num_stream);
	    std::vector<std::vector<int32> > feats(num_stream);
	    std::vector<std::vector<int32> > targets(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 0);

	    // bptt batch buffer
	    Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
	    Vector<BaseFloat> feat(batch_size * num_stream, kSetZero);
        Matrix<BaseFloat> featmat(batch_size * num_stream, 1, kSetZero);
        CuMatrix<BaseFloat> words(batch_size * num_stream, 1, kSetZero);
	    std::vector<int32> target(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedword_id(batch_size * num_stream, kSetZero);
	    std::vector<int32> sortedword_id_index(batch_size * num_stream, kSetZero);

	    SeqLabelNnetExample *example;
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
	            while ((example = dynamic_cast<SeqLabelNnetExample*>(repository_->ProvideExample())) != NULL)
	            {
	            	// checks ok, put the data in the buffers,
	            	keys[s] = example->utt;
	            	feats[s] = example->input_wordids;
	            	targets[s] = example->input_labelids;

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
	                    target[t * num_stream + s] = targets[s][curt[s]-targets_delay];
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

	        // sort input word id
	        util.SortUpdateWord(feat, sortedword_id, sortedword_id_index);
	        word_transf->SetUpdateWordId(sortedword_id, sortedword_id_index);

	        // forward pass
	        featmat.CopyColFromVec(feat, 0);
            words.CopyFromMat(featmat);

	        nnet.Propagate(words, &nnet_out);

	        // evaluate objective function we've chosen
	        if (objective_function == "xent") {
	        	xent.Eval(frame_mask, nnet_out, target, &nnet_diff);
	        } else {
	            KALDI_ERR << "Unknown objective function code : " << objective_function;
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
		 }else {
			 KALDI_ERR<< "Unknown objective function code : " << objective_function;
		 }

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


void SeqLabelLstmParallel(const SeqLabelLstmUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string label_rspecifier,
		Nnet *nnet,
		SeqLabelStats *stats)
{
		nnet0::ExamplesRepository repository(128*30);
		LmModelSync model_sync(nnet, opts->parallel_opts);

		SeqLabelLstmParallelClass c(opts, &model_sync,
								model_filename,
								&repository, nnet, stats);


	  {

		SequentialInt32VectorReader feature_reader(feature_rspecifier);
		RandomAccessInt32VectorReader label_reader(label_rspecifier);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<SeqLabelLstmParallelClass> mc(opts->parallel_opts->num_threads, c);
	    nnet0::NnetExample *example;
	    std::vector<nnet0::NnetExample*> examples;
	    for (; !feature_reader.Done(); feature_reader.Next()) {
	    	example = new nnet0::SeqLabelNnetExample(opts, &feature_reader, &label_reader);
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


