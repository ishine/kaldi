// lm/lm-compute-lstm-parallel.cc

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

#include "lm/lm-compute-lstm-parallel.h"

namespace kaldi {
namespace lm {

class TrainLstmlmParallelClass: public MultiThreadable {

	typedef nnet0::NnetTrainOptions NnetTrainOptions;
	typedef nnet0::NnetDataRandomizerOptions NnetDataRandomizerOptions;
	typedef nnet0::NnetParallelOptions NnetParallelOptions;
	typedef nnet0::ExamplesRepository  ExamplesRepository;
	typedef nnet0::ClassAffineTransform ClassAffineTransform;
	typedef nnet0::WordVectorTransform WordVectorTransform;
	typedef nnet0::CBSoftmax CBSoftmax;
	typedef nnet0::Nnet nnet;
	typedef nnet0::Component Component;
	typedef nnet0::LmNnetExample LmNnetExample;


private:
    const LstmlmUpdateOptions *opts;
    LmModelSync *model_sync;

	std::string feature_transform,
				model_filename,
                target_model_filename,
				classboundary_file,
				si_model_filename,
				zt_mean_filename;

	ExamplesRepository *repository_;
    LmStats *stats_;

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
    TrainLstmlmParallelClass(const LstmlmUpdateOptions *opts,
    		LmModelSync *model_sync,
			std::string	model_filename,
            std::string target_model_filename,
			ExamplesRepository *repository,
			Nnet *nnet,
			LmStats *stats):
				opts(opts),
				model_sync(model_sync),
				model_filename(model_filename),
                target_model_filename(target_model_filename),
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
				zt_mean_filename = opts->zt_mean_filename;

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
	    else {
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

	    // using activations directly: remove cbsoftmax, if use constant class zt
	    if (zt_mean_filename != "" && crossvalidate) {
	    	if (nnet.GetComponent(nnet.NumComponents()-1).GetType() == kaldi::nnet0::Component::kCBSoftmax) {
	    		KALDI_LOG << "Removing cbsoftmax from the nnet " << model_filename;
	    		nnet.RemoveComponent(nnet.NumComponents()-1);
			} else {
			  KALDI_LOG << "The nnet was without cbsoftmax " << model_filename;
			}
	    }

		TrainlmUtil util;
	    ClassAffineTransform *class_affine = NULL;
	    WordVectorTransform *word_transf = NULL;
	    CBSoftmax *cb_softmax = NULL;
	    for (int32 c = 0; c < nnet.NumComponents(); c++) {
	    	if (nnet.GetComponent(c).GetType() == Component::kClassAffineTransform)
	    		class_affine = &(dynamic_cast<ClassAffineTransform&>(nnet.GetComponent(c)));
	    	else if (nnet.GetComponent(c).GetType() == Component::kWordVectorTransform)
	    		word_transf = &(dynamic_cast<WordVectorTransform&>(nnet.GetComponent(c)));
	    	else if (nnet.GetComponent(c).GetType() == Component::kCBSoftmax)
	    		cb_softmax = &(dynamic_cast<CBSoftmax&>(nnet.GetComponent(c)));
	    }

	    if (classboundary_file != "") {
		    Input in;
		    Vector<BaseFloat> classinfo;
		    in.OpenTextMode(classboundary_file);
		    classinfo.Read(in.Stream(), false);
		    in.Close();
		    util.SetClassBoundary(classinfo, class_boundary_, word2class_);
	    }

	    Vector<BaseFloat> class_zt;
		if (zt_mean_filename != "" && crossvalidate) {
			Input in;
			in.OpenTextMode(zt_mean_filename);
			class_zt.Read(in.Stream(), false);
			in.Close();
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
	    bool use_kld = (this->kld_scale > 0 && si_model_filename != "") ? true : false;
	    if (use_kld)
	    	si_nnet.Read(si_model_filename);

	    model_sync->Initialize(&nnet, this->thread_id_);

	    nnet0::CBXent cbxent;
	    nnet0::Xent xent(*opts->loss_opts);
	    nnet0::Mse mse(*opts->loss_opts);

        if (NULL != class_affine) {
	        class_affine->SetClassBoundary(class_boundary_);
	        cbxent.SetClassBoundary(class_boundary_);
	        cbxent.SetVarPenalty(opts->var_penalty);
	        if (crossvalidate)
	        	cbxent.SetConstClassZt(class_zt);
        }

        if (NULL != cb_softmax) {
        	cb_softmax->SetClassBoundary(class_boundary_);
        	cbxent.SetZt(cb_softmax->GetZt(), cb_softmax->GetZtPatches());
        }

		CuMatrix<BaseFloat> feats_transf, nnet_out, nnet_diff;
		Matrix<BaseFloat> nnet_out_h, nnet_diff_h;

		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0, num_dump = 0;
		kaldi::int64 total_frames = 0;

		int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;
		int32 targets_delay = opts->targets_delay;

	    //  book-keeping for multi-streams
	    std::vector<std::string> keys(num_stream);
	    std::vector<std::vector<int32> > feats(num_stream);
	    std::vector<Posterior> targets(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 0);

	    // bptt batch buffer
	    //int32 feat_dim = nnet.InputDim();
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

	    LmNnetExample *example;
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
	            while ((example = dynamic_cast<LmNnetExample*>(repository_->ProvideExample())) != NULL)
	            {
	            	// checks ok, put the data in the buffers,
	            	keys[s] = example->utt;
	            	feats[s] = example->input_wordids;

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

	                curt[s]++;
	            }
	        }

			num_frames = feat.Dim();
			// report the speed
			if (num_done % 5000 == 0) {
			    time_now = time.Elapsed();
			    KALDI_LOG << "After " << num_done << " utterances: time elapsed = "
							<< time_now/60 << " min; processed " << total_frames/time_now
							<< " frames per second.";
			}

	        // for streams with new utterance, history states need to be reset
	        nnet.ResetLstmStreams(new_utt_flags);

			if (NULL != class_affine) {
				// sort output class id
				util.SortUpdateClass(target, sorted_target, sortedclass_target,
						sortedclass_target_index, sortedclass_target_reindex, frame_mask, sorted_frame_mask, word2class_);
				class_affine->SetUpdateClassId(sortedclass_target, sortedclass_target_index, sortedclass_target_reindex);
			}

			if (NULL != cb_softmax) {
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
	        if (objective_function == "xent") {
	        	xent.Eval(frame_mask, nnet_out, target, &nnet_diff);
	        } else if (objective_function == "cbxent") {
	        	cbxent.Eval(sorted_frame_mask, nnet_out, sorted_target, &nnet_diff);
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

		if (objective_function == "xent") {
			//KALDI_LOG << xent.Report();
			stats_->xent.Add(&xent);
		} else if (objective_function == "cbxent") {
			//KALDI_LOG << xent.Report();
			stats_->cbxent.Add(&cbxent);
		} else if (objective_function == "mse") {
			//KALDI_LOG << mse.Report();
			stats_->mse.Add(&mse);
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

			if (this->thread_id_ == 0 && parallel_opts->myid == 0) {
				KALDI_VLOG(1) << "Last thread upload model to host.";
				nnet.Write(target_model_filename, opts->binary);
			}
		}
	}

};

void TrainlmUtil::SortUpdateClass(const std::vector<int32>& update_id, std::vector<int32>& sorted_id,
		std::vector<int32>& sortedclass_id, std::vector<int32>& sortedclass_id_index, std::vector<int32>& sortedclass_id_reindex,
			const Vector<BaseFloat>& frame_mask, Vector<BaseFloat>& sorted_frame_mask, const std::vector<int32> &word2class)
{
	int size = update_id.size();
	std::vector<Word> words(size);

	for (int i = 0; i < size; i++)
	{
		words[i].idx = i;
		words[i].wordid = update_id[i];
		words[i].classid = word2class[update_id[i]];
	}

	std::sort(words.begin(), words.end(), TrainlmUtil::compare_classid);

	sorted_id.resize(size);
	sortedclass_id.resize(size);
	sortedclass_id_index.resize(size);
	sortedclass_id_reindex.resize(size);

	for (int i = 0; i < size; i++)
	{
		sorted_id[i] = words[i].wordid;
		sortedclass_id[i] = words[i].classid;
		sortedclass_id_index[i] = words[i].idx;
		sortedclass_id_reindex[words[i].idx] = i;
		sorted_frame_mask(i) = frame_mask(words[i].idx);
	}
}

void TrainlmUtil::SortUpdateWord(const Vector<BaseFloat>& update_id,
		std::vector<int32>& sortedword_id, std::vector<int32>& sortedword_id_index)
{
	int size = update_id.Dim();
	std::vector<Word> words(size);

	for (int i = 0; i < size; i++)
	{
		words[i].idx = i;
		words[i].wordid = (int32)update_id(i);
		words[i].classid = (int32)update_id(i);
	}

	std::sort(words.begin(), words.end(), TrainlmUtil::compare_wordid);
	sortedword_id.resize(size);
	sortedword_id_index.resize(size);

	for (int i = 0; i < size; i++)
	{
		sortedword_id[i] = words[i].wordid;
		sortedword_id_index[i] = words[i].idx;
	}
}

void TrainlmUtil::SetClassBoundary(const Vector<BaseFloat>& classinfo,
		std::vector<int32> &class_boundary, std::vector<int32> &word2class)
{
	class_boundary.resize(classinfo.Dim());
	int32 num_class = class_boundary.size()-1;
    for (int i = 0; i < classinfo.Dim(); i++)
    	class_boundary[i] = classinfo(i);
	int i,j = 0;
	word2class.resize(class_boundary[num_class]);
	for (i = 0; i < class_boundary[num_class]; i++)
	{
		if (i>=class_boundary[j] && i<class_boundary[j+1])
			word2class[i] = j;
		else
			word2class[i] = ++j;
	}
}

void LstmlmUpdateParallel(const LstmlmUpdateOptions *opts,
		std::string	model_filename,
        std::string target_model_filename,
		std::string feature_rspecifier,
		Nnet *nnet,
		LmStats *stats)
{
		nnet0::ExamplesRepository repository(128*30);
		LmModelSync model_sync(nnet, opts->parallel_opts);

		TrainLstmlmParallelClass c(opts, &model_sync,
								model_filename,
                                target_model_filename,
								&repository, nnet, stats);


	  {

		SequentialInt32VectorReader feature_reader(feature_rspecifier);

	    // The initialization of the following class spawns the threads that
	    // process the examples.  They get re-joined in its destructor.
	    MultiThreader<TrainLstmlmParallelClass> mc(opts->parallel_opts->num_threads, c);
	    nnet0::NnetExample *example;
	    std::vector<nnet0::NnetExample*> examples;
	    for (; !feature_reader.Done(); feature_reader.Next()) {
	    	example = new nnet0::LmNnetExample(&feature_reader, opts);
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


