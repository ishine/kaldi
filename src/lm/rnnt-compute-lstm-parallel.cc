// lm/rnnt-compute-lstm-parallel.cc

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
#include "nnet0/nnet-multi-net-component.h"
#include "nnet0/nnet-rnnt-join-transform.h"
#include "nnet0/nnet-word-vector-transform.h"

#include "lm/lm-compute-lstm-parallel.h"
#include "lm/rnnt-compute-lstm-parallel.h"

namespace kaldi {
namespace lm {

class TrainRNNTLstmParallelClass: public MultiThreadable {

	typedef nnet0::NnetTrainOptions NnetTrainOptions;
	typedef nnet0::NnetDataRandomizerOptions NnetDataRandomizerOptions;
	typedef nnet0::NnetParallelOptions NnetParallelOptions;
	typedef nnet0::ExamplesRepository  ExamplesRepository;
	typedef nnet0::Nnet nnet;
	typedef nnet0::Component Component;
    typedef nnet0::NnetExample NnetExample;
	typedef nnet0::RNNTNnetExample RNNTNnetExample;
	typedef nnet0::MultiNetComponent MultiNetComponent;
	typedef nnet0::RNNTJoinTransform RNNTJoinTransform;
    typedef nnet0::WordVectorTransform WordVectorTransform;

private:
    const RNNTLstmUpdateOptions *opts;
    LmModelSync *model_sync;

	std::string feature_transform,
				model_filename,
				si_model_filename;

	ExamplesRepository *repository_;
    Nnet *host_nnet_;
    RNNTStats *stats_;

    const NnetTrainOptions *trn_opts;
    const NnetDataRandomizerOptions *rnd_opts;
    const NnetParallelOptions *parallel_opts;

    BaseFloat 	kld_scale;
    std::string use_gpu;
    std::string objective_function;
    int32 num_threads;
    bool crossvalidate;

 public:
  // This constructor is only called for a temporary object
  // that we pass to the RunMultiThreaded function.
    TrainRNNTLstmParallelClass(const RNNTLstmUpdateOptions *opts,
    		LmModelSync *model_sync,
			std::string	model_filename,
			ExamplesRepository *repository,
			Nnet *nnet,
			RNNTStats *stats):
				opts(opts),
				model_sync(model_sync),
				model_filename(model_filename),
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
	void operator ()() {

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

	    // encoder predict join network
	    MultiNetComponent *multi_net = NULL;
	    RNNTJoinTransform *join_com = NULL;
        WordVectorTransform *word_transf = NULL;
	    for (int32 c = 0; c < nnet.NumComponents(); c++) {
	    	if (nnet.GetComponent(c).GetType() == Component::kMultiNetComponent)
	    		multi_net = &(dynamic_cast<MultiNetComponent&>(nnet.GetComponent(c)));
	    }
        if (multi_net == NULL)
            KALDI_ERR << "RNNT network not exist" ;

        TrainlmUtil util;
	    Nnet *am = multi_net->GetNestNnet("encoder");
	    Nnet *lm = multi_net->GetNestNnet("predict");
	    Nnet *join = multi_net->GetNestNnet("join");
	    if (am == NULL || lm == NULL || join == NULL)
	    	KALDI_ERR << "encoder, predict or join network not exist" ;

		for (int32 c = 0; c < join->NumComponents(); c++) {
			if (join->GetComponent(c).GetType() == Component::kRNNTJoinTransform)
				join_com = &(dynamic_cast<RNNTJoinTransform&>(join->GetComponent(c)));
		}

        for (int32 c = 0; c < lm->NumComponents(); c++) {
            if (lm->GetComponent(c).GetType() == Component::kWordVectorTransform)
                word_transf = &(dynamic_cast<WordVectorTransform&>(lm->GetComponent(c)));
        }   

        // using activations directly: remove softmax, if present
        if (join->NumComponents() > 0 && join->GetComponent(join->NumComponents()-1).GetType() == kaldi::nnet0::Component::kSoftmax) {
            KALDI_LOG << "Removing softmax from the nnet " << model_filename;
            join->RemoveComponent(join->NumComponents()-1);
        } else {
            KALDI_LOG << "The nnet was without softmax " << model_filename;
        }    

	    // speaker independent network
	    Nnet si_nnet;
	    if (this->kld_scale > 0 && si_model_filename != "")
	    	si_nnet.Read(si_model_filename);

	    model_sync->Initialize(&nnet, this->thread_id_);

	    nnet0::WarpRNNT rnnt;
	    rnntOptions &rnnt_opts = rnnt.GetOption();
	    rnnt_opts.batch_first = false;
	    rnnt_opts.blank_label = opts->blank_label;

	    CuMatrix<BaseFloat> feats_transf, words, join_in, join_out, nnet_diff, join_in_diff;
		//double t1, t2, t3, t4;
		int32 update_frames = 0, num_frames = 0, num_done = 0, num_dump = 0;
		kaldi::int64 total_frames = 0;

		int32 num_stream = opts->num_stream;
		int32 frame_limit = opts->max_frames;
		int32 targets_delay = opts->targets_delay;
		int32 am_truncated = opts->am_truncated;
		int32 lm_truncated = opts->lm_truncated;
        int32 skip_frames = opts->skip_frames;

	    std::vector<Matrix<BaseFloat> > feats_utt(num_stream);  // Feature matrix of every utterance
	    std::vector<std::vector<int> > labels_utt(num_stream);  // Label vector of every utterance
	    std::vector<int> num_utt_frame_in, num_utt_frame_out;
	    std::vector<int> num_utt_word_in;
	    std::vector<int> new_utt_flags;
	    std::vector<int> sortedword_id;
	    std::vector<int> sortedword_id_index;

	    // am
	    Matrix<BaseFloat> am_featmat;
	    Vector<BaseFloat> am_frame_mask;
	    // lm
	    Vector<BaseFloat> lm_featvec;
	    Vector<BaseFloat> lm_frame_mask;
	    Matrix<BaseFloat> lm_featmat;

		int32 cur_stream_num = 0, num_skip, in_rows, out_rows,
              in_frames_pad, out_frames_pad,
			  in_words_pad, out_words_pad,
              sos_id;
		int32 am_feat_dim, am_out_dim, lm_word_dim, lm_out_dim, max_out_dim;

		am_feat_dim = am->InputDim();
		am_out_dim = am->OutputDim();
		lm_word_dim = lm->InputDim();
		lm_out_dim = lm->OutputDim();
		max_out_dim = am_out_dim > lm_out_dim ? am_out_dim : lm_out_dim;
	    num_skip = opts->skip_inner ? skip_frames : 1;
        frame_limit *= num_skip;
        sos_id = opts->sos_id;

        std::string utt;
        RNNTNnetExample *rnnt_example = NULL;
	    NnetExample		*example = NULL;
	    Timer time;
	    double time_now = 0;

	    while (num_stream) {

			int s = 0, max_frame_num = 0, max_words_num = 0, cur_frames = 0;
			cur_stream_num = 0;
			num_frames = 0;
			num_utt_frame_in.clear();
			num_utt_frame_out.clear();
			num_utt_word_in.clear();

			if (NULL == example)
				example = repository_->ProvideExample();

			if (NULL == example)
				break;

			while (s < num_stream && cur_frames < frame_limit && NULL != example) {
				utt = example->utt;
				Matrix<BaseFloat> &mat = example->input_frames;

				rnnt_example = dynamic_cast<RNNTNnetExample*>(example);
				labels_utt[s] = rnnt_example->input_wordids;

				if ((s+1)*mat.NumRows() > frame_limit || (s+1)*max_frame_num > frame_limit) break;
				if (max_frame_num < mat.NumRows()) max_frame_num = mat.NumRows();
				if (max_words_num < labels_utt[s].size()+1) max_words_num = labels_utt[s].size()+1;

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

				// lm feature
				num_utt_word_in.push_back(labels_utt[s].size()+1);

				s++;
				num_done++;
				cur_frames = max_frame_num * s;

				delete example;
				example = repository_->ProvideExample();
			}

			// am model padding
			targets_delay *=  num_skip;
			cur_stream_num = s;
			in_frames_pad = cur_stream_num * max_frame_num;
			out_frames_pad = cur_stream_num * ((max_frame_num+num_skip-1)/num_skip);
			new_utt_flags.resize(cur_stream_num, 1);
			rnnt_opts.maxT = (max_frame_num+num_skip-1)/num_skip;
			rnnt_opts.maxU = max_words_num;

			// Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
			am_featmat.Resize(in_frames_pad, am_feat_dim, kSetZero);
			for (int s = 0; s < cur_stream_num; s++) {
				for (int r = 0; r < num_utt_frame_in[s]; r++) {
				  if (r + targets_delay < num_utt_frame_in[s]) {
					  am_featmat.Row(r*cur_stream_num + s).CopyFromVec(feats_utt[s].Row(r+targets_delay));
				  } else{
					  int last = (num_utt_frame_in[s]-1); // frame_num_utt[s]-1
					  am_featmat.Row(r*cur_stream_num + s).CopyFromVec(feats_utt[s].Row(last));
				  }
				}
			}

			// language model padding
			in_words_pad = cur_stream_num * max_words_num;
			out_words_pad = cur_stream_num * max_words_num;
			lm_featvec.Resize(in_words_pad, kSetZero);
			for (int s = 0; s < cur_stream_num; s++) {
				for (int r = 0; r < num_utt_word_in[s]; r++) {
                  if (r == 0) { // sos input
                      lm_featvec(r*cur_stream_num + s) = sos_id;
				  } else if (r+targets_delay < num_utt_word_in[s]) {
					  lm_featvec(r*cur_stream_num + s) = labels_utt[s][r-1+targets_delay];
				  } else{
					  int last = (num_utt_word_in[s]-1);
					  lm_featvec(r*cur_stream_num + s) = labels_utt[s][last-1];
				  }
				}
			}

	        // report the speed
	        if (num_done % 5000 == 0) {
	          time_now = time.Elapsed();
	          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
	                        << time_now/60 << " min; processed " << total_frames/time_now
	                        << " frames per second.";
	        }

			// lstm
			am->ResetLstmStreams(new_utt_flags, am_truncated);
            am->SetSeqLengths(num_utt_frame_out, am_truncated);
			lm->ResetLstmStreams(new_utt_flags, lm_truncated);
			if (join_com != NULL) {
				join_com->SetRNNTStreamSize(num_utt_frame_out, num_utt_word_in, rnnt_opts.maxT, rnnt_opts.maxU);
			}

			join_in.Resize(out_frames_pad+out_words_pad, max_out_dim, kSetZero);
			CuSubMatrix<BaseFloat> am_out(join_in, 0, out_frames_pad, 0, am_out_dim);
			CuSubMatrix<BaseFloat> lm_out(join_in, out_frames_pad, out_words_pad, 0, lm_out_dim);
			join_in_diff.Resize(out_frames_pad+out_words_pad, max_out_dim, kUndefined);
			CuSubMatrix<BaseFloat> am_out_diff(join_in_diff, 0, out_frames_pad, 0, am_out_dim);
			CuSubMatrix<BaseFloat> lm_out_diff(join_in_diff, out_frames_pad, out_words_pad, 0, lm_out_dim);

	        // sort input word id
	        util.SortUpdateWord(lm_featvec, sortedword_id, sortedword_id_index);
	        word_transf->SetUpdateWordId(sortedword_id, sortedword_id_index);

			lm_featmat.Resize(lm_featvec.Dim(), lm_word_dim);
			lm_featmat.CopyColFromVec(lm_featvec, 0);

			// Propagation and CTC training
			am->Propagate(CuMatrix<BaseFloat>(am_featmat), &am_out);
			lm->Propagate(CuMatrix<BaseFloat>(lm_featmat), &lm_out);
			join->Propagate(join_in, &join_out);
			
			// evaluate objective function we've chosen
			if (objective_function == "rnnt") {
				rnnt.EvalParallel(num_utt_frame_out, join_out, labels_utt, &nnet_diff);
				//rnnt.ErrorRateMSeq(num_utt_frame_out, join_out, labels_utt);
			}

		    // backward pass
			if (!crossvalidate) {
				// backpropagate
				join->Backpropagate(nnet_diff, &join_in_diff, true);
				am->Backpropagate(am_out_diff, NULL, true);
				lm->Backpropagate(lm_out_diff, NULL, !opts->freeze_lm);

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
			if (!crossvalidate && this->thread_id_ == 0 && parallel_opts->myid == 0 && opts->dump_time > 0)
			{
				int num_procs = parallel_opts->num_procs > 1 ? parallel_opts->num_procs : 1;
				if ((total_frames*parallel_opts->num_threads*num_procs)/(3600*100*opts->dump_time) > num_dump)
				{
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

		if (objective_function == "rnnt"){
			//KALDI_LOG << xent.Report();
			stats_->rnnt.Add(&rnnt);
		 } else {
			 KALDI_ERR<< "Unknown objective function code : " << objective_function;
		 }

		model_sync->UnlockStates();

		//last merge
		if (!crossvalidate){
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
			} else if(parallel_opts->num_threads == 1) {
                // upload current model
                model_sync->GetWeight(&nnet, this->thread_id_);
            }

			if (this->thread_id_ == 0) {
				KALDI_VLOG(1) << "Last thread upload model to host.";
                // prevent copy local nnet component propagate buffer (e.g. lstm,cnn)
				// model_sync->CopyToHost(&nnet);
                host_nnet_->Read(model_filename);
				// download last model
				model_sync->SetWeight(host_nnet_, this->thread_id_);
			}
		}
	}

};


void RNNTLstmUpdateParallel(const RNNTLstmUpdateOptions *opts,
		std::string	model_filename,
		std::string feature_rspecifier,
		std::string targets_rspecifier,
		Nnet *nnet,
		RNNTStats *stats)
{
		nnet0::ExamplesRepository repository(128*30);
		LmModelSync model_sync(nnet, opts->parallel_opts);

		TrainRNNTLstmParallelClass c(opts, &model_sync,
								model_filename,
								&repository, nnet, stats);
		{

			SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
			RandomAccessInt32VectorReader targets_reader(targets_rspecifier);
			RandomAccessBaseFloatMatrixReader si_feature_reader(opts->si_feature_rspecifier);

			// The initialization of the following class spawns the threads that
			// process the examples.  They get re-joined in its destructor.
			MultiThreader<TrainRNNTLstmParallelClass> mc(opts->parallel_opts->num_threads, c);

			// prepare sample
			nnet0::NnetExample *example;
			std::vector<nnet0::NnetExample*> examples;
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

				example = new nnet0::RNNTNnetExample(&feature_reader, &si_feature_reader, &targets_reader, stats, opts);
				example->SetSweepFrames(loop_frames, opts->skip_inner);
				if (example->PrepareData(examples)) {
					for (int i = 0; i < examples.size(); i++) {
						repository.AcceptExample(examples[i]);
					}
					if (examples[0] != example)
						delete example;
				} else {
					delete example;
				}
			}
			repository.ExamplesDone();
		}
}

} // namespace lm
} // namespace kaldi


