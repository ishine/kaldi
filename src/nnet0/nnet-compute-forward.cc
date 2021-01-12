// nnet0/nnet-compute-forward.cc

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

#include "lat/lattice-functions.h"
#include "util/kaldi-semaphore.h"
#include "util/kaldi-mutex.h"
#include "util/kaldi-thread.h"

#include "nnet0/nnet-example.h"
#include "nnet0/nnet-compute-forward.h"

namespace kaldi {
namespace nnet0 {

class NnetForwardParallelClass: public MultiThreadable {
private:
	const NnetForwardOptions *opts;
	std::string model_filename;
	ExamplesRepository *repository;
	BaseFloatMatrixWriter *feature_writer;
	Mutex *examples_mutex;
	NnetForwardStats *stats;

public:
	NnetForwardParallelClass(const NnetForwardOptions *opts,
							std::string model_filename,
							ExamplesRepository *repository,
							BaseFloatMatrixWriter *feature_writer,
							Mutex *examples_mutex,
							NnetForwardStats *stats):
								opts(opts),model_filename(model_filename),
								repository(repository),feature_writer(feature_writer),
								examples_mutex(examples_mutex), stats(stats)
	{

	}

    static bool cmp(std::pair<BaseFloat, int> x1, std::pair<BaseFloat, int> x2) {
        return x1.first > x2.first;
    }

	void TopKPerRow(Matrix<BaseFloat> &nnet_out, Matrix<BaseFloat> &out_topk, int32 topk) {
		int nr = nnet_out.NumRows(), nc = nnet_out.NumCols(), key;
		float logp, *row_data, *out_data;

		if (topk <= 0 || topk >= nc) {
			out_topk = nnet_out;
			return;
		}

		out_topk.Resize(nr, topk*2, kUndefined);
        std::vector<BaseFloat> row_buffer(nc);
        std::vector<std::pair<BaseFloat, int> > vec;
        vec.reserve(topk-1);
		for (int n = 0; n < nr; n++) {
			row_data = nnet_out.RowData(n);
			out_data = out_topk.RowData(n);
            memcpy(&row_buffer.front(), row_data, nc*sizeof(BaseFloat));
            std::nth_element(row_buffer.begin(), row_buffer.begin()+topk-1, row_buffer.end(), std::greater<BaseFloat>());
			key = 0;
			out_data[key] = nnet_out(n, 0);
			out_data[topk+key] = 0;
			key = 1;
			for (int k = 0; k < nc; k++) {
				logp = nnet_out(n, k);
				if (logp > row_buffer[topk-1] && key < topk) {
					out_data[key] = logp;
					out_data[topk+key] = k;
					key++;
				}
			}

            for (; key < topk; key++) {
                out_data[key] = kLogZeroFloat;
				out_data[topk+key] = -1;
            }

            // sort
            for (int i = 1; i < topk; i++)
                vec.push_back(std::make_pair(out_data[i], out_data[topk+i]));

            std::stable_sort(vec.begin(), vec.end(), cmp);
            for (int i = 1; i < topk; i++) {
                out_data[i] = vec[i-1].first;
                out_data[topk+i] = vec[i-1].second;
            }
            vec.clear();
                
		    //if (key != topk)
	        //	KALDI_WARN << "topk: " << topk << " actually: " << key;
		}
	}

	  // This does the main function of the class.
	void operator ()()
	{

		examples_mutex->Lock();
		// Select the GPU
		#if HAVE_CUDA == 1
			if (opts->use_gpu == "yes") {
		    	CuDevice::Instantiate().SelectGpu();
                CuDevice::Instantiate().SetCuAllocatorOptions(*opts->cuallocator_opts);
            }
		    //CuDevice::Instantiate().DisableCaching();
		#endif

		examples_mutex->Unlock();

		bool no_softmax = opts->no_softmax;
		std::string feature_transform = opts->feature_transform;
		bool apply_log = opts->apply_log;
		int32 time_shift = opts->time_shift;
		const PdfPriorOptions *prior_opts = opts->prior_opts;
		int32 num_stream = opts->num_stream;
		int32 batch_size = opts->batch_size;
		int32 skip_frames = opts->skip_frames;
		float blank_posterior_scale = opts->blank_posterior_scale;


		Nnet nnet_transf;

	    if (feature_transform != "") {
	      nnet_transf.Read(feature_transform);
	    }

	    Nnet nnet;
	    nnet.Read(model_filename);

	    // optionally remove softmax,
	    Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
	    if (no_softmax) {
	      if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
	        KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename;
	        nnet.RemoveComponent(nnet.NumComponents()-1);
	      } else {
	        KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
	      }
	    }

	    // avoid some bad option combinations,
	    if (apply_log && no_softmax) {
	    	KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
	    }

        /*
	    if (blank_posterior_scale >= 0 && prior_opts->class_frame_counts != "") {
	    	KALDI_ERR << "Cannot use both --blank-posterior-scale --class-frame-counts, use only one of the two!";
	    }
        */

	    // we will subtract log-priors later,
	    PdfPrior pdf_prior(*opts->prior_opts);

	    // disable dropout,
	    nnet_transf.SetDropoutRetention(1.0);
	    nnet.SetDropoutRetention(1.0);

        int in_skip = opts->skip_inner ? 1 : skip_frames,
        out_skip = opts->skip_inner ? skip_frames : 1;

	    CuMatrix<BaseFloat>  cufeat, feats_transf, nnet_out;

	    std::vector<std::string> keys(num_stream);
	    std::vector<Matrix<BaseFloat> > feats(num_stream);
	    std::vector<Posterior> targets(num_stream);
	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> frame_num_utt(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 0);
	    std::vector<int> valid_input_frames(num_stream, 0);
	    std::vector<int> utt_state_flags(num_stream, 0);

	    std::vector<Matrix<BaseFloat> > utt_nnet_out(num_stream);
	    std::vector<int> utt_curt(num_stream, 0);
	    std::vector<bool> utt_copied(num_stream, 0);

	    // bptt batch buffer
	    int32 feat_dim = nnet.InputDim();
	    int32 out_dim = nnet.OutputDim();
	    Matrix<BaseFloat> feat, nnet_out_host, utt_out_topk;
	    if (batch_size * num_stream > 0) {
			feat.Resize(out_skip * batch_size * num_stream, feat_dim, kSetZero);
			nnet_out_host.Resize(batch_size * num_stream, out_dim, kSetZero);
	    }


	    kaldi::int64 total_frames = 0;
	    int32 num_done = 0, num_frames;
	    Timer time;
	    double time_now = 0;


	    FeatureExample *example;

	    //num_stream=1 for lstm debug
	    if (num_stream >= 1)
	    while (1) {
			// loop over all streams, check if any stream reaches the end of its utterance,
			// if any, feed the exhausted stream with a new utterance, update book-keeping infos
			for (int s = 0; s < num_stream; s++) {
				// this stream still has valid frames
				if (curt[s] < lent[s] || utt_state_flags[s] == 1) {
					new_utt_flags[s] = 0;
					utt_state_flags[s] = 1; // utterance append
					continue;
				}

				if (utt_curt[s] > 0 && !utt_copied[s]) {
					TopKPerRow(utt_nnet_out[s], utt_out_topk, opts->topk);
					examples_mutex->Lock();
					//feature_writer->Write(keys[s], utt_nnet_out[s]);
					feature_writer->Write(keys[s], utt_out_topk);
					examples_mutex->Unlock();
					utt_copied[s] = true;
				}

				while ((example = dynamic_cast<FeatureExample*>(repository->ProvideExample())) != NULL) {
					std::string key = example->utt;
					Matrix<BaseFloat> &mat = example->input_frames;
					// forward the features through a feature-transform,
					nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

					num_done++;

					// checks ok, put the data in the buffers,
					keys[s] = key;
					feats[s].Resize(feats_transf.NumRows(), feats_transf.NumCols());
					feats_transf.CopyToMat(&feats[s]);
					//feats[s] = mat;
					curt[s] = 0;
					lent[s] = feats[s].NumRows();
					new_utt_flags[s] = 1;  // a new utterance feeded to this stream
					utt_state_flags[s] = 0; // utterance start

					//frame_num_utt[s] = (lent[s]+skip_frames-1)/skip_frames;
					frame_num_utt[s] = lent[s]/skip_frames;
					frame_num_utt[s] += lent[s]%skip_frames > 0 ? 1 : 0;
					lent[s] = lent[s] > frame_num_utt[s]*skip_frames ? frame_num_utt[s]*skip_frames : lent[s];
					int32 utt_frames = opts->copy_posterior ? lent[s]:frame_num_utt[s];
					utt_nnet_out[s].Resize(utt_frames, out_dim, kUndefined);
					utt_copied[s] = false;
					utt_curt[s] = 0;

					delete example;
					break;
				}
			}

			// we are done if all streams are exhausted
			int done = 1;
			for (int s = 0; s < num_stream; s++) {
				if (curt[s] < lent[s] || utt_state_flags[s] == 1) done = 0;  // this stream still contains valid data, not exhausted
			}

			if (done) break;

			int N, n, nframes;
			if (opts->network_type == "lstm") {
				// fill a multi-stream bptt batch
				// * target: padded to batch_size
				// * feat: first shifted to achieve targets delay; then padded to batch_size
				for (int t = 0; t < batch_size * out_skip; t++) {
				   for (int s = 0; s < num_stream; s++) {
					   // feat shifting & padding
					   if (curt[s] + time_shift*skip_frames < lent[s]) {
						   feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]+time_shift*skip_frames));
					   } else {
						   int last = (frame_num_utt[s]-1)*skip_frames; // lent[s]-1
						   if (last >= 0)
						   feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(last));
					   }

					   curt[s] += in_skip;
					   if (curt[s] >= lent[s]) utt_state_flags[s] = 2;
				   }
			   }
			} else if (opts->network_type == "fsmn"){ // fsmn tdnn ...
	        	for (int s = 0; s < num_stream; s++) {
	        		N = opts->skip_inner ? skip_frames : 1;
	        		n = curt[s]+batch_size*N < lent[s] ? batch_size*N : lent[s]-curt[s];
	        		nframes = curt[s]+n < feats[s].NumRows() ? n : feats[s].NumRows()-curt[s];
                    nframes = nframes < 0 ? 0 : nframes;
                    valid_input_frames[s] = (nframes+N-1)/N;

	        		if (nframes > 0)
	        			feat.RowRange(s*batch_size*N, nframes).CopyFromMat(feats[s].RowRange(curt[s], nframes));
	        		if (nframes < batch_size*N)
	        			feat.RowRange(s*batch_size*N+nframes, batch_size*N-nframes).SetZero();

	        		if(nframes == 0) utt_state_flags[s] = 2; // utterance end
	        		curt[s] += nframes;
	        	}
			}


			num_frames = feat.NumRows();
			// report the speed
			if (num_done % 5000 == 0) {
			  time_now = time.Elapsed();
			  KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
							<< time_now/60 << " min; processed " << total_frames/time_now
							<< " frames per second.";
			}

			// apply optional feature transform
			//nnet_transf.Feedforward(CuMatrix<BaseFloat>(feat), &feats_transf);

			// for streams with new utterance, history states need to be reset
            if (opts->network_type == "lstm") {
			    nnet.ResetLstmStreams(new_utt_flags);
			    nnet.SetSeqLengths(new_utt_flags);
            } else if (opts->network_type == "fsmn")
			    nnet.SetStreamStatus(utt_state_flags, valid_input_frames);

			// forward pass
			//nnet.Feedforward(CuMatrix<BaseFloat>(feat), &nnet_out);
			nnet.Propagate(CuMatrix<BaseFloat>(feat), &nnet_out);
            //nnet_out.Resize(nnet_out_host.NumRows(),nnet_out_host.NumCols());

			// ctc prior, only scale blank label posterior
			if (blank_posterior_scale >= 0) {
			   nnet_out.ColRange(0, 1).Scale(blank_posterior_scale);
			}

			// convert posteriors to log-posteriors,
			if (apply_log) {
			  nnet_out.Add(1e-20); // avoid log(0),
			  nnet_out.ApplyLog();
			}

			// subtract log-priors from log-posteriors or pre-softmax,
			if (prior_opts->class_frame_counts != "") {
			  pdf_prior.SubtractOnLogpost(&nnet_out);
			}

			nnet_out.CopyToMat(&nnet_out_host);

			if (opts->network_type == "lstm") {
				for (int t = 0; t < batch_size; t++) {
				   for (int s = 0; s < num_stream; s++) {
					   // feat shifting & padding
					   if (opts->copy_posterior) {
						   for (int k = 0; k < skip_frames; k++){
								if (utt_curt[s] < lent[s]) {
									utt_nnet_out[s].Row(utt_curt[s]).CopyFromVec(nnet_out_host.Row(t*num_stream+s));
									utt_curt[s]++;
								}
						   }
					   } else {
						   if (utt_curt[s] < frame_num_utt[s]) {
							   utt_nnet_out[s].Row(utt_curt[s]).CopyFromVec(nnet_out_host.Row(t*num_stream+s));
							   utt_curt[s]++;
						   }
					   }
				   }
				}
			} else if (opts->network_type == "fsmn"){
				for (int s = 0; s < num_stream; s++) {
					for (int t = 0; t < valid_input_frames[s]; t++) {
						if (opts->copy_posterior) {
						   for (int k = 0; k < skip_frames; k++){
								if (utt_curt[s] < lent[s]) {
									utt_nnet_out[s].Row(utt_curt[s]).CopyFromVec(nnet_out_host.Row(s*batch_size+t));
									utt_curt[s]++;
								}
						   }
						} else {
						   if (utt_curt[s] < frame_num_utt[s]) {
							   utt_nnet_out[s].Row(utt_curt[s]).CopyFromVec(nnet_out_host.Row(s*batch_size+t));
							   utt_curt[s]++;
						   }
						}
					}
				}
			}

			total_frames += num_frames;
	    }

	    // for feed forward network, e.g. cnn,dnn,fsmn
	    if (num_stream < 1)
	    while ((example = dynamic_cast<FeatureExample*>(repository->ProvideExample())) != NULL)
	    {
			std::string utt = example->utt;
			Matrix<BaseFloat> &mat = example->input_frames;

            int len = mat.NumRows(), cur = 0;
            if (opts->skip_frames > len) {
                skip_frames = len;
                out_skip = opts->skip_inner ? skip_frames : 1;
                new_utt_flags.resize(1);
                new_utt_flags[0] = skip_frames;
            }
			if (time_shift > 0 || skip_frames > 1) {
				len = mat.NumRows()/skip_frames, cur = 0;
				len += mat.NumRows()%skip_frames > 0 ? 1 : 0;
				len *= out_skip;
				len += time_shift*out_skip;
				feat.Resize(len, mat.NumCols(), kUndefined);

				for (int32 i = 0; i < len; i++) {
                    if (cur >= mat.NumRows()) cur = mat.NumRows()-1;
					feat.Row(i).CopyFromVec(mat.Row(cur));
					cur += in_skip;
				}
				cufeat = feat; // push it to gpu,
			} else
				cufeat = mat; // push it to gpu,

			///only for nnet with fsmn component
			Vector<BaseFloat> flags;
			//flags.Resize(len/out_skip, kSetZero);
			flags.Resize(len, kSetZero);
			flags.Set(1.0);
			nnet.SetFlags(flags);

			// subsample forward
			// nnet.ResetSubSample(1, skip_frames);
			nnet.SetSeqLengths(new_utt_flags);
			// fwd-pass, feature transform,
			nnet_transf.Feedforward(cufeat, &feats_transf);

			// fwd-pass, nnet,
			//nnet.Feedforward(feats_transf, &nnet_out);
			nnet.Propagate(feats_transf, &nnet_out);

			if (time_shift > 0) {
				CuMatrix<BaseFloat> tmp = nnet_out;
				int rows = tmp.NumRows() - time_shift;
				nnet_out.Resize(rows, tmp.NumCols(), kUndefined);
				nnet_out.CopyFromMat(tmp.RowRange(time_shift, rows));
			}

			// ctc prior, only scale blank label posterior
			if (blank_posterior_scale >= 0) {
			   nnet_out.ColRange(0, 1).Scale(blank_posterior_scale);
			}

			// convert posteriors to log-posteriors,
			if (apply_log) {
				if (!(nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0)) {
				KALDI_WARN << utt << " "
						   << "Applying 'log' to data which don't seem to be probabilities "
						   << "(is there a softmax somwhere?)";
				}
				nnet_out.Add(1e-20); // avoid log(0),
				nnet_out.ApplyLog();
			}

			// subtract log-priors from log-posteriors or pre-softmax,
			if (prior_opts->class_frame_counts != "") {
			  if (nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0) {
				KALDI_WARN << utt << " "
						   << "Subtracting log-prior on 'probability-like' data in range [0..1] "
						   << "(Did you forget --no-softmax=true or --apply-log=true ?)";
			  }
			  pdf_prior.SubtractOnLogpost(&nnet_out);
			}

			// download from GPU,
			nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
			nnet_out.CopyToMat(&nnet_out_host);

			// check there's no nan/inf,
			if (!KALDI_ISFINITE(nnet_out_host.Sum())) {
			  KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
			}

			if (opts->copy_posterior) {
				Matrix<BaseFloat> tmp(nnet_out_host);
				nnet_out_host.Resize(mat.NumRows(), nnet_out.NumCols());

				int32 cur = 0;
				for (int32 i = 0; i < tmp.NumRows(); i++) {
					for (int k = 0; k < skip_frames; k++) {
						if (cur < mat.NumRows()) {
							nnet_out_host.Row(cur).CopyFromVec(tmp.Row(i));
							cur++;
						}
					}
				}
			}

			// write,
			TopKPerRow(nnet_out_host, utt_out_topk, opts->topk);
			examples_mutex->Lock();
			feature_writer->Write(utt, utt_out_topk);
			examples_mutex->Unlock();

			// progress log
			if (num_done % 100 == 0) {
			  time_now = time.Elapsed();
			  KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
							<< time_now/60 << " min; processed " << total_frames/time_now
							<< " frames per second.";
			}
			num_done++;
			total_frames += example->input_frames.NumRows();

			// release the buffers we don't need anymore
			delete example;
	    }

		examples_mutex->Lock();
		stats->num_done += num_done;
		stats->total_frames += total_frames;
		examples_mutex->Unlock();

	}


};


void NnetForwardParallel(const NnetForwardOptions *opts,
						std::string	model_filename,
						std::string feature_rspecifier,
						std::string sweep_frames_rspecifier,
						std::string feature_wspecifier,
						NnetForwardStats *stats)
{
    ExamplesRepository repository;
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader sweep_frames_reader(sweep_frames_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);
    Mutex examples_mutex;

    NnetForwardParallelClass c(opts, model_filename, &repository, &feature_writer, &examples_mutex, stats);

    		// The initialization of the following class spawns the threads that
    	    // process the examples.  They get re-joined in its destructor.
    	    MultiThreader<NnetForwardParallelClass> m(opts->num_threads, c);

		// iterate over all feature files
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

			example = new FeatureExample(&feature_reader, &sweep_frames_reader, opts);
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

} // namespace nnet
} // namespace kaldi
