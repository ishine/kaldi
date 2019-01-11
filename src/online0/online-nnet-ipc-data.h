// online0/online-nnet-ipc-data.h

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

#ifndef ONLINE0_ONLINE_NNET_IPC_DATA_H_
#define ONLINE0_ONLINE_NNET_IPC_DATA_H_

#include "util/circular-queue.h"
#include "util/kaldi-mutex.h"
#include "util/kaldi-thread.h"
#include "nnet0/nnet-trnopts.h"
#include "nnet0/nnet-pdf-prior.h"

#include "online0/online-ipc-message.h"
#include "online0/kaldi-unix-domain-socket.h"
#include "online0/online-nnet-ipc-forward.h"

namespace kaldi {

class OnlineNnetIpcDataClass : public MultiThreadable {
private:
	const static int MAX_BUFFER_SIZE = 10;
	const static int MATRIX_INC_STEP = 1024;
	const OnlineNnetIpcForwardOptions &opts_;
	std::vector<UnixDomainSocket*> &client_socket_;
    IpcForwardSync &forward_sync_;

    // check client data sample validity
    inline bool CheckSample(SocketSample &sample, int in_rows, int input_dim) {
        int size = sample.dim * sample.num_sample;
        int max_size = in_rows * input_dim;
        if (size < 0) {
            KALDI_LOG << Timer::CurrentTime() <<" Invalid sample, dim = " << sample.dim << " num_sample = " << sample.num_sample;
            return false;
        } else if (size > max_size) {
            KALDI_LOG << Timer::CurrentTime() <<" Client sample size " << size << " exceed maximum socket sample size " << max_size;
            return false;
        } else if (input_dim != sample.dim) {
            KALDI_LOG << Timer::CurrentTime() <<" Client sample dim " << sample.dim << " is not consistent with model input dim " << input_dim;
            return false;
        } else if (sample.is_end == 0 && sample.num_sample != opts_.batch_size) {
            KALDI_LOG << Timer::CurrentTime() << " number of frame in client sample " << sample.num_sample << " is not consistent with forward batch size " << opts_.batch_size;
            return false;
        }
        return true;
    }
        

public:
	OnlineNnetIpcDataClass(const OnlineNnetIpcForwardOptions &opts,
							std::vector<UnixDomainSocket*> &client_socket, IpcForwardSync &forward_sync):
							opts_(opts), client_socket_(client_socket), forward_sync_(forward_sync) {

	}

	~OnlineNnetIpcDataClass() {}

	void operator () () {
        using namespace kaldi::nnet0;

		int32 num_stream = opts_.num_stream;
		int32 batch_size = opts_.batch_size;
		int32 skip_frames = opts_.skip_frames;
		int32 input_dim = opts_.input_dim;
		int32 output_dim = opts_.output_dim;


	    CircularQueue<SocketDecodable* > default_queue(MAX_BUFFER_SIZE, NULL);
	    std::vector<CircularQueue<SocketDecodable* > > decodable_buffer(num_stream, default_queue);

	    std::vector<int> curt(num_stream, 0);
	    std::vector<int> lent(num_stream, 0);
	    std::vector<int> frame_num_utt(num_stream, 0);
	    std::vector<int> utt_curt(num_stream, 0);
	    std::vector<int> recv_end(num_stream, 0);
	    std::vector<int> send_end(num_stream, 1);
	    std::vector<int> send_frames(num_stream, 0);
	    std::vector<int> new_utt_flags(num_stream, 1);
	    std::vector<int> update_state_flags(num_stream, 1);
	    std::vector<int> forward_processed(num_stream, 1);

	 	std::vector<Matrix<BaseFloat> > feats(num_stream);
	    Matrix<BaseFloat> sample, *p_nnet_in = NULL, *p_nnet_out = NULL;

	    int in_rows, out_rows;
	    int in_skip, out_skip;
	    int sc_sample_size, sc_decodable_size;
	    int t, s , k, len, idx = 0;
	    bool in_sucess = false;

	    in_skip = opts_.skip_inner ? 1 : opts_.skip_frames;
	    out_skip = opts_.skip_inner ? opts_.skip_frames : 1;
	    in_rows = batch_size*in_skip;
	    out_rows = (batch_size+out_skip-1)/out_skip;

        SocketSample *socket_sample = NULL;
        SocketDecodable *decodable = NULL;
	    sc_sample_size = sizeof(SocketSample) + in_rows*input_dim*sizeof(BaseFloat);
	    sc_decodable_size = sizeof(SocketDecodable) + out_rows*output_dim*sizeof(BaseFloat);
	    socket_sample = (SocketSample*) new char[sc_sample_size];

	    DataInBuffer **data_in_buffer = new DataInBuffer[2](in_rows, input_dim, num_stream);
	    DataOutBuffer *out_buffer = NULL;
	    p_nnet_in = &data_in_buffer[idx]->data_in_;

        Timer time, gap_time;

	    while (true) {
            // send network output data
	    	for (s = 0; s < num_stream; s++) {
	    		while (!decodable_buffer[s].Empty() && send_end[s] == 0 && client_socket_[s] != NULL) {
	    			SocketDecodable* decodable = *(decodable_buffer[s].Front());
                    gap_time.Reset();
	    			int ret = client_socket_[s]->Send((void*)decodable, sc_decodable_size, MSG_NOSIGNAL);
	    			forward_sync_.time_send_ += gap_time.Elapsed();

	    			if (ret > 0 && ret != sc_decodable_size) 
                        KALDI_WARN << Timer::CurrentTime() <<" Send socket decodable: " << ret << " less than " << sc_decodable_size;
	    			if (ret <= 0) break;

	    			// send successful
	    			decodable_buffer[s].Pop();
                    send_frames[s] += decodable->num_sample;

	    			// send a utterance finished
	    			if(decodable->is_end)
	    				send_end[s] = 1;
	    		}
	    	}

	    	// loop over all streams, feed each stream with new data
	    	for (s = 0; s < num_stream; s++) {
	    		if (client_socket_[s] == NULL)
	    			continue;

	    		if (client_socket_[s]->isClosed()) {
	    			delete client_socket_[s];
	    			client_socket_[s] = NULL;
	    			send_end[s] = 1;
    				lent[s] = 0;
                    decodable_buffer[s].Resize(MAX_BUFFER_SIZE);
                    KALDI_LOG << Timer::CurrentTime() << " Client decoder " << s << " disconnected.";
	    			continue;
	    		}

	    		if (send_end[s] == 1) {
	    			recv_end[s] = 0;
    				lent[s] = 0;
    				curt[s] = 0;
    				utt_curt[s] = 0;
    				new_utt_flags[s] = 1;
                    send_frames[s] = 0;
                    frame_num_utt[s] = 0;
                    decodable_buffer[s].Resize(MAX_BUFFER_SIZE);
	    		}

	    		while (recv_end[s] == 0) {
                    gap_time.Reset();
                    len = client_socket_[s]->Receive((void*)socket_sample, sc_sample_size);
                    forward_sync_.time_received_ += gap_time.Elapsed();

                    if (len <= 0)
                        break;

                    // socket sample validity
                    if (len != sc_sample_size || !CheckSample(*socket_sample, in_rows, input_dim)) {
                        send_end[s] = 1;
                        client_socket_[s]->Close();
                        break;
                    }

                    if (feats[s].NumRows() == 0)
                        feats[s].Resize(MATRIX_INC_STEP, socket_sample->dim, kUndefined, kStrideEqualNumCols);

					if (feats[s].NumRows() < lent[s]+socket_sample->num_sample) {
						Matrix<BaseFloat> tmp(feats[s].NumRows()+MATRIX_INC_STEP, socket_sample->dim, kUndefined, kStrideEqualNumCols);
						tmp.RowRange(0, lent[s]).CopyFromMat(feats[s].RowRange(0, lent[s]));
						feats[s].Swap(&tmp);
					}

                    int size = socket_sample->dim * socket_sample->num_sample * sizeof(float);
                    if (size > 0)
					    memcpy((char*)feats[s].RowData(lent[s]), (char*)socket_sample->sample, size);
					lent[s] += socket_sample->num_sample;
					recv_end[s] = socket_sample->is_end;

					frame_num_utt[s] = (lent[s]+out_skip-1)/out_skip;
	    		}
	    	}

	        // we are done if all streams are exhausted
	        bool done = true;
	        for (s = 0; s < num_stream; s++) {
	            if (curt[s] < lent[s] || (recv_end[s] == 1 && send_end[s] == 0)) done = false;  // this stream still contains valid data, not exhausted
	        }

	        if (done) {
	        	usleep(0.02*1000000);
	        	continue;
	        }

	    	 // fill a multi-stream bptt batch
	    	for (t = 0; t < batch_size; t++) {
	    		for (s = 0; s < num_stream; s++) {
		    		if (forward_processed[s] == 0)
		    			continue;
					if (curt[s] < lent[s]) {
						p_nnet_in->Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]));
					    curt[s] += in_skip;
					    update_state_flags[s] = 1;
					} else {
						update_state_flags[s] = 0;
					}
	    		}
	    	}
	    	for (s = 0; s < num_stream; s++)
	    		forward_processed[s] = update_state_flags[s] == 1 ? 0 : 1;


	    	// push forward data
	    	data_in_buffer[idx]->update_state_flags_ = update_state_flags;
	    	data_in_buffer[idx]->new_utt_flags_ = new_utt_flags;
	    	in_sucess = forward_sync_.repo_in_->TryAccept(data_in_buffer[idx]);
	    	if (in_sucess) {
	    		for (s = 0; s < num_stream; s++)
	    			forward_processed[s] = 1;
	    		idx = (idx+1)%2;
	    		p_nnet_in = &data_in_buffer[idx]->data_in_;
	    	}


            double curt_time = time.Elapsed();
            if (curt_time - forward_sync_.time_now_ >= 2) {
                KALDI_LOG << Timer::CurrentTime() <<
                		" Thread " << this->thread_id_ << ", time elapsed: " << curt_time
                		<< " s, socket send: " << forward_sync_.time_send_
						<< " s, socket receive: " << forward_sync_.time_received_
						<< " s, gpu forward: " << forward_sync_.time_forward_ << " s.";
                forward_sync_.time_now_ = time.Elapsed();
            }

			// receive forward data
            out_buffer = (DataOutBuffer *)forward_sync_.repo_out_->TryProvide();
            if (NULL == out_buffer)
            	continue;

            p_nnet_out = &out_buffer->data_out_;
            // rearrange output for each client
			for (s = 0; s < num_stream; s++) {
				if (opts_.copy_posterior) {
					if ((utt_curt[s] >= lent[s] && send_end[s] == 1) || 
                        (utt_curt[s] >= lent[s] && recv_end[s] == 0 && send_end[s] == 0))
						continue;
				} else if ((utt_curt[s] >= frame_num_utt[s] && send_end[s] == 1) ||
                        (utt_curt[s] >= frame_num_utt[s] && recv_end[s] == 0 && send_end[s] == 0))
					continue;

				// get new decodable buffer
				decodable_buffer[s].Push();
				SocketDecodable **pp_decodable = decodable_buffer[s].Back();
				if (*pp_decodable == NULL)
					*pp_decodable = (SocketDecodable*)new char[sc_decodable_size];
				decodable = *pp_decodable;
                decodable->clear(); 

				output_dim = p_nnet_out->NumCols();
				float *dest = decodable->sample;
                int nframes = opts_.copy_posterior ? skip_frames : 1;
                int ncurt = opts_.copy_posterior ? curt[s] : (curt[s]+skip_frames-1)/skip_frames;
                int nlen = opts_.copy_posterior ? lent[s] : frame_num_utt[s];

				for (t = 0; t < out_rows; t++) {
                    for (k = 0; k < nframes; k++) {
						if (utt_curt[s] < ncurt && utt_curt[s] < nlen) {
                            memcpy((char*)dest, (char*)p_nnet_out->RowData(t * num_stream + s), output_dim*sizeof(float));
							dest += output_dim;
							utt_curt[s]++;
							decodable->num_sample++;
                        }
                    }
				}

				decodable->dim = output_dim;
				decodable->is_end = recv_end[s] == 1 && ((opts_.copy_posterior && utt_curt[s] == lent[s]) ||
										(!opts_.copy_posterior && utt_curt[s] == frame_num_utt[s]));
				new_utt_flags[s] = decodable->is_end ? 1 : 0;
				send_end[s] = 0;
			} // rearrangement

	    } // while loop
	}
};


}// namespace kaldi

#endif /* ONLINE0_ONLINE_NNET_IPC_DATA_H_ */
