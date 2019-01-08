// online0/online-ipc-message.h

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

#ifndef ONLINE0_ONLINE_IPC_MESSAGE_H_
#define ONLINE0_ONLINE_IPC_MESSAGE_H_

#include <sys/types.h>
#include <unistd.h>

#define MAX_FILE_PATH 256
#define MAX_KEY_LEN 256

namespace kaldi {

struct MQSample {
	MQSample():pid(-1),is_end(false),num_sample(0),dim(0),prio(0){}
	pid_t pid;
	char mq_callback_name[MAX_FILE_PATH];
	char uttt_key[MAX_KEY_LEN];
	int is_end;
	int  num_sample;
	int	 dim;
	int	 prio;
	float sample[0];
};

struct MQDecodable {
	MQDecodable():is_end(0),num_sample(0),dim(0),prio(0){}
	int is_end;
	int num_sample;
	int	dim;
	int	prio;
	float sample[0];
};

// feature for network input
struct SocketSample {
	SocketSample():pid(-1),is_end(0),
			num_sample(0),dim(0){}
    void clear() { 
        pid = -1;
        is_end = false;
        num_sample = 0;
        dim = 0;
    }
    // client decoder pid
	pid_t pid;
    // utterance name 
	char utt_key[MAX_KEY_LEN];
    // utterance end flag
	int is_end;
    // number of frame
	int  num_sample;
    // a frame dim
	int	 dim;
    // frames data
	float sample[0];
};

// network output for client decoder
struct SocketDecodable {
	SocketDecodable():is_end(0),
			num_sample(0),dim(0){}
    void clear() { 
        is_end = false;
        num_sample = 0;
        dim = 0;
    }
    // utterance end flag
	int is_end;
    // number of frame
	int  num_sample;
    // a frame dim
	int	 dim;
    // frames data
	float sample[0];
};

}

#endif /* ONLINE0_ONLINE_IPC_MESSAGE_H_ */
