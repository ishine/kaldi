// online0/online-vad.h
// Copyright 2018	Alibaba Inc (author: Wei Deng)

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

#ifndef ONLINE0_ONLINE_VAD_H_
#define ONLINE0_ONLINE_VAD_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "decoder/decodable-matrix.h"

namespace kaldi {


struct OnlineVadOptions {

	int32 vad_smooth_win;
	int32 vad_num_end_win;
	float vad_start_rate;
	float vad_end_rate;

	OnlineVadOptions():vad_smooth_win(3), vad_num_end_win(2),
			vad_start_rate(0.6), vad_end_rate(0.9) {}

	void Register(OptionsItf *po) {
		po->Register("vad-smooth-win", &vad_smooth_win, "Vad smooth window at the start of utterance.");
		po->Register("vad-num-end-win", &vad_num_end_win, "the number of window to detect the utterance ending.");
		po->Register("vad-start-rate", &vad_start_rate, "Exceed the number, we believe the utterance starting");
		po->Register("vad-end-rate", &vad_end_rate, "Under the number, we believe the utterance ending");
	}
};

typedef enum {
	UTT_START,
	UTT_APPEND,
	UTT_END,
}UttState;

class OnlineVad {
public:
	OnlineVad(const OnlineVadOptions &opts):
		opts_(opts), state_(UTT_END), num_end_win_(0) {
	}

	FeatState FeedData(Matrix<BaseFloat> &post) {
		int nr, swin, bs;
		BaseFloat pb = 0.0;
		bool flag = false;
		nr = post.NumRows();
		if (state_ == UTT_END) {
			swin = opts_.vad_smooth_win > nr ? nr : opts_.vad_smooth_win;
			bs = nr-swin+1;
			smooth_buffer_.Resize(bs, kSetZero);
			for (int i = 0; i < bs; i++) {
				for (int j = 0; j < swin; j++) {
					smooth_buffer_[i] += post(i+j, 0);
				}
				smooth_buffer_[i] /= swin;
				if (smooth_buffer_[i] >= opts_.vad_start_rate) {
					state_ = UTT_START;
					break;
				}
			}
		} else if (state_ == UTT_START || state_ == UTT_APPEND) {
			pb = post.ColRange(0, 1).Sum()/nr;
			num_end_win_ = pb >= opts_.vad_end_rate ? num_end_win_+1 : 0;
			state_ = num_end_win_ >= opts_.vad_num_end_win ? UTT_END : UTT_APPEND;
		}
		return state_;
	}

	void Reset() {
		num_end_win_ = 0;
		state_ = UTT_END;
	}

private:
	const OnlineVadOptions &opts_;
	FeatState state_;
	Vector<BaseFloat> smooth_buffer_;
	int num_end_win_;
};

}	 // namespace kaldi

#endif /* ONLINE0_ONLINE_VAD_H_ */
