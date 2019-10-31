// online0/online-am-vad.h
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

#ifndef ONLINE0_ONLINE_AM_VAD_H_
#define ONLINE0_ONLINE_AM_VAD_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "decoder/decodable-matrix.h"

namespace kaldi {


struct OnlineAmVadOptions {

	int32 vad_smooth_win;
	int32 vad_num_end_win;
	float vad_start_rate;
	float vad_end_rate;

	OnlineAmVadOptions():vad_smooth_win(3), vad_num_end_win(2),
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

class OnlineAmVad {
public:
	OnlineAmVad(const OnlineAmVadOptions &opts):
		opts_(opts), state_(UTT_END), num_end_win_(0) {
	}

	UttState FeedData(Matrix<BaseFloat> &post) {
		int nr, swin, bs;
		BaseFloat pb = 0.0;
		nr = post.NumRows();
        blank_post_.Resize(nr, kSetZero);
        for (int i = 0; i < nr; i++)
            blank_post_(i) = post(i, 0);

		if (state_ == UTT_END) {
			swin = opts_.vad_smooth_win > nr ? nr : opts_.vad_smooth_win;
			bs = nr-swin+1;
			smooth_buffer_.Resize(bs, kSetZero);
			for (int i = 0; i < bs; i++) {
				for (int j = 0; j < swin; j++) {
					smooth_buffer_(i) += blank_post_(i+j);
				}
				smooth_buffer_(i) /= swin;
				if (smooth_buffer_(i) <= opts_.vad_start_rate) {
					state_ = UTT_START;
                    num_end_win_ = 0;
					break;
				}
			}
		} else if (state_ == UTT_START || state_ == UTT_APPEND) {
			pb = blank_post_.Sum()/nr;
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
	const OnlineAmVadOptions &opts_;
	UttState state_;
	Vector<BaseFloat> smooth_buffer_;
    Vector<BaseFloat> blank_post_;
	int num_end_win_;
};

}	 // namespace kaldi

#endif /* ONLINE0_ONLINE_VAD_H_ */
