// lm/example.h

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

#ifndef LM_LM_EXAMPLE_H_
#define LM_LM_EXAMPLE_H_

#include <deque>
#include "util/table-types.h"

#include "lm/am-compute-parallel.h"
#include "lm/am-compute-lstm-parallel.h"
#include "lm/am-compute-ctc-parallel.h"

namespace kaldi {
namespace lm {


struct Example {
	SequentialBaseFloatMatrixReader *feature_reader;
	RandomAccessBaseFloatMatrixReader *si_feature_reader;
	RandomAccessTokenReader *spec_aug_reader;

	std::string utt;
	Matrix<BaseFloat> input_frames;
	std::vector<int32> sweep_frames;
	bool inner_skipframes;

	Matrix<BaseFloat> si_input_frames;
	bool use_kld;

	Example(SequentialBaseFloatMatrixReader *feature_reader,
			RandomAccessBaseFloatMatrixReader *si_feature_reader, 
			RandomAccessTokenReader *spec_aug_reader):
		feature_reader(feature_reader), si_feature_reader(si_feature_reader),
		spec_aug_reader(spec_aug_reader), inner_skipframes(false), use_kld(false) {}

    void SetSweepFrames(const std::vector<int32> &frames, bool inner = false) {
        sweep_frames = frames;
        inner_skipframes = inner;
    }

	virtual ~Example() {}
	virtual bool PrepareData(std::vector<Example*> &examples) = 0;
};

struct DNNExample : Example {
	RandomAccessPosteriorReader *targets_reader;
	RandomAccessBaseFloatVectorReader *weights_reader;

	LmModelSync *model_sync;
	NnetStats *stats;
	const NnetUpdateOptions *opts;


	Posterior targets;
	Vector<BaseFloat> frames_weights;

	DNNExample(SequentialBaseFloatMatrixReader *feature_reader,
					RandomAccessBaseFloatMatrixReader *si_feature_reader,
					RandomAccessTokenReader *spec_aug_reader,
					RandomAccessPosteriorReader *targets_reader,
					RandomAccessBaseFloatVectorReader *weights_reader,
					LmModelSync *model_sync,
					NnetStats *stats,
					const NnetUpdateOptions *opts):
	Example(feature_reader, si_feature_reader, spec_aug_reader),
				targets_reader(targets_reader), weights_reader(weights_reader),
				model_sync(model_sync), stats(stats), opts(opts) {
		if (opts->kld_scale > 0 && opts->si_feature_rspecifier != "")
			use_kld = true;
	}

    
	bool PrepareData(std::vector<Example*> &examples);
};

struct CTCExample : Example {
	RandomAccessInt32VectorReader *targets_reader;

	LmModelSync *model_sync;
	NnetCtcStats *stats;
	const NnetUpdateOptions *opts;

	std::vector<int32> targets;

	CTCExample(SequentialBaseFloatMatrixReader *feature_reader,
					RandomAccessBaseFloatMatrixReader *si_feature_reader,
					RandomAccessTokenReader *spec_aug_reader,
					RandomAccessInt32VectorReader *targets_reader,
					LmModelSync *model_sync,
					NnetCtcStats *stats,
					const NnetUpdateOptions *opts):
	Example(feature_reader, si_feature_reader, spec_aug_reader), targets_reader(targets_reader),
	model_sync(model_sync), stats(stats), opts(opts) {
		if (opts->kld_scale > 0 && opts->si_feature_rspecifier != "")
			use_kld = true;
	}
	bool PrepareData(std::vector<Example*> &examples);
};


struct RNNTExample : Example {
	RandomAccessInt32VectorReader *wordid_reader;
	NnetStats *stats;
	const NnetUpdateOptions *opts;

	std::vector<int32> input_wordids;

	RNNTExample(SequentialBaseFloatMatrixReader *feature_reader,
					RandomAccessBaseFloatMatrixReader *si_feature_reader,
					RandomAccessTokenReader *spec_aug_reader,
					RandomAccessInt32VectorReader *wordid_reader,
					NnetStats *stats,
					const NnetUpdateOptions *opts):
	Example(feature_reader, si_feature_reader, spec_aug_reader), wordid_reader(wordid_reader),
	stats(stats), opts(opts) {}

	bool PrepareData(std::vector<Example*> &examples);
};

struct LmExample : Example {
    SequentialInt32VectorReader *wordid_reader;

    const NnetUpdateOptions *opts;

    std::vector<int32> input_wordids;

    LmExample(SequentialInt32VectorReader *wordid_reader,
                    const NnetUpdateOptions *opts):
    Example(NULL, NULL, NULL), wordid_reader(wordid_reader), opts(opts) {}


    bool PrepareData(std::vector<Example*> &examples);
};

struct SluExample : Example {
	const NnetUpdateOptions *opts;
	SequentialInt32VectorReader *wordid_reader;
	RandomAccessInt32VectorReader *slot_reader;
	RandomAccessInt32VectorReader *intent_reader;

	std::vector<int32> input_wordids;
	std::vector<int32> input_slotids;
	std::vector<int32> input_intentids;

	SluExample(const NnetUpdateOptions *opts,
					SequentialInt32VectorReader *wordid_reader,
					RandomAccessInt32VectorReader *slot_reader = NULL,
					RandomAccessInt32VectorReader *intent_reader = NULL):
	Example(NULL, NULL, NULL), opts(opts), wordid_reader(wordid_reader),
	slot_reader(slot_reader), intent_reader(intent_reader) {}


	bool PrepareData(std::vector<Example*> &examples);
};

struct SeqLabelExample : Example {
	const NnetUpdateOptions *opts;
	SequentialInt32VectorReader *wordid_reader;
	RandomAccessInt32VectorReader *label_reader;

	std::vector<int32> input_wordids;
	std::vector<int32> input_labelids;

	SeqLabelExample(const NnetUpdateOptions *opts,
					SequentialInt32VectorReader *wordid_reader,
					RandomAccessInt32VectorReader *label_reader):
	Example(NULL, NULL, NULL), opts(opts), wordid_reader(wordid_reader),
	label_reader(label_reader) {}


	bool PrepareData(std::vector<Example*> &examples);
};



/** This struct stores neural net training examples to be used in
    multi-threaded training.  */
class ExamplesRepository {
 public:
  /// The following function is called by the code that reads in the examples.
  void AcceptExample(Example *example);

  /// The following function is called by the code that reads in the examples,
  /// when we're done reading examples; it signals this way to this class
  /// that the stream is now empty
  void ExamplesDone();

  /// This function is called by the code that does the training.  If there is
  /// an example available it will provide it, or it will sleep till one is
  /// available.  It returns NULL when there are no examples left and
  /// ExamplesDone() has been called.
  Example *ProvideExample();

  ExamplesRepository(int32 buffer_size = 128): buffer_size_(buffer_size),
                                      empty_semaphore_(buffer_size_),
                                      done_(false) {}
 private:
  int32 buffer_size_;
  Semaphore full_semaphore_;
  Semaphore empty_semaphore_;
  Mutex examples_mutex_; // mutex we lock to modify examples_.

  std::deque<Example*> examples_;
  bool done_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(ExamplesRepository);
};

} // namespace nnet
} // namespace kaldi


#endif /* NNET_NNET_EXAMPLE_H_ */
