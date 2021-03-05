// lm/example.cc

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
#include "hmm/posterior.h"
#include "lat/lattice-functions.h"
#include "lm/example.h"

namespace kaldi {
namespace lm {

bool DNNExample::PrepareData(std::vector<Example*> &examples) {
	utt = feature_reader->Key();
	KALDI_VLOG(3) << "Reading " << utt;
	// check that we have targets
	if (!targets_reader->HasKey(utt)) {
	  KALDI_WARN << utt << ", missing targets";
	  model_sync->LockStates();
	  stats->num_no_tgt_mat++;
	  model_sync->UnlockStates();
	  return false;
	}
	// check we have per-frame weights
	if (opts->frame_weights != "" && !weights_reader->HasKey(utt)) {
	  KALDI_WARN << utt << ", missing per-frame weights";
	  model_sync->LockStates();
	  stats->num_other_error++;
	  model_sync->UnlockStates();
	  return false;
	}

	// get feature / target pair
	input_frames = feature_reader->Value();
	if (use_kld) si_input_frames = si_feature_reader->Value(utt);
	targets = targets_reader->Value(utt);
	// get per-frame weights
	if (opts->frame_weights != "") {
		frames_weights = weights_reader->Value(utt);
	} else { // all per-frame weights are 1.0
		frames_weights.Resize(targets.size());
		frames_weights.Set(1.0);
	}

	// split feature
	int32 skip_frames = opts->skip_frames;
	int32 sweep_time = opts->sweep_time;

	// correct small length mismatch ... or drop sentence
	{
	  // add lengths to vector
	  std::vector<int32> lenght;
	  lenght.push_back(input_frames.NumRows());
	  lenght.push_back(targets.size());
	  lenght.push_back(frames_weights.Dim());
	  // find min, max
	  int32 min = *std::min_element(lenght.begin(),lenght.end());
	  int32 max = *std::max_element(lenght.begin(),lenght.end());
	  // fix or drop ?
	  if (skip_frames > 1 && (max+skip_frames-1)/skip_frames >= min && (max+skip_frames-1)/skip_frames - min < opts->length_tolerance) {
		  if((input_frames.NumRows()+skip_frames-1)/skip_frames > min) input_frames.Resize(min*skip_frames, input_frames.NumCols(), kCopyData);
		  if(targets.size() != min) targets.resize(min);
		  if(frames_weights.Dim() != min) frames_weights.Resize(min, kCopyData);
	  } else if (max - min < opts->length_tolerance) {
		if(input_frames.NumRows() != min) input_frames.Resize(min, input_frames.NumCols(), kCopyData);
		if(targets.size() != min) targets.resize(min);
		if(frames_weights.Dim() != min) frames_weights.Resize(min, kCopyData);
	  } else {
		KALDI_WARN << utt << ", length mismatch of targets " << targets.size()
				   << " and features " << input_frames.NumRows();
		model_sync->LockStates();
		stats->num_other_error++;
		model_sync->UnlockStates();
		return false;
	  }
	}

	examples.clear();

	if (sweep_time>skip_frames) {
		KALDI_WARN << "sweep time for each utterance should less than skip frames (it reset to skip frames)";
		sweep_time = skip_frames;
	}

	if (skip_frames <= 1) {
		examples.push_back(this);
		return true;
	}

	if (sweep_time == skip_frames) {
		this->sweep_frames.resize(sweep_time);
		for (int i = 0; i < sweep_time; i++)
			sweep_frames[i] = i;
	}

	DNNExample *example = NULL;
	int32 lent, feat_lent, cur,
		utt_len = input_frames.NumRows();
	bool ali_skip = (utt_len+skip_frames-1)/skip_frames == targets.size() ? false : true;

	for (int i = 0; i < sweep_frames.size(); i++) {
		example = new DNNExample(feature_reader, si_feature_reader, spec_aug_reader,
                                        targets_reader, weights_reader, model_sync, stats, opts);
		example->utt = utt;
		lent = utt_len/skip_frames;
		lent += utt_len%skip_frames > sweep_frames[i] ? 1 : 0;
		//feat_lent = this->inner_skipframes ? utt_len-sweep_frames[i] : lent;
		feat_lent = this->inner_skipframes ? lent*skip_frames : lent;
		example->input_frames.Resize(feat_lent, input_frames.NumCols());
		if (use_kld) example->si_input_frames.Resize(feat_lent, input_frames.NumCols());
		example->targets.resize(lent);
		example->frames_weights.Resize(lent);

		cur = sweep_frames[i];
		for (int j = 0; j < feat_lent; j++) {
			example->input_frames.Row(j).CopyFromVec(input_frames.Row(cur));
			if (use_kld) example->si_input_frames.Row(j).CopyFromVec(si_input_frames.Row(cur));
			cur = this->inner_skipframes ? cur+1 : cur+skip_frames;
			if (cur >= utt_len) cur = utt_len-1;
		}

		cur = ali_skip ? sweep_frames[i] : 0;
		for (int j = 0; j < lent; j++) {
			example->targets[j] = targets[cur];
			example->frames_weights(j) = frames_weights(cur);
			cur += ali_skip ? skip_frames : 0;
		}
		examples.push_back(example);

    	// spectrum augmentation
    	if (opts->use_specaug) {
    		if (this->spec_aug_reader == NULL || (this->spec_aug_reader != NULL && this->spec_aug_reader->HasKey(utt))) {
				DNNExample *spec_example = new DNNExample(feature_reader, si_feature_reader, spec_aug_reader,
                                                                    targets_reader, weights_reader, model_sync, stats, opts);
				*spec_example = *example;
				spec_example->input_frames.SpecAugment(opts->spec_opts->num_time_mask, opts->spec_opts->max_time_mask,
						opts->spec_opts->time_mask_ratio, opts->spec_opts->num_freq_mask, opts->spec_opts->max_freq_mask);
				examples.push_back(spec_example);
    		}
    	}
	}

	return true;
}

bool CTCExample::PrepareData(std::vector<Example*> &examples) {
    utt = feature_reader->Key();
    KALDI_VLOG(3) << "Reading " << utt;
    // check that we have targets
    if (!targets_reader->HasKey(utt)) {
      KALDI_WARN << utt << ", missing targets";
      model_sync->LockStates();
      stats->num_no_tgt_mat++;
      model_sync->UnlockStates();
      return false;
    }

    // get feature / target pair
    input_frames = feature_reader->Value();
    targets = targets_reader->Value(utt);
    if (use_kld) si_input_frames = si_feature_reader->Value(utt);

    examples.clear();

    // split feature
    int32 skip_frames = opts->skip_frames;
    int32 sweep_time = opts->sweep_time;

    if (sweep_time>skip_frames) {
    	KALDI_WARN << "sweep time for each utterance should less than skip frames (it reset to skip frames)";
    	sweep_time = skip_frames;
    }

    if (skip_frames <= 1) {
    	examples.push_back(this);
    	return true;
    }

    if (sweep_time == skip_frames) {
    	this->sweep_frames.resize(sweep_time);
    	for (int i = 0; i < sweep_time; i++)
    		sweep_frames[i] = i;
    }

    CTCExample *example = NULL;
    int32 lent, feat_lent, cur,
		utt_len = input_frames.NumRows();
    for (int i = 0; i < sweep_frames.size(); i++) {
    	example = new CTCExample(feature_reader, si_feature_reader, spec_aug_reader,
                                    targets_reader, model_sync, stats, opts);
    	example->utt = utt;
    	example->targets = targets;

    	lent = utt_len/skip_frames;
    	lent += utt_len%skip_frames > sweep_frames[i] ? 1 : 0;
		//feat_lent = this->inner_skipframes ? utt_len-sweep_frames[i] : lent;
    	feat_lent = this->inner_skipframes ? lent*skip_frames : lent;
    	example->input_frames.Resize(feat_lent, input_frames.NumCols());
    	if (use_kld) example->si_input_frames.Resize(feat_lent, input_frames.NumCols());

    	cur = sweep_frames[i];
    	for (int j = 0; j < feat_lent; j++) {
    		example->input_frames.Row(j).CopyFromVec(input_frames.Row(cur));
    		if (use_kld) example->si_input_frames.Row(j).CopyFromVec(si_input_frames.Row(cur));
    		cur = this->inner_skipframes ? cur+1 : cur+skip_frames;
    		if (cur >= utt_len) cur = utt_len-1;
    	}
    	examples.push_back(example);

    	// spectrum augmentation
    	if (opts->use_specaug) {
    		if (this->spec_aug_reader == NULL || (this->spec_aug_reader != NULL && this->spec_aug_reader->HasKey(utt))) {
				CTCExample *spec_example = new CTCExample(feature_reader, si_feature_reader, spec_aug_reader,
                                                                    targets_reader, model_sync, stats, opts);
				*spec_example = *example;
				spec_example->input_frames.SpecAugment(opts->spec_opts->num_time_mask, opts->spec_opts->max_time_mask,
								opts->spec_opts->time_mask_ratio, opts->spec_opts->num_freq_mask, opts->spec_opts->max_freq_mask);
				examples.push_back(spec_example);
    		}
    	}
    }

    return true;
}

bool RNNTExample::PrepareData(std::vector<Example*> &examples) {
    utt = feature_reader->Key();
    KALDI_VLOG(3) << "Reading " << utt;
    // check that we have targets
    if (!wordid_reader->HasKey(utt)) {
      KALDI_WARN << utt << ", missing targets";
      stats->num_no_tgt_mat++;
      return false;
    }

    // get feature / target pair
    input_frames = feature_reader->Value();
    input_wordids = wordid_reader->Value(utt);
    if (use_kld) si_input_frames = si_feature_reader->Value(utt);

    examples.clear();

    // split feature
    int32 skip_frames = opts->skip_frames;
    int32 sweep_time = opts->sweep_time;

    if (sweep_time>skip_frames) {
    	KALDI_WARN << "sweep time for each utterance should less than skip frames (it reset to skip frames)";
    	sweep_time = skip_frames;
    }

    if (skip_frames <= 1) {
    	examples.push_back(this);
    	return true;
    }

    if (sweep_time == skip_frames) {
    	this->sweep_frames.resize(sweep_time);
    	for (int i = 0; i < sweep_time; i++)
    		sweep_frames[i] = i;
    }

    RNNTExample *example = NULL;
    int32 lent, feat_lent, cur,
		utt_len = input_frames.NumRows();
    for (int i = 0; i < sweep_frames.size(); i++) {
    	example = new RNNTExample(feature_reader,
    			si_feature_reader, spec_aug_reader, wordid_reader, stats, opts);
    	example->utt = utt;
    	example->input_wordids = input_wordids;

    	lent = utt_len/skip_frames;
    	lent += utt_len%skip_frames > sweep_frames[i] ? 1 : 0;
		//feat_lent = this->inner_skipframes ? utt_len-sweep_frames[i] : lent;
    	feat_lent = this->inner_skipframes ? lent*skip_frames : lent;
    	example->input_frames.Resize(feat_lent, input_frames.NumCols());
    	if (use_kld) example->si_input_frames.Resize(feat_lent, input_frames.NumCols());

    	cur = sweep_frames[i];
    	for (int j = 0; j < feat_lent; j++) {
    		example->input_frames.Row(j).CopyFromVec(input_frames.Row(cur));
    		if (use_kld) example->si_input_frames.Row(j).CopyFromVec(si_input_frames.Row(cur));
    		cur = this->inner_skipframes ? cur+1 : cur+skip_frames;
    		if (cur >= utt_len) cur = utt_len-1;
    	}

    	examples.push_back(example);

    	// spectrum augmentation
    	if (opts->use_specaug) {
    		if (this->spec_aug_reader == NULL || (this->spec_aug_reader != NULL && this->spec_aug_reader->HasKey(utt))) {
    	        RNNTExample *spec_example = new RNNTExample(feature_reader,
    			                                si_feature_reader, spec_aug_reader, wordid_reader, stats, opts);
				*spec_example = *example;
				spec_example->input_frames.SpecAugment(opts->spec_opts->num_time_mask, opts->spec_opts->max_time_mask,
								opts->spec_opts->time_mask_ratio, opts->spec_opts->num_freq_mask, opts->spec_opts->max_freq_mask);
				examples.push_back(spec_example);
    		}
    	}
    }

    return true;
}

bool LmExample::PrepareData(std::vector<Example*> &examples) {
    utt = wordid_reader->Key();
    KALDI_VLOG(3) << "Reading " << utt;

    // get feature
    input_wordids = wordid_reader->Value();

    // split feature
    int32 skip_frames = opts->skip_frames;
    int32 sweep_time = opts->sweep_time;
    int32 lent, cur;

    if (sweep_time>skip_frames) {
		KALDI_WARN << "sweep time for each utterance should less than skip frames (it reset to skip frames)";
		sweep_time = skip_frames;
    }

    examples.clear();

    if (sweep_time <= 1) {
    	examples.push_back(this);
		return true;
    }

    LmExample *example = NULL;
    for (int i = 0; i < sweep_time; i++) {
		example = new LmExample(wordid_reader, opts);
		example->utt = utt;
		example->input_wordids = input_wordids;

		lent = input_wordids.size()/skip_frames;
		lent += input_wordids.size()%skip_frames > i ? 1 : 0;
		example->input_wordids.resize(lent);
		cur = i;
		for (int j = 0; j < example->input_wordids.size(); j++) {
			example->input_wordids[j] = input_wordids[cur];
			cur += skip_frames;
		}

		examples.push_back(example);
    }

    return true;
}

bool SluExample::PrepareData(std::vector<Example*> &examples) {
    utt = wordid_reader->Key();
    KALDI_VLOG(3) << "Reading " << utt;

    // check that we have targets
	if (slot_reader != NULL && !slot_reader->HasKey(utt)) {
	  KALDI_WARN << utt << ", missing slot labels";
	  return false;
	}

	// check that we have targets
	if (intent_reader != NULL && !intent_reader->HasKey(utt)) {
	  KALDI_WARN << utt << ", missing intent labels";
	  return false;
	}

    // get feature
    input_wordids = wordid_reader->Value();
    if (slot_reader != NULL)
    	input_slotids = slot_reader->Value(utt);
    if (intent_reader != NULL)
    	input_intentids = intent_reader->Value(utt);

    // split feature
    int32 skip_frames = opts->skip_frames;
    int32 sweep_time = opts->sweep_time;
    int32 lent, cur;

    if (sweep_time>skip_frames) {
    	KALDI_WARN << "sweep time for each utterance should less than skip frames (it reset to skip frames)";
    	sweep_time = skip_frames;
    }

    examples.resize(sweep_time);

    if (sweep_time <= 1) {
    	examples[0] = this;
    	return true;
    }

    SluExample *example = NULL;
    for (int i = 0; i < sweep_time; i++) {
    	example = new SluExample(opts, wordid_reader, slot_reader, intent_reader);
    	example->utt = utt;

    	lent = input_wordids.size()/skip_frames;
    	lent += input_wordids.size()%skip_frames > i ? 1 : 0;
    	example->input_wordids.resize(lent);
    	example->input_slotids.resize(lent);
    	example->input_intentids.resize(lent);
    	cur = i;
    	for (int j = 0; j < example->input_wordids.size(); j++)
    	{
    		example->input_wordids[j] = input_wordids[cur];
        	example->input_slotids[j] = input_slotids[cur];
        	example->input_intentids[j] = input_intentids[cur];
    		cur += skip_frames;
    	}

    	examples[i] = example;
    }

    return true;
}

bool SeqLabelExample::PrepareData(std::vector<Example*> &examples) {
    utt = wordid_reader->Key();
    KALDI_VLOG(3) << "Reading " << utt;

    // check that we have targets
	if (!label_reader->HasKey(utt)) {
	  KALDI_WARN << utt << ", missing labels";
	  return false;
	}

    // get feature
    input_wordids = wordid_reader->Value();
    input_labelids = label_reader->Value(utt);

    // split feature
    int32 skip_frames = opts->skip_frames;
    int32 sweep_time = opts->sweep_time;
    int32 lent, cur;

    if (sweep_time>skip_frames) {
    	KALDI_WARN << "sweep time for each utterance should less than skip frames (it reset to skip frames)";
    	sweep_time = skip_frames;
    }

    examples.resize(sweep_time);

    if (sweep_time <= 1) {
    	examples[0] = this;
    	return true;
    }

    SeqLabelExample *example = NULL;
    for (int i = 0; i < sweep_time; i++) {
    	example = new SeqLabelExample(opts, wordid_reader, label_reader);
    	example->utt = utt;

    	lent = input_wordids.size()/skip_frames;
    	lent += input_wordids.size()%skip_frames > i ? 1 : 0;
    	example->input_wordids.resize(lent);
    	example->input_labelids.resize(lent);
    	cur = i;
    	for (int j = 0; j < example->input_wordids.size(); j++) {
    		example->input_wordids[j] = input_wordids[cur];
    		example->input_labelids[j] = input_labelids[cur];
    		cur += skip_frames;
    	}

    	examples[i] = example;
    }

    return true;
}

void ExamplesRepository::AcceptExample(
		Example *example) {
  empty_semaphore_.Wait();
  examples_mutex_.Lock();
  examples_.push_back(example);
  examples_mutex_.Unlock();
  full_semaphore_.Signal();
}

void ExamplesRepository::ExamplesDone() {
  for (int32 i = 0; i < buffer_size_; i++)
    empty_semaphore_.Wait();
  examples_mutex_.Lock();
  KALDI_ASSERT(examples_.empty());
  examples_mutex_.Unlock();
  done_ = true;
  full_semaphore_.Signal();
}

Example*
ExamplesRepository::ProvideExample() {
  full_semaphore_.Wait();
  if (done_) {
    KALDI_ASSERT(examples_.empty());
    full_semaphore_.Signal(); // Increment the semaphore so
    // the call by the next thread will not block.
    return NULL; // no examples to return-- all finished.
  } else {
    examples_mutex_.Lock();
    KALDI_ASSERT(!examples_.empty());
    Example *ans = examples_.front();
    examples_.pop_front();
    examples_mutex_.Unlock();
    empty_semaphore_.Signal();
    return ans;
  }
}


} // namespace lm
} // namespace kaldi

