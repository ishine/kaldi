#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a LSTM network on FBANK features.
# The LSTM code comes from Yiayu DU, and Wei Li, thanks!

. ./cmd.sh
. ./path.sh

dev=data/test
train=data/train

dev_original=data/test
train_original=data/train

gmm=exp/tri3b

stage=0
. utils/parse_options.sh || exit 1;

false && \
{
# Make the FBANK features
[ ! -e $dev ] && if [ $stage -le 0 ]; then
  # Dev set
  utils/copy_data_dir.sh $dev_original $dev || exit 1; rm $dev/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd" \
     $dev $dev/log $dev/data || exit 1;
  steps/compute_cmvn_stats.sh $dev $dev/log $dev/data || exit 1;
  # Training set
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank_pitch.sh --nj 10 --cmd "$train_cmd -tc 10" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
  # Split the training set
  utils/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $train ${train}_tr90 ${train}_cv10
fi
}

cuda_cmd=run.pl
dir=exp/lstm_baseline
ali=exp/tri_gmm_ali
train=data/train300_lstm/train50

false && \
{
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh --network-type lstm --learn-rate 0.0001 --copy-feats false --nnet-proto $dir/nnet.proto \
      --feat-type plain --splice 0 \
      --train-opts "--momentum 0.9 --halving-factor 0.5" \
      --train-tool "nnet-train-lstm-streams --num-stream=40 --batch-size=20 --targets-delay=5" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  #--cmvn-opts "--norm-means=true --norm-vars=true"
fi
}

decode_cmd=run.pl

data=/sgfs/users/wd007/asr/baseline_chn_50h/sequence/s5/data
lang=/sgfs/users/wd007/asr/baseline_chn_7000h/sequence/s5/data/lang
#ai_mcsnor_evl13jun_v1  ai_mcsntr_evl13mar_v1  ai_mcsntr_evl14jan_v1  ai_mcsntr_evl14mar_v1


opts="--prior-scale=0.0"
#false && \
{
  for test in ai_mcsnor_evl13jun_v1  ai_mcsntr_evl13mar_v1  ai_mcsntr_evl14jan_v1  ai_mcsntr_evl14mar_v1;do
	for acwt in 0.05; do
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
    $lang $data/test_lstm/$test $dir/decode/${test}_"$acwt" || exit 1;
  steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir  \
    --nnet-forward-opts "--no-softmax=true --prior-scale=1.0 --time-shift=5" \
    $lang $data/test_lstm/$test $dir/decode_time-shift5/${test}_"$acwt" || exit 1;
	done
  done
}

acwt=0.1
#false && \
{
  # Decode
  for test in ai_mcsnor_evl13jun_v1;do
      for ITER in 1 3 5 7 9 11 13; do
  	# Decode (reuse HCLG graph)
  	steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config \
	--nnet $dir/nnet/${ITER}.nnet --acwt $acwt --srcdir $dir \
    	$lang data/test/$test $dir/decode/${test}_it${ITER} || exit 1;
      done
  done
}

# TODO : sequence training,

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
