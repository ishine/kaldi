#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a LSTM network on FBANK features.
# The LSTM code comes from Yiayu DU, and Wei Li, thanks!

. ./cmd.sh
. ./path.sh


cuda_cmd=run.pl
dir=exp/plstm_student_small
ali=exp/tri_phone_ali
train=data/train/train

#false && \
{
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  si_model=$dir/si.nnet
  feature_transform=$dir/final.feature_transform
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_asgd_htk.sh --learn-rate 0.00005 --nnet-proto $dir/nnet.proto --si-model $si_model --kld-scale 1.0 \
      --feat-type plain --splice 0 --sort-by-len false --cmvn-opts "--norm-means=true --norm-vars=false" \
      --feature-transform $feature_transform --skip-opts "--skip-frames=2 --sweep-time=2" \
      --train-tool "nnet-train-lstm-streams-asgd --num-stream=40 --batch-size=20 --targets-delay=0" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  # --cmvn-opts "--norm-means=true --norm-vars=false"
  # --online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false"
  # --splice 0
  # --train-tool "nnet-train-lstm-streams-asgd --num-stream=40 --batch-size=20 --targets-delay=5 --skip-frames=2" \
  # --train-tool "nnet-train-ctc-parallel  --num-stream=20 --max_frames=25000 --objective-function=\"xent\" " \
  # --delta_opts "--delta-order=2"
  # --feature-transform $feature_transform
}

decode_cmd=run.pl

data=data/test_set
lang=/sgfs/users/wd007/asr/baseline_chn_7000h/sequence/s5/data/lang_smalllm
#ai_mcsnor_evl13jun_v1  ai_mcsntr_evl13mar_v1  ai_mcsntr_evl14jan_v1  ai_mcsntr_evl14mar_v1


opts="--prior-scale=0.0"
false && \
{
  for test in ai_mcsnor_evl13jun_v1  ai_mcsntr_evl13mar_v1  ai_mcsntr_evl14jan_v1  ai_mcsntr_evl14mar_v1;do
	for acwt in 0.05; do
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
    $lang $data/test_lstm_fbank80/$test $dir/decode/newforward/${test}_"$acwt" || exit 1;
  #steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir  \
  #  --nnet-forward-opts "--no-softmax=true --prior-scale=1.0 --time-shift=5" \
  #  $lang $data/test_lstm/$test $dir/decode_time-shift5/${test}_"$acwt" || exit 1;
	done
  done
}

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
