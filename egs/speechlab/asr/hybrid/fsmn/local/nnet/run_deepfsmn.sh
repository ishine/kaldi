#!/bin/bash

# This example script trains a LSTM network on FBANK features.

. ./cmd.sh
. ./path.sh


stage=0
. utils/parse_options.sh || exit 1;

dir=exp/fsmn_baseline
ali=exp/tri3b_gmm_ali
train=data/train/train
cuda_cmd=run.pl
#cuda_cmd="queue.pl -l hostname=chengdu"

#false && \
{
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  # --train-tool "nnet-train-lstm-streams-asgd --num-stream=40 --batch-size=20 --targets-delay=5" \
  feature_transform=$dir/final.feature_transform
  # --feature-transform $feature_transform
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_asgd.sh --learn-rate 0.00001 --nnet-proto $dir/nnet.proto --feature-transform $feature_transform \
      --feat-type plain --splice 5 --cmvn-opts "--norm-means=true --norm-vars=false" \
      --train-opts "--momentum 0.9 --halving-factor 0.5" --skip-frames 3 --skip-inner true --sweep-loop false \
      --train-tool "nnet-train-frmshuff-parallel " \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  # --cmvn-opts "--norm-means=true --norm-vars=false"
  # --splice 0 
  # --splice-left 0 --splice_right 7
  # --feature-transform $feature_transform 
  # --skip-opts "--skip-frames=2"
  # --skip-frames 3 --skip-inner true --sweep-loop false
  # --online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false"
  # --delta-opts "--delta-order=2"
  
fi
}

decode_cmd=run.pl
lang=data/lang_decode
data=/aifs1/users/wd007/asr/baseline_chn_2000h/nn/lstm/s1/data/test
test_set="assist_test  read_test aishell_test ai_mcsntr_test400 keywords_test"

false && \
{
  # Decode (reuse HCLG graph)
  for test in $test_set; do
        for acwt in 0.05;do
        steps/nnet/decode_fsmn.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode_comm/$test"_"$acwt || exit 1;
        done
  done
}


echo Success
exit 0
