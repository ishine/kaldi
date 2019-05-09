#!/bin/bash

. ./cmd.sh
. ./path.sh


stage=0
. utils/parse_options.sh || exit 1;

dir=exp/plstm_noclass
train=data/train/train
cuda_cmd=run.pl
lang=data/lang
#cuda_cmd="queue.pl -l hostname=chengdu"

#false && \
{
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  # --train-tool "nnet-train-lstm-streams-asgd --num-stream=40 --batch-size=20 --targets-delay=5" \
  # --cmvn-opts "--norm-means=true --norm-vars=false"
  # --objective-function=cbxent
  # --class-boundary $lang/class_boundary.txt
  feature_transform=$dir/final.feature_transform
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_asgd.sh --learn-rate 0.002 --nnet-proto $dir/nnet.proto --num-threads 1 \
      --train-opts "--halving-factor 0.5" --momentum 0.0 \
      --train-tool "lm-train-lstm-parallel --global-momentum=0.5 --num-stream=40 --batch-size=15 --targets-delay=0 --objective-function=xent" \
    ${train}_tr90 ${train}_cv10 data/lang $dir || exit 1;

  # --cmvn-opts "--norm-means=true --norm-vars=true"
  # --splice 0
  # --splice-left 0 --splice_right 7
  # --feature-transform $feature_transform 
  # --skip-opts "--skip-frames=2"
fi
}

echo Success
exit 0
