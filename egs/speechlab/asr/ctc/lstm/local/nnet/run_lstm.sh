#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

. ./cmd.sh
. ./path.sh


cuda_cmd=run.pl
dir=exp/cldnn_classic_ce
ali=exp/tri3b_phone_ali
train=data/train/train

#false && \
{
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  feature_transform=$dir/final.feature_transform
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_asgd_htk.sh --learn-rate 0.00005 --nnet-init $dir/nnet.init \
      --feat-type plain --splice-left 5 --splice-right 5 --feature-transform $feature_transform \
      --skip-frames 3 --skip-inner true --online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false" \
      --train-tool "nnet-train-lstm-streams-asgd --batch-size=20 --targets-delay=5 --num-stream=64" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  # --cmvn-opts "--norm-means=true --norm-vars=false"
  # --online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false"
  # --splice 0 --nnet-proto $dir/nnet.proto
  # --train-tool "nnet-train-lstm-streams-asgd --num-stream=40 --batch-size=20 --targets-delay=5 --skip-frames=2" \
  # --train-tool "nnet-train-ctc-parallel  --num-stream=20 --max_frames=25000 --objective-function=\"xent\" " \
  # --delta_opts "--delta-order=2"
  # --feature-transform $feature_transform
  # --nnet-init $dir/nnet.init
  # --nnet-proto $dir/nnet.proto
  # --splice-left 5 --splice-right 5
}

echo Success
exit 0
