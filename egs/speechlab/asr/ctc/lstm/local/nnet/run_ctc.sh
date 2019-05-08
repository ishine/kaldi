#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a LSTM network on FBANK features.
# The LSTM code comes from Yiayu DU, and Wei Li, thanks!

. ./cmd.sh
. ./path.sh

stage=0
. utils/parse_options.sh || exit 1;

cuda_cmd=run.pl
dir=exp/cldnn_classic_ctc
ali=exp/tri3b_ctc_ali
train=data/train/trainctc
#false && \
{
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  nnet=$dir/nnet.init
  # --nnet-proto $dir/nnet.proto
  # --nnet-init $nnet
  # --cmvn-opts "--norm-means=true --norm-vars=false"
  feature_transform=$dir/final.feature_transform
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_ctc.sh --network-type lstm --sort-by-len false --learn-rate 0.000002 --nnet-init $nnet \
      --feat-type plain --splice 0  --online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false" \
      --feature-transform $feature_transform --skip-frames 3 --skip-inner true \
      --train-tool "nnet-train-ctc-parallel --num-stream=15 --max-frames=15000 --objective-function=ctc " \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  #--online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false"
  #--delta_opts "--delta-order=2" --batch-size=60
  # --splice 0  --dump-time=500 --batch-size=120
fi
}


echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
