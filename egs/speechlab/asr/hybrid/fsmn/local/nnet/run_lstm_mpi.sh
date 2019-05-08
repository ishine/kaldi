#!/bin/bash

# This example script trains a LSTM network on FBANK features.
# The LSTM code comes from Yiayu DU, and Wei Li, thanks!

. ./cmd.sh
. ./path.sh


stage=0
. utils/parse_options.sh || exit 1;

dir=exp/plstm_baseline1
ali=exp/tri1_gmm_ali_aug
train=data/train/train
cuda_cmd=run.pl

#false && \
{
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  # --train-tool "nnet-train-lstm-streams-asgd --num-stream=40 --batch-size=20 --targets-delay=5" \
  # --nnet-proto $dir/nnet.proto \
  # --feature-transform $dir/final.feature_transform \
  nnet=$dir/nnet.init
  feature_transform=$dir/final.feature_transform
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_mpi.sh --network-type lstm --learn-rate 0.00005 --nnet-proto $dir/nnet.proto \
      --splice 0 --online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false" \
      --train-opts "--momentum 0.9 --halving-factor 0.5" --skip-frames 3 --skip-inner true --sweep-loop false --feature-transform $feature_transform \
      --train-tool "mpiexec -n 8 -f hostfile -wdir . nnet-train-lstm-streams-parallel-mpi --global-momentum=0.875 --num-stream=40 --batch-size=20 --targets-delay=5" \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  #--cmvn-opts "--norm-means=true --norm-vars=true"
  # --nnet-init $nnet
  # --splice-left 0 --splice_right 7
  # --feature-transform $feature_transform 
  # --skip-opts "--skip-frames=2"
  # --skip-frames 2 --skip-inner true
  # --online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false"

fi
}

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
