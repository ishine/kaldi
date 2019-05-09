#!/bin/bash

# Copyright 2012-2015  Shanghai Jiao Tong University (Author: Wei Deng)
# Apache 2.0

# This example script trains a DNN on top of fbank features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)


stage=0
cuda_cmd=run.pl
data_fbank=data/train_mix
ali=exp/tri_mix_ali
dir=exp/dnn_relu_mix

#false && \
{
if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  feature_transform=$dir/final.feature_transform
  nnet=$dir/nnet.init
  si_model=$dir/si.nnet
  # --nnet-init $dir/nnet.init
  # --nnet-proto $dir/nnet.proto 
  # --learn-rate 0.00001
  # --feature-transform $feature_transform
  # apply-cmvn-sliding --cmn-window=300 --min-cmn-window=50 --norm-vars=false
  #(tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_asgd_htk.sh --learn-rate 0.00001 --nnet-init $dir/nnet.init --si-model $si_model --kld-scale 0.5 \
    --delta_opts "--delta-order=2" --online true --cmvn-opts "--cmn-window=300 --min-cmn-window=30 --norm-vars=false" --feature-transform $feature_transform \
   	$data_fbank/train_tr90 $data_fbank/train_cv10 data/lang $ali $ali $dir || exit 1;
fi
}


echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

# to see how model conversion to nnet2 works, run run_dnn_convert_nnet2.sh at this point.

