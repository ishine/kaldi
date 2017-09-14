#!/bin/bash

# Copyright 2017 (author: Kaituo Xu)

echo "$0 $@"

[ -f path.sh ] && . ./path.sh;
[ -f cmd.sh ] && . ./cmd.sh;

# Begin input feature, output label and decode config 
train=data_fbank/train
dev=data_fbank/dev
test=data_fbank/test

gmm=exp/tri5a # TODO: make it an argument
ali=${gmm}_ali
dev_ali=${gmm}_dev_ali

conf_dir=local/nnet1
decode_dnn_conf=$conf_dir/decode_dnn.config
# End input, output and decode config

# Begin prune config
prune_ratio="0.5,0.5,0.5,0.5,0.5,0"
# End prune config

stage=0
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <dense-dir|old-sparse-dir> <new-sparse-dir>"
  echo ""
  echo "main options (for others, see top of script file)"
  echo " --prune-ratio <ration-list> # e.g.:0.8,0.8,0.8"
  exit 0;
fi

in_dir=$1
out_dir=$2

[ ! -d $out_dir ] && mkdir -p $out_dir

# Step 1: pruning dense\old-sparse DNN to sparse DNN
[ ! -f $in_dir/final.nnet ] && echo "$in_dir/final.nnet does not exist!" && exit 1;
mlp_final=$in_dir/final.nnet
mlp_sparse_init=$out_dir/nnet_sparse_dnn.init
nnet-init-sparse-dnn "--prune-ratio=$prune_ratio" $mlp_final $mlp_sparse_init

# Step 2: train sparse DNN as common nnet1
# set learning rate as 1/10 of dense net
$cuda_cmd $out_dir/log/train_sparse_nnet.log \
  steps/nnet/train.sh \
    --splice 5 --cmvn-opts "--norm-means=true --norm-vars=true" --copy-feats false \
    --nnet-init $mlp_sparse_init \
    --learn-rate 0.00008 --scheduler-opts "--momentum 0.9 --halving-factor 0.5" \
    --train-tool "nnet-train-frmshuff" \
    --train-tool-opts "--minibatch-size=256" \
  $train $dev data/lang $ali $dev_ali $out_dir || exit 1;

# Step 3: decode
steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config $decode_dnn_conf --acwt 0.1 \
  $gmm/graph $test $out_dir/decode_test || exit 1;
