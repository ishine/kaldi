#!/bin/bash

# Created on 2018-04-08
# Author: Kaituo Xu
# Function: Decode nnet1.

[ -f cmd.sh ] && . ./cmd.sh
[ -f path.sh ] && . ./path.sh

if [ $# -lt 2 ]; then
  echo "Usage: $0 <exp-dir> <test-data-dir>";
  echo "e.g.: $0 exp/lstm data/test";
  exit 1;
fi

dir=$1  # exp/lstm4x1024x512-lr0.00004-mb128
test_original=$2  # data/test_rvb

base=`basename $test_original`
test=data_fbank/$base

[ ! -d $dir ] && echo "$dir does not exist!" && exit 1;
[ ! -d $test_original ] && echo "$dir does not exist!" && exit 1;

gmm=exp/tri5a # TODO: make it a argument
conf_dir=local/nnet1/
fbank_conf=$conf_dir/fbank.conf
decode_dnn_conf=$conf_dir/decode_dnn.config

stage=0
. utils/parse_options.sh || exit 1;

# Step 1: Make the FBANK features
[ ! -e $test ] && if [ $stage -le 0 ]; then
  for data in test; do 
    data_fbank=`eval echo '$'$data`
    data_original=`eval echo '$'${data}_original`
    utils/copy_data_dir.sh $data_original $data_fbank || exit 1; rm $data_fbank/{cmvn,feats}.scp
    steps/make_fbank.sh --nj 10 --cmd "$train_cmd" --fbank-config $fbank_conf \
        $data_fbank $data_fbank/log $data_fbank/data || exit 1;
    steps/compute_cmvn_stats.sh $data_fbank $data_fbank/log $data_fbank/data || exit 1;
  done
fi

# Step 2: Decode
if [ $stage -le 1 ]; then
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config $decode_dnn_conf --acwt 0.1 \
    $gmm/graph $test $dir/decode_$base || exit 1;
fi
