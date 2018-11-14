#!/bin/bash

set -o pipefail
set -e
. cmd.sh


stage=1
use_gpu=true  # for training

frames_per_chunk=150
extra_left_context=50
extra_right_context=0
extra_left_context_initial=0
extra_right_context_final=0

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

srcdir=exp/chain/lstm_6j_online_49M_7000h_chunk_100_150_ld5
train_data_dir=data/train_sogou_fbank_500h
online_ivector_dir=
lang=data/lang

[ ! -z "$frames_per_chunk" ] && context_opts="$context_opts --frames-per-chunk $frames_per_chunk"
[ ! -z "$extra_left_context" ] && context_opts="$context_opts --extra-left-context $extra_left_context"
[ ! -z "$extra_right_context" ] && context_opts="$context_opts --extra-right-context $extra_right_context"
[ ! -z "$extra_left_context_initial" ] && context_opts="$context_opts --extra-left-context-initial $extra_left_context_initial"
[ ! -z "$extra_right_context_final" ] && context_opts="$context_opts --extra-right-context-final $extra_right_context_final"

if [ ! -f ${srcdir}/final.mdl ]; then
  echo "$0: expected ${srcdir}/final.mdl to exist; first run run_tdnn.sh or run_lstm.sh"
  exit 1;
fi

frame_subsampling_opt=
frame_subsampling_factor=1
if [ -f $srcdir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $srcdir/frame_subsampling_factor)
  frame_subsampling_opt="--frame-subsampling-factor $(cat $srcdir/frame_subsampling_factor)"
fi

if [ $stage -le 1 ]; then
  # hardcode no-GPU for alignment, although you could use GPU [you wouldn't
  # get excellent GPU utilization though.]
  nj=30 # have a high number of jobs because this could take a while, and we might
         # have some stragglers.
  steps/nnet3/align.sh  --cmd "$decode_cmd" --use-gpu true \
    $context_opts \
    --scale-opts "--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0" \
    --nj $nj $train_data_dir $lang $srcdir ${train_data_dir}_ali
fi

