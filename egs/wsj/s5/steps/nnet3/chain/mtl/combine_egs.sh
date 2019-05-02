#!/bin/bash

# Copyright 2019     Desh Raj
# Apache 2.0

# This script combines training egs from ASR egs dir
# and x-vector egs dir. It is based on the multilingual
# combine_egs.sh script
# This script produces 2 sets of files --
# egs.*.scp, egs.output.*.ark
#
# egs.*.scp are the SCP files of the training examples.
# egs.output.*.ark map from the key of the example to the name of
# the output-node in the neural net for that specific task, e.g.
# 'output' or 'xvec.output'.
#
# Begin configuration section.
cmd=run.pl
block_size=256          # This is the number of consecutive egs that we take from
                        # each source, and it only affects the locality of disk
                        # access.
stage=0

shift 1
args=("$@")
egs_dir=${args[1]}
egs_dir_xvec=${[2]}
egs_dir_final=${[3]}

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 11 ]; then
  cat <<EOF
  This script generates examples for multitask training of neural network
  using separate input egs dir per task as input. Currently, it supports
  training ASR along with x-vectors.
  See top of the script for details.

  Usage: $0 [opts] <egs-dir-asr> <egs-dir-xvec> <egs-dir-final>
   e.g.: $0 [opts] exp/chain/tdnn_1a/egs exp/chain/tdnn_1a/egs_xvec exp/chain/tdnn_1a/egs_final

  Options:
      --cmd (utils/run.pl|utils/queue.pl <queue opts>)  # how to run jobs.
      --block-size <int|512>      # it is the number of consecutive egs that we take from 
                                  # each source, and it only affects the locality of disk 
                                  # access. This does not have to be the actual minibatch size
EOF
  exit 1;
fi

mkdir -p $egs_dir_final
mkdir -p $egs_dir_final/info

required="egs.scp combine.scp train_diagnostic.scp valid_diagnostic.scp"
train_scp_list=
train_diagnostic_scp_list=
valid_diagnostic_scp_list=
combine_scp_list=

# read paramter from $egs_dir/info and cmvn_opts
# to write in egs_dir_final.
check_params="info/feat_dim info/left_context info/right_context info/left_context_initial info/right_context_final cmvn_opts"

for param in $check_params info/frames_per_eg; do
  cat ${egs_dir}/$param > ${egs_dir_final}/$param || exit 1;
done

tot_num_archives=0

for dir in ($egs_dir $egs_dir_xvec);do
  for f in $required; do
    if [ ! -f ${dir}/$f ]; then
      echo "$0: no such file ${dir}/$f." && exit 1;
    fi
  done
  num_archives=$(cat ${dir}/info/num_archives)
  tot_num_archives=$[tot_num_archives+num_archives]
  train_scp_list="$train_scp_list ${dir}/egs.scp"
  train_diagnostic_scp_list="$train_diagnostic_scp_list ${dir}/train_diagnostic.scp"
  valid_diagnostic_scp_list="$valid_diagnostic_scp_list ${dir}/valid_diagnostic.scp"
  combine_scp_list="$combine_scp_list ${dir}/combine.scp"
done

if [ $stage -le 12 ]; then
  echo "$0: allocating examples for training."
  # Generate egs.*.scp for multilingual setup.
  $cmd $egs_dir_final/log/allocate_egs_train.log \
    steps/nnet3/chain/mtl/allocate_egs.py $egs_opt \
      --num-archives $tot_num_archives \
      --block-size $block_size \
      $train_scp_list $egs_dir_final || exit 1;
fi

if [ $stage -le 13 ]; then
  echo "$0: combine combine.scp examples in $egs_dir_final/combine.scp."
  # Generate combine.scp for MTL setup.
  $cmd $egs_dir_final/log/allocate_egs_combine.log \
    steps/nnet3/chain/mtl/allocate_egs.py $egs_opt \
      --num-archives 1 \
      --block-size $block_size \
      --egs-prefix "combine." \
      $combine_scp_list $egs_dir_final || exit 1;

  echo "$0: combine train_diagnostic.scp examples in $egs_dir_final/train_diagnostic.scp."
  # Generate train_diagnostic.scp for MTL setup.
  $cmd $egs_dir_final/log/allocate_egs_train_diagnostic.log \
    steps/nnet3/chain/mtl/allocate_egs.py $egs_opt \
      --num-archives 1 \
      --block-size $block_size \
      --egs-prefix "train_diagnostic." \
      $train_diagnostic_scp_list $egs_dir_final || exit 1;


  echo "$0: combine valid_diagnostic.scp examples in $egs_dir_final/valid_diagnostic.scp."
  # Generate valid_diagnostic.scp for MTL setup.
  $cmd $egs_dir_final/log/allocate_egs_valid_diagnostic.log \
    steps/nnet3/chain/mtl/allocate_egs.py $egs_opt \
      --num-archives 1 \
      --block-size $block_size \
      --egs-prefix "valid_diagnostic." \
      $valid_diagnostic_scp_list $egs_dir_final || exit 1;

fi
for egs_type in combine train_diagnostic valid_diagnostic; do
  mv $egs_dir_final/${egs_type}.output.1.ark $egs_dir_final/${egs_type}.output.ark || exit 1;
  mv $egs_dir_final/${egs_type}.1.scp $egs_dir_final/${egs_type}.scp || exit 1;
done
mv $egs_dir_final/info/egs.num_archives $egs_dir_final/info/num_archives || exit 1;
echo "$0: Finished preparing MTL training example."
