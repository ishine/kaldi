#!/bin/bash

# d5 is based on d1, but we half the learning-rate to 0.0005-0.00005
# Conclusion:
set -e

# configs for 'chain'
affix=
stage=13
train_stage=-10
get_egs_stage=-10
speed_perturb=false
dir=exp/chain/dfsmn_2500h_e9_10fsmn_L9R20_subsample3_noAvg_2048-512_LR0.0008-0.00008_11frms  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=

# training options
num_epochs=6
initial_effective_lrate=0.0008
final_effective_lrate=0.00008
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=8
minibatch_size=64
frames_per_eg=150
remove_egs=false
common_egs_dir=exp/chain/dfsmn_2500h_e9_10fsmn_L9R20_subsample3_noAvg_2048-512_LR0.0008-0.00008_11frms/egs
xent_regularize=0.1

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}${affix:+_$affix}$suffix
train_set=train_sogou_fbank_2500h
ali_dir=exp/tri3b_ali
treedir=exp/chain/tri5_7000houres_tree
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
#local/nnet3/run_ivector_common.sh --stage $stage \
#  --speed-perturb $speed_perturb \
#  --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.002"
  linear_opts="orthonormal-constraint=-1.0 l2-regularize=0.002"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=71 name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  relu-batchnorm-layer name=tdnn1 input=Append(-5,-4,-3,-2,-1,0,1,2,3,4,5) dim=2048 $opts
  linear-component name=tdnn1l dim=512 $linear_opts
  
  blocksum-layer name=dfsmn1_blocksum input=Append(-9,-6,-3,0,3) dim=512
  relu-batchnorm-layer name=dfsmn1_inter dim=2048  input=Sum(dfsmn1_blocksum, tdnn1l) $opts
  linear-component name=dfsmn1_projection dim=512  $linear_opts
  blocksum-layer name=dfsmn2_blocksum input=Append(-9,-6,-3,0) dim=512
  relu-batchnorm-layer name=dfsmn2_inter dim=2048  input=Sum(dfsmn2_blocksum, dfsmn1_blocksum, dfsmn1_projection) $opts
  linear-component name=dfsmn2_projection dim=512  $linear_opts
  blocksum-layer name=dfsmn3_blocksum input=Append(-9,-6,-3,0,3) dim=512
  relu-batchnorm-layer name=dfsmn3_inter dim=2048  input=Sum(dfsmn3_blocksum, dfsmn2_blocksum, dfsmn2_projection) $opts
  linear-component name=dfsmn3_projection dim=512  $linear_opts
  blocksum-layer name=dfsmn4_blocksum input=Append(-9,-6,-3,0) dim=512
  relu-batchnorm-layer name=dfsmn4_inter dim=2048  input=Sum(dfsmn4_blocksum, dfsmn3_blocksum, dfsmn3_projection) $opts
  linear-component name=dfsmn4_projection dim=512  $linear_opts
  blocksum-layer name=dfsmn5_blocksum input=Append(-9,-6,-3,0,3) dim=512
  relu-batchnorm-layer name=dfsmn5_inter dim=2048  input=Sum(dfsmn5_blocksum, dfsmn4_blocksum, dfsmn4_projection) $opts
  linear-component name=dfsmn5_projection dim=512  $linear_opts
  blocksum-layer name=dfsmn6_blocksum input=Append(-9,-6,-3,0) dim=512
  relu-batchnorm-layer name=dfsmn6_inter dim=2048  input=Sum(dfsmn6_blocksum, dfsmn5_blocksum, dfsmn5_projection) $opts
  linear-component name=dfsmn6_projection dim=512  $linear_opts
  blocksum-layer name=dfsmn7_blocksum input=Append(-9,-6,-3,0,3) dim=512
  relu-batchnorm-layer name=dfsmn7_inter dim=2048  input=Sum(dfsmn7_blocksum, dfsmn6_blocksum, dfsmn6_projection) $opts
  linear-component name=dfsmn7_projection dim=512  $linear_opts
  blocksum-layer name=dfsmn8_blocksum input=Append(-9,-6,-3,0) dim=512
  relu-batchnorm-layer name=dfsmn8_inter dim=2048  input=Sum(dfsmn8_blocksum, dfsmn7_blocksum, dfsmn7_projection) $opts
  linear-component name=dfsmn8_projection dim=512  $linear_opts
  blocksum-layer name=dfsmn9_blocksum input=Append(-9,-6,-3,0,3) dim=512
  relu-batchnorm-layer name=dfsmn9_inter dim=2048  input=Sum(dfsmn9_blocksum, dfsmn8_blocksum, dfsmn8_projection) $opts
  linear-component name=dfsmn9_projection dim=512  $linear_opts
  blocksum-layer name=dfsmn10_blocksum input=Append(-9,-6,-3,0) dim=512
  relu-batchnorm-layer name=dfsmn10_inter dim=2048  input=Sum(dfsmn10_blocksum, dfsmn9_blocksum, dfsmn9_projection) $opts
  linear-component name=dfsmn10_projection dim=512  $linear_opts
  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=dfsmn10_projection $opts dim=2048
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  relu-batchnorm-layer name=prefinal-xent input=dfsmn10_projection $opts dim=2048
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --lat-dir exp/tri3b_lats_2500h \
    --dir $dir  || exit 1;

fi
<<!
if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi
!
decode_suff=online
graph_dir=exp/chain/lstm_6j_offline_1536_512_sogoufeat_7000h_ld5/graph_online
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in not_on_screen_sogou test8000_sogou testIOS_sogou testset_testND_sogou ; do
      (
      steps/nnet3/decode_sogou.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          $graph_dir data/${decode_set} $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
      ) &
  done
fi
wait;
exit 0;
