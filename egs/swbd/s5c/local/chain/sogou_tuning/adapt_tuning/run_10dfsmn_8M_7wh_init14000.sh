#!/bin/bash

# d5 is based on d1, but we half the learning-rate to 0.0005-0.00005
# Conclusion:
set -e

# configs for 'chain'
affix=
stage=12
train_stage=-2
get_egs_stage=5
speed_perturb=false
dir=exp/chain/10dfsmn_8M_7wh_adapt_init14000_2epoch  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
src_mdl=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/exp/chain/dfsmn_2wh_8M_f16_10fsmn_L64R14_subsample3_interval_skip_LR0.4_6epoch/14000.mdl
src_lr_factor=0.25


# training options
num_epochs=3
initial_effective_lrate=0.0004
final_effective_lrate=0.000025
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=5
num_jobs_final=8
minibatch_size=64
frames_per_eg=150,100
remove_egs=false
common_egs_dir=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/exp/chain/10dfsmn_18M_7wh_adapt_init30000_3epoch/egs
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
train_set=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/data/train_sogou_fbank_8w
ali_dir=exp/tri3b_ali
treedir=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/exp/chain/tri4_hunshu_1whoures_tree
lang=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/data/lang_chain_2y_hunshu


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
  opts="l2-regularize=0.0005"
  mem_block_opts="max-change=0.5 learning-rate-factor=0.4"
  linear_opts="orthonormal-constraint=-1.0 l2-regularize=0.0005"
  output_opts="l2-regularize=0.0005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=71 name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=1280 $opts
  linear-component name=tdnn1l dim=256 $linear_opts
  
  blocksum-layer name=dfsmn1_blocksum input=Append(-9,-6,-3,0,3) dim=256 $mem_block_opts 
  relu-batchnorm-layer name=dfsmn1_inter dim=1280  input=Sum(dfsmn1_blocksum, tdnn1l) $opts
  linear-component name=dfsmn1_projection dim=256  $linear_opts
  blocksum-layer name=dfsmn2_blocksum input=Append(-9,-6,-3,0) dim=256 $mem_block_opts
  relu-batchnorm-layer name=dfsmn2_inter dim=1280  input=Sum(dfsmn2_blocksum, dfsmn1_projection) $opts
  linear-component name=dfsmn2_projection dim=256  $linear_opts
  blocksum-layer name=dfsmn3_blocksum input=Append(-9,-6,-3,0,3) dim=256 $mem_block_opts 
  relu-batchnorm-layer name=dfsmn3_inter dim=1280  input=Sum(dfsmn3_blocksum, dfsmn1_blocksum, dfsmn2_projection) $opts
  linear-component name=dfsmn3_projection dim=256  $linear_opts
  blocksum-layer name=dfsmn4_blocksum input=Append(-6,-3,0) dim=256 $mem_block_opts
  relu-batchnorm-layer name=dfsmn4_inter dim=1280  input=Sum(dfsmn4_blocksum, dfsmn2_blocksum, dfsmn3_projection) $opts
  linear-component name=dfsmn4_projection dim=256  $linear_opts
  blocksum-layer name=dfsmn5_blocksum input=Append(-3,0,3) dim=256 $mem_block_opts
  relu-batchnorm-layer name=dfsmn5_inter dim=1280  input=Sum(dfsmn5_blocksum, dfsmn3_blocksum, dfsmn4_projection) $opts
  linear-component name=dfsmn5_projection dim=256  $linear_opts
  blocksum-layer name=dfsmn6_blocksum input=Append(-6,-3,0) dim=256 $mem_block_opts
  relu-batchnorm-layer name=dfsmn6_inter dim=1280  input=Sum(dfsmn6_blocksum, dfsmn4_blocksum, dfsmn5_projection) $opts
  linear-component name=dfsmn6_projection dim=256  $linear_opts
  blocksum-layer name=dfsmn7_blocksum input=Append(-3,0,3) dim=256 $mem_block_opts
  relu-batchnorm-layer name=dfsmn7_inter dim=1280  input=Sum(dfsmn7_blocksum, dfsmn5_blocksum, dfsmn6_projection) $opts
  linear-component name=dfsmn7_projection dim=256  $linear_opts
  blocksum-layer name=dfsmn8_blocksum input=Append(-6,-3,0) dim=256 $mem_block_opts
  relu-batchnorm-layer name=dfsmn8_inter dim=1280  input=Sum(dfsmn8_blocksum, dfsmn6_blocksum, dfsmn7_projection) $opts
  linear-component name=dfsmn8_projection dim=256  $linear_opts
  blocksum-layer name=dfsmn9_blocksum input=Append(-6,-3,0) dim=256 $mem_block_opts
  relu-batchnorm-layer name=dfsmn9_inter dim=1280  input=Sum(dfsmn9_blocksum, dfsmn7_blocksum, dfsmn8_projection) $opts
  linear-component name=dfsmn9_projection dim=256  $linear_opts
  blocksum-layer name=dfsmn10_blocksum input=Append(-6,-3,0) dim=256 $mem_block_opts
  relu-batchnorm-layer name=dfsmn10_inter dim=1280  input=Sum(dfsmn10_blocksum, dfsmn8_blocksum, dfsmn9_projection) $opts
  linear-component name=dfsmn10_projection dim=256  $linear_opts
  
  ## fully connected layers
  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=dfsmn10_projection $opts big-dim=1280 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=dfsmn10_projection $opts big-dim=1280 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF

  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --config-dir $dir/configs/

  # Set the learning-rate-factor to be src_lr_factor for transferred layers "
  # and adding new layers to them.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$src_lr_factor" $src_mdl - \| \
    nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
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
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 500 \
    --feat-dir ${train_set} \
    --tree-dir $treedir \
    --lat-dir exp/tri3b_8w_offline_lats \
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
decode_suff=0528
graph_dir=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/exp/chain/lstm_6j_16k_500h_ld5/graph_translate2_hunshu_20191122
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in not_on_screen_sogou test8000_sogou testIOS_sogou testset_testND_sogou ; do
      steps/nnet3/decode_sogou.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --use-gpu false \
          --nj 8 --cmd "$decode_cmd" $iter_opts \
          $graph_dir data/${decode_set} $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
  done
fi
wait;
exit 0;
