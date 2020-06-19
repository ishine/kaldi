#!/bin/bash

# This script uses weight transfer as a transfer learning method to transfer
# already trained neural net model to a new one with new data. 

# Model preparation: The last layer (output layer) from
# already-trained model is removed and 1 randomly initialized layer
# (output layer) are added to the model.

set -e

# configs for 'chain'
stage=13
train_stage=-3
get_egs_stage=-10
speed_perturb=false
dir=exp/chain/9tdnn_4blstm_65M_10wh_adapt_2epoch # Note: _sp will get added to this if $speed_perturb == true.
src_mdl=exp/chain/9tdnn_4blstm_65M_10wh_adapt_2epoch/src_final_mdl/final.mdl
src_lr_factor=0.25   # The learning-rate factor for transferred layers from source
                     # model. e.g. if 0, the paramters transferred from source model
                     # are fixed.
                     # The learning-rate factor for new added layers is 1.0.

decode_iter=
decode_dir_affix=

# training options
leftmost_questions_truncate=-1
chunk_width=
chunk_left_context=40
chunk_right_context=40
xent_regularize=0.025
self_repair_scale=0.00001
label_delay=0
# decode options
extra_left_context=50
extra_right_context=50

frames_per_chunk=150,100
frames_per_chunk_primary=$(echo $frames_per_chunk | cut -d, -f1)
#frames_per_chunk_primary=10000

remove_egs=false
common_egs_dir=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/exp/chain/9tdnn_4blstm_65M_10wh/egs

affix=
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

if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
dir=${dir}$suffix
train_set=train_sogou_fbank_10w
ali_dir=exp/tri3b_ali
treedir=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/exp/chain/tri4_hunshu_1whoures_tree
lang=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/data/lang_chain_2y_hunshu
mfcc_data=data/train_mfcc

<<!
# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;
!

fbankdir=fbank
if [ $stage -le 8 ]; then 
  # first make fbank features for NN trainging
  cp -r data/local/train data/train_fbank || exit 1;
  cp -r data/local/not_on_screen data/not_on_screen || exit 1;
  cp -r data/local/test8000 data/test8000 || exit 1;
  cp -r data/local/testIOS data/testIOS || exit 1;
  
  # modify conf/fbank.conf to set fbank feature config
  for x in train_fbank not_on_screen test8000 testIOS; do
    steps/make_fbank.sh --nj 40 --cmd "$train_cmd" \
      data/$x exp/make_fbank/$x $fbankdir
    steps/compute_cmvn_stats.sh data/$x exp/make_fbank/$x $fbankdir
    utils/fix_data_dir.sh data/$x
  done
fi

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri3b_ali/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" $mfcc_data \
    data/lang exp/tri3b exp/tri3b_lats_nodup$suffix
  rm exp/tri3b_lats_nodup$suffix/fsts.*.gz # save space
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
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 9000 $mfcc_data $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: Create neural net configs using the xconfig parser for";
  echo " generating new layers, that are specific to new model. These layers ";
  echo " are added to the transferred part of the old network.";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  ## adding the layers for chain branch
  output-layer name=output input=Append(blstm4-forward.rp, blstm4-backward.rp) output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.0

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=Append(blstm4-forward.rp, blstm4-backward.rp) output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.0

EOF
  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --config-dir $dir/configs/

  # Set the learning-rate-factor to be primary_lr_factor for transferred layers "
  # and adding new layers to them.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$src_lr_factor" $src_mdl - \| \
      nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;

fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1200000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 2 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 5 \
    --trainer.optimization.num-jobs-final 8 \
    --trainer.optimization.initial-effective-lrate 0.0004 \
    --trainer.optimization.final-effective-lrate 0.00002 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_chunk \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 300 \
    --feat-dir /public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/data/${train_set} \
    --tree-dir $treedir \
    --lat-dir /public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/exp/tri3b_10w_lats \
    --dir $dir  || exit 1;
fi
<<!
if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_bigG $dir $dir/graph_bigG
fi
!
testset="testC1pen_0215eva_0.3m_sogou testRecordv2-dialog2-represent-long-0.3m_sogou testRecordv2-dialog2-interview-long-1m_sogou testRecordv2-dialog2-meeting-long-2m_sogou"
decode_suff=0528
graph_dir=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/exp/chain/lstm_6j_16k_500h_ld5/graph_0528
if [ $stage -le 15 ]; then
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in $testset; do
#  for decode_set in testNewLong_sogou testNewNoise_sogou testNewSilence_sogou testNewSpeed_sogou; do
       steps/nnet3/decode_sogou.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 8 --cmd "$decode_cmd" $iter_opts \
          --extra-left-context $extra_left_context  \
          --extra-right-context $extra_right_context  \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk "$frames_per_chunk_primary" \
         $graph_dir /public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/data/${decode_set} \
         $dir/decode_${decode_set}_${decode_suff} || exit 1;
  done
fi
wait;
exit 0;
