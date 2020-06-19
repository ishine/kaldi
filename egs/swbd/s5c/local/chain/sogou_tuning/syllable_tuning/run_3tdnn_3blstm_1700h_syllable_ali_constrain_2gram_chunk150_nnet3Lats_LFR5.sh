#!/bin/bash

# run_tdnn_lstm_1c.sh is like run_tdnn_lstm_1b.sh but using the
# new 'fast-lstm' layer.  Results are slightly improved, plus
# it's faster.  See PR #1243 on github, and issue #1237.
# This used to be called run_tdnn_fastlstm_1b.sh.

set -e

# configs for 'chain'
stage=13
train_stage=-5
get_egs_stage=-10
speed_perturb=false
dir=exp/chain/3tdnn_3blstm_1700h_5s_syllable_ali_constrain_2gram_L55R55_chunk150_nnet3Lats_LFR5 # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_dir_affix=

# training options
leftmost_questions_truncate=-1
chunk_width=150,100
chunk_left_context=55
chunk_right_context=55
xent_regularize=0.025
self_repair_scale=0.00001
label_delay=0
# decode options
extra_left_context=60
extra_right_context=60

frames_per_chunk=150

remove_egs=false
common_egs_dir=

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
train_set=train_sogou_fbank_1700h_5s
ali_dir=exp/tri3b_ali_syllable_length_1700h_5s
treedir=exp/chain/tri5_7000houres_mono_tree_syllable
lang=data/lang_chain_2y
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
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=71 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 input=Append(-4,-3,-2,-1,0,1,2,3,4) dim=625
  relu-renorm-layer name=tdnn2 input=Append(-5,0,5) dim=625 
  relu-renorm-layer name=tdnn3 input=Append(-5,0,5) dim=625 

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmr-layer name=blstm1-forward input=tdnn3 cell-dim=1024 recurrent-projection-dim=192 delay=-5
  fast-lstmr-layer name=blstm1-backward input=tdnn3 cell-dim=1024 recurrent-projection-dim=192 delay=5
  fast-lstmr-layer name=blstm2-forward input=Append(blstm1-forward, blstm1-backward) cell-dim=768 recurrent-projection-dim=192 delay=-5
  fast-lstmr-layer name=blstm2-backward input=Append(blstm1-forward, blstm1-backward) cell-dim=768 recurrent-projection-dim=192 delay=5
  fast-lstmr-layer name=blstm3-forward input=Append(blstm2-forward, blstm2-backward) cell-dim=768 recurrent-projection-dim=192 delay=-5
  fast-lstmr-layer name=blstm3-backward input=Append(blstm2-forward, blstm2-backward) cell-dim=768 recurrent-projection-dim=192 delay=5

  ## adding the layers for chain branch
  output-layer name=output input=Append(blstm3-forward, blstm3-backward) output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.0

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=Append(blstm3-forward, blstm3-backward) output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.0

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.right-tolerance 1 \
    --chain.left-tolerance 1 \
    --chain.frame-subsampling-factor 5 \
    --chain.alignment-subsampling-factor 1 \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1200000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 4 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 8 \
    --trainer.optimization.initial-effective-lrate 0.0013 \
    --trainer.optimization.final-effective-lrate 0.0002 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir /public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/data/${train_set} \
    --tree-dir $treedir \
    --lat-dir /public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/exp/chain/tdnn_lstm_1c_sogoufeat_1700hours_noLD_syllable_ali_constrain_5s_2gram_denfst_LFR5_lats_beam20 \
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
decode_suff=0528
graph_dir=exp/chain/tdnn_lstm_1c_sogoufeat_1700hours_noLD_syllable_ali_constrain_5s_2gram_denfst_nnet3Lats_retrain/graph_0528_syllable
if [ $stage -le 15 ]; then
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in not_on_screen_sogou test8000_sogou testIOS_sogou testset_testND_sogou; do
       steps/nnet3/decode_sogou.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 8 --cmd "$decode_cmd" $iter_opts \
          --extra-left-context $extra_left_context  \
          --extra-right-context $extra_right_context  \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk "$frames_per_chunk" \
         $graph_dir data/${decode_set} \
         $dir/decode_${decode_set}_${decode_suff} || exit 1;
  done
fi
wait;
exit 0;
