#!/bin/bash

# f5 is based on the tdnnf config, to see whether the deep structure is fit to fsmn.
set -e

# configs for 'chain'
stage=15
train_stage=-10
get_egs_stage=-10
speed_perturb=false

decode_iter=
decode_nj=50

# training options
frames_per_eg=150,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

suffix=
$speed_perturb && suffix=_sp
dir=exp/chain/dfsmn_f5_based_tdnnf

if ! cuda-compiled; then
 cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

train_set=train_sogou_fbank_7300h
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
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  affine_opts="l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.001 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.001 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.001"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=71 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 input=Append(-1,0,1) dim=1536 
  linear-component name=tdnn1l dim=256 orthonormal-constraint=-1.0 l2-regularize=0.001

  blocksum-layer name=dfsmn1_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-layer name=dfsmn1_inter dim=1536 input=Sum(dfsmn1_projection_pre, tdnn1l) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt1 input=Sum(dfsmn1_inter, Scale(0.66,tdnn1))
  linear-component name=dfsmn1_projection dim=256  input=no-opt1 orthonormal-constraint=-1.0 l2-regularize=0.001
   
  blocksum-layer name=dfsmn2_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-layer name=dfsmn2_inter dim=1536 input=Sum(dfsmn2_projection_pre, dfsmn1_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt2 input=Sum(dfsmn2_inter, Scale(0.66, no-opt1))
  linear-component name=dfsmn2_projection dim=256  input=no-opt2 orthonormal-constraint=-1.0 l2-regularize=0.001
   
  blocksum-layer name=dfsmn3_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-layer name=dfsmn3_inter dim=1536 input=Sum(dfsmn3_projection_pre, dfsmn2_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt3 input=Sum(dfsmn3_inter, Scale( 0.66,no-opt2))
  linear-component name=dfsmn3_projection dim=256  input=no-opt3 orthonormal-constraint=-1.0 l2-regularize=0.001

  blocksum-layer name=dfsmn4_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-layer name=dfsmn4_inter dim=1536 input=Sum(dfsmn4_projection_pre, dfsmn3_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt4 input=Sum(dfsmn4_inter, Scale(0.66,no-opt3))
  linear-component name=dfsmn4_projection dim=256  input=no-opt4 orthonormal-constraint=-1.0 l2-regularize=0.001

  blocksum-layer name=dfsmn5_projection_pre dim=256  input=Append(-2,-1,0,1,2)  
  relu-batchnorm-layer name=dfsmn5_inter dim=1536 input=Sum(dfsmn5_projection_pre, dfsmn4_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt5 input=Sum(dfsmn5_inter, Scale( 0.66,no-opt4))
  linear-component name=dfsmn5_projection dim=256  input=no-opt5 orthonormal-constraint=-1.0 l2-regularize=0.001

  blocksum-layer name=dfsmn6_projection_pre dim=256  input=Append(-2,-1,0,1,2)  
  relu-batchnorm-layer name=dfsmn6_inter dim=1536 input=Sum(dfsmn6_projection_pre, dfsmn5_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt6 input=Sum(dfsmn6_inter, Scale( 0.66,no-opt5))
  linear-component name=dfsmn6_projection dim=256  input=no-opt6 orthonormal-constraint=-1.0 l2-regularize=0.001

  blocksum-layer name=dfsmn7_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-layer name=dfsmn7_inter dim=1536 input=Sum(dfsmn7_projection_pre, dfsmn6_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt7 input=Sum(dfsmn7_inter, Scale( 0.66,no-opt6))
  linear-component name=dfsmn7_projection dim=256  input=no-opt7 orthonormal-constraint=-1.0 l2-regularize=0.001

  blocksum-layer name=dfsmn8_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-layer name=dfsmn8_inter dim=1536 input=Sum(dfsmn8_projection_pre, dfsmn7_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt8 input=Sum(dfsmn8_inter, Scale( 0.66,no-opt7))
  linear-component name=dfsmn8_projection dim=256  input=no-opt8 orthonormal-constraint=-1.0 l2-regularize=0.001

  blocksum-layer name=dfsmn9_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-layer name=dfsmn9_inter dim=1536 input=Sum(dfsmn9_projection_pre, dfsmn8_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt9 input=Sum(dfsmn9_inter, Scale( 0.66,no-opt8))
  linear-component name=dfsmn9_projection dim=256  input=no-opt9 orthonormal-constraint=-1.0 l2-regularize=0.001

  blocksum-layer name=dfsmn10_projection_pre dim=256  input=Append(-3,0,3) 
  relu-batchnorm-layer name=dfsmn10_inter dim=1536 input=Sum(dfsmn10_projection_pre, dfsmn9_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt10 input=Sum(dfsmn10_inter, Scale( 0.66,no-opt9))
  linear-component name=dfsmn10_projection dim=256  input=no-opt10 orthonormal-constraint=-1.0 l2-regularize=0.001

  blocksum-layer name=dfsmn11_projection_pre dim=256  input=Append(-3,0,3) 
  relu-batchnorm-layer name=dfsmn11_inter dim=1536 input=Sum(dfsmn11_projection_pre, dfsmn10_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt11 input=Sum(dfsmn11_inter, Scale( 0.66,no-opt10))
  linear-component name=dfsmn11_projection dim=256  input=no-opt11 orthonormal-constraint=-1.0 l2-regularize=0.001
   
  blocksum-layer name=dfsmn12_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-layer name=dfsmn12_inter dim=1536 input=Sum(dfsmn12_projection_pre, dfsmn11_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt12 input=Sum(dfsmn12_inter, Scale( 0.66,no-opt11))
  linear-component name=dfsmn12_projection dim=256  input=no-opt12 orthonormal-constraint=-1.0 l2-regularize=0.001
   
  blocksum-layer name=dfsmn13_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-layer name=dfsmn13_inter dim=1536 input=Sum(dfsmn13_projection_pre, dfsmn12_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt13 input=Sum(dfsmn13_inter, Scale( 0.66,no-opt12))
  linear-component name=dfsmn13_projection dim=256  input=no-opt13 orthonormal-constraint=-1.0 l2-regularize=0.001
   
  blocksum-layer name=dfsmn14_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-layer name=dfsmn14_inter dim=1536 input=Sum(dfsmn14_projection_pre, dfsmn13_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt14 input=Sum(dfsmn14_inter, Scale( 0.66,no-opt13))
  linear-component name=dfsmn14_projection dim=256  input=no-opt14 orthonormal-constraint=-1.0 l2-regularize=0.001
   
  blocksum-layer name=dfsmn15_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-layer name=dfsmn15_inter dim=1536 input=Sum(dfsmn15_projection_pre, dfsmn14_projection) l2-regularize=0.001 dropout-proportion=0.0 dropout-per-dim-continuous=true
  no-op-component name=no-opt15 input=Sum(dfsmn15_inter, Scale( 0.66,no-opt14))
  linear-component name=dfsmn15_projection dim=256  input=no-opt15 orthonormal-constraint=-1.0 l2-regularize=0.001


  prefinal-layer name=prefinal-chain input=dfsmn15_projection $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=dfsmn15_projection $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
 steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

#    --cmd "queue.pl --config /home/dpovey/queue_conly.conf" \


  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 8 \
    --trainer.optimization.initial-effective-lrate 0.0003 \
    --trainer.optimization.final-effective-lrate 0.00003 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --lat-dir exp/tri3b_lats_nodup \
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
iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in not_on_screen_sogou test8000_sogou testIOS_sogou testset_testND_sogou; do
      (
      steps/nnet3/decode_sogou.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          $graph_dir data/${decode_set} \
          $dir/decode_${decode_set}_${decode_suff} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0;
