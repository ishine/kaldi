#!/bin/bash

# Copyright 2019 Desh Raj

# This script performs multi-task learning by jointly
# training an x-vector DNN and an ASR DNN. The MTL setup
# is based on babel_multilingual. The ASR DNN is the same
# as that in tdnn_1h, and the x-vector part is borrowed
# from the SRE16 x-vector setup.

echo "$0 $@"  # Print the command line for logging
. ./cmd.sh
set -e

remove_egs=false
cmd=queue.pl
srand=0
stage=0
train_stage=0
decode_stage=-10
num_jobs_initial=2
num_jobs_final=8
speed_perturb=true
decode_nj=10
train_set=train_clean_5
test_sets=dev_clean_2
gmm=tri3b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1i   # affix for the TDNN directory name
tree_affix=
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
dropout_schedule='0,0@0.20,0.3@0.50,0'
common_egs_dir=
xent_regularize=0.1

# training options
remove_egs=true

#decode options
test_online_decoding=true  # if true, it will run the last decoding stage.

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
egs_dir=${dir}/egs
egs_dir_xvec=${dir}/egs_xvec
egs_dir_final=${dir}/egs_final
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

echo "$0 $@"  # Print the command line for logging
if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $lores_train_data_dir/feats.scp \
    $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 1 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/${train_set}_sp || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 2 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 3 ]; then
  # Create high-resolution MFCC features (with 40 cepstra instead of 13).
  # this shows how you can split across multiple file-systems.
  echo "$0: creating high-resolution MFCC features"
  mfccdir=data/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/fs0{1,2}/$USER/kaldi-data/mfcc/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi

  for datadir in ${train_set}_sp ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires || exit 1;

  for datadir in ${train_set}_sp ${test_sets}; do
    steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" --write_utt2num_frames true \
      data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 5 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 6 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi

num_pdfs=$(awk '{print $2}' $train_data_dir/utt2spk | sort | uniq -c | wc -l)
num_train_utt=$(awk '{print $2}' $train_data_dir/utt2spk | wc -l)
num_repeat=$(( num_train_utt/num_pdfs ))
# Now we create the xvector nnet examples using sid/nnet3/xvector/get_egs.sh.
# The argument --num-repeats is related to the number of times a speaker
# repeats per archive.  If it seems like you're getting too many archives
# (e.g., more than 200) try increasing the --frames-per-iter option.  The
# arguments --min-frames-per-chunk and --max-frames-per-chunk specify the
# minimum and maximum length (in terms of number of frames) of the features
# in the examples.
#
# To make sense of the egs script, it may be necessary to put an "exit 1"
# command immediately after stage 3.  Then, inspect
# exp/<your-dir>/egs/temp/ranges.* . The ranges files specify the examples that
# will be created, and which archives they will be stored in.  Each line of
# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23

# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  You might
# need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
# number of examples per archive.
if [ $stage -le 7 ]; then
  echo "$0: Getting neural network training egs";
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir_xvec/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,04,05,06}/$USER/kaldi-data/egs/mini_librispeech/s5/xvector-$(date +'%m_%d_%H_%M')/$egs_dir_xvec/storage $egs_dir_xvec/storage
  fi
  local/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 1000000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk 50 \
    --max-frames-per-chunk 200 \
    --num-diagnostic-archives 3 \
    "${train_data_dir}" ${egs_dir_xvec}
fi

if [ $stage -le 8 ]; then
  echo "$0: creating MTL-based neural net configs using the xconfig parser";
  mkdir -p $dir/configs
  
  num_targets_xvec=$(wc -w $egs_dir_xvec/pdf2num | awk '{print $1}')
  num_targets_asr=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  # This chunk-size corresponds to the maximum number of frames the
  # stats layer is able to pool over.  In this script, it corresponds
  # to 100 seconds.  If the input recording is greater than 100 seconds,
  # we will compute multiple xvectors from the same recording and average
  # to produce the final xvector.
  max_chunk_size=10000

  # The smallest number of frames we're comfortable computing an xvector from.
  # Note that the hard minimum is given by the left and right context of the
  # frame-level layers.
  min_chunk_size=25
  
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  tdnn_opts="l2-regularize=0.03 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.03"
  output_opts="l2-regularize=0.015"
  
  cat <<EOF > $dir/configs/network.xconfig
  
  input dim=40 name=input
  relu-batchnorm-layer name=xvec.tdnn1 input=Append(-2,-1,0,1,2) dim=512
  relu-batchnorm-layer name=xvec.tdnn2 input=Append(-2,0,2) dim=512
  relu-batchnorm-layer name=xvec.tdnn3 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=xvec.tdnn4 dim=512
  relu-batchnorm-layer name=xvec.tdnn5 dim=1500
  
  # The stats pooling layer. Layers after this are segment-level.
  # In the config below, the first and last argument (0, and ${max_chunk_size})
  # means that we pool over an input segment starting at frame 0
  # and ending at frame ${max_chunk_size} or earlier.  The other arguments (1:1)
  # mean that no subsampling is performed.
  stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size})
  # This is where we usually extract the embedding (aka xvector) from.
  relu-batchnorm-layer name=xvec.tdnn6 dim=512 input=stats
  # This is where another layer the embedding could be extracted
  # from, but usually the previous one works better.
  relu-batchnorm-layer name=xvec.tdnn7 dim=512
  output-layer name=xvec.output include-log-softmax=true dim=${num_targets_xvec} 
  
  affine-layer name=lda input=Append(input@-1,input,input@1,xvec.tdnn7) dim=512

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=768
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output include-log-softmax=false dim=$num_targets_asr $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output-xent dim=$num_targets_asr learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig \
    --config-dir $dir/configs/ 
fi

# Get values for $model_left_context, $model_right_context
. $dir/configs/vars

left_context=$model_left_context
right_context=$model_right_context
frame_subsampling_factor=1

egs_left_context=$(perl -e "print (int($left_context + $frame_subsampling_factor / 2))")
egs_right_context=$(perl -e "print (int($right_context + $frame_subsampling_factor / 2))")

phone_lm_scales="1"

if [ $stage -le 9 ]; then
  echo "$0: compute {den,normalization}.fst using weighted phone LM."
  steps/nnet3/chain/make_weighted_den_fst.sh --cmd "$train_cmd" \
    --num-repeats $phone_lm_scales \
    --lm-opts '--num-extra-lm-states=200' \
    $tree_dir $dir || exit 1;
fi

if [ $stage -le 10 ]; then
  echo "$0: Generates egs for the ASR training."
  . $dir/configs/vars || exit 1;
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,04,05,06}/$USER/kaldi-data/egs/mini_librispeech/s5/asr-$(date +'%m_%d_%H_%M')/$egs_dir/storage $egs_dir/storage
  fi
  steps/nnet3/chain/get_egs.sh --cmd "$train_cmd" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --left-context $model_left_context --right-context $model_right_context \
    --generate-egs-scp true \
    "${train_data_dir}" $dir $lat_dir $egs_dir
  
fi
exit 1
if [ $stage -le 11 ] && [ ! -z $egs_dir_final ]; then
  echo "$0: Generate final egs dir by combining x-vector"
  echo "and ASR training egs."
  steps/nnet3/chain/mtl/combine_egs.sh --cmd "$decode_cmd" \
    ${egs_dir} ${egs_dir_xvec} ${egs_dir_final} || exit 1;
fi
exit 1
if [ $stage -le 12 ]; then
  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --trainer.samples-per-iter=400000 \
    --trainer.max-param-change=2.0 \
    --trainer.srand=$srand \
    --feat-dir ${multi_data_dirs[0]} \
    --feat.online-ivector-dir "$common_ivec_dir" \
    --egs.dir $egs_dir_final \
    --use-dense-targets false \
    --targets-scp ${multi_ali_dirs[0]} \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --use-gpu true \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  for lang_index in `seq 0 $[$num_langs-1]`;do
    lang_dir=$dir/${lang_list[$lang_index]}
    mkdir -p  $lang_dir
    echo "$0: rename output name for each lang to 'output' and "
    echo "add transition model."
    nnet3-copy --edits="rename-node old-name=output-$lang_index new-name=output" \
      $dir/final.raw - | \
      nnet3-am-init ${multi_ali_dirs[$lang_index]}/final.mdl - \
      $lang_dir/final.mdl || exit 1;
    cp $dir/cmvn_opts $lang_dir/cmvn_opts || exit 1;
    echo "$0: compute average posterior and readjust priors for language ${lang_list[$lang_index]}."
    steps/nnet3/adjust_priors.sh --cmd "$decode_cmd" \
      --use-gpu true \
      --iter final --use-raw-nnet false --use-gpu true \
      $lang_dir ${multi_egs_dirs[$lang_index]} || exit 1;
  done
fi

# decoding different languages
if [ $stage -le 13 ]; then
  num_decode_lang=${#decode_lang_list[@]}
  for lang_index in `seq 0 $[$num_decode_lang-1]`; do
    if [ ! -f $dir/${decode_lang_list[$lang_index]}/decode_dev10h.pem/.done ]; then
      echo "Decoding lang ${decode_lang_list[$lang_index]} using multilingual hybrid model $dir"
      local/nnet3/run_decode_lang.sh --use-ivector $use_ivector \
        --use-pitch-ivector $use_pitch_ivector --iter final_adj \
        --nnet3-affix "$nnet3_affix" \
        ${decode_lang_list[$lang_index]} $dir || exit 1;
      touch $dir/${decode_lang_list[$lang_index]}/decode_dev10h.pem/.done
    fi
  done
fi
