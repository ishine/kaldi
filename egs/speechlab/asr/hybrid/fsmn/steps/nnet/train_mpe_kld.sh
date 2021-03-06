#!/bin/bash
# Copyright 2013-2014  Brno University of Technology (Author: Karel Vesely)  
# Apache 2.0.

# Sequence-discriminative MPE/sMBR training of DNN.
# 4 iterations (by default) of Stochastic Gradient Descent with per-utterance updates.
# We select between MPE/sMBR optimization by '--do-smbr <bool>' option.

# For the numerator we have a fixed alignment rather than a lattice--
# this actually follows from the way lattices are defined in Kaldi, which
# is to have a single path for each word (output-symbol) sequence.


# Begin configuration section.
cmd=run.pl
num_iters=4
acwt=0.1
lmwt=1.0
learn_rate=0.00001
halving_factor=1.0 #ie. disable halving
do_smbr=true
exclude_silphones=true # exclude silphones from approximate accuracy computation
unkphonelist= # exclude unkphones from approximate accuracy computation (overrides exclude_silphones)
one_silence_class=true # true : reduce insertions in sMBR/MPE FW/BW, more stable training,
verbose=1

seed=777    # seed value used for training data shuffling
skip_cuda_check=false
# End configuration section

num_threads=4
frame_limit=10000
num_stream=10
lstm_opts=" --frame-limit=10000 --num-stream=10 --dump-time=250 --batch-size=60 "
skip_opts=
online=false
skip_frames_opts=" --sweep-frames=0"
criterion=smbr
si_model=
kld_scale=0.3


echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: steps/$0 <data> <lang> <srcdir> <ali> <denlats> <exp>"
  echo " e.g.: steps/$0 data/train_all data/lang exp/tri3b_dnn exp/tri3b_dnn_ali exp/tri3b_dnn_denlats exp/tri3b_dnn_smbr"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --num-iters <N>                                  # number of iterations to run"
  echo "  --acwt <float>                                   # acoustic score scaling"
  echo "  --lmwt <float>                                   # linguistic score scaling"
  echo "  --learn-rate <float>                             # learning rate for NN training"
  echo "  --do-smbr <bool>                                 # do sMBR training, otherwise MPE"
  
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
alidir=$4
denlatdir=$5
dir=$6

for f in $data/feats.scp $alidir/{tree,final.mdl} $denlatdir/lat.scp $srcdir/{final.nnet,final.feature_transform}; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# check if CUDA is compiled in,
if ! $skip_cuda_check; then
  cuda-compiled || { echo 'CUDA was not compiled in, skipping! Check src/kaldi.mk and src/configure' && exit 1; }
fi

mkdir -p $dir/log

cp $alidir/{final.mdl,tree} $dir

silphonelist=`cat $lang/phones/silence.csl` || exit 1;
#silphonelist=`cat $lang/silence.csl`; 

#Get the files we will need
nnet=$srcdir/$(readlink $srcdir/final.nnet || echo final.nnet);
[ -z "$nnet" ] && echo "Error nnet '$nnet' does not exist!" && exit 1;
cp $nnet $dir/0.nnet; 
nnet=$dir/0.nnet

class_frame_counts=$srcdir/ali_train_pdf.counts
[ -z "$class_frame_counts" ] && echo "Error class_frame_counts '$class_frame_counts' does not exist!" && exit 1;
cp $srcdir/ali_train_pdf.counts $dir

feature_transform=$srcdir/final.feature_transform
if [ ! -f $feature_transform ]; then
  echo "Missing feature_transform '$feature_transform'"
  exit 1
fi
cp $feature_transform $dir/final.feature_transform

model=$dir/final.mdl
[ -z "$model" ] && echo "Error transition model '$model' does not exist!" && exit 1;

#enable/disable silphones from MPE training
mpe_silphones_arg= #empty
$exclude_silphones && mpe_silphones_arg="--silence-phones=$silphonelist" # all silphones
[ ! -z $unkphonelist ] && mpe_silphones_arg="--silence-phones=$unkphonelist" # unk only


# Shuffle the feature list to make the GD stochastic!
# By shuffling features, we have to use lattices with random access (indexed by .scp file).
#cat $data/feats.scp | utils/shuffle_list.pl --srand $seed > $dir/train.scp
cp $data/feats.scp $dir/train.scp

###
### PREPARE FEATURE EXTRACTION PIPELINE
###
# import config,
cmvn_opts=
delta_opts=
D=$srcdir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
[ -e $D/skip_opts ] && skip_opts=$(cat $D/skip_opts)
[ -e $D/online ] && online=$(cat $D/online)
#
# Create the feature stream,
feats="ark,o:copy-feats scp:$dir/train.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $data/cmvn.scp ] && echo "$0: Missing $data/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" -a "$online" == "false" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
[ ! -z "$cmvn_opts" -a "$online" == "true" ] && feats="$feats apply-cmvn-sliding $cmvn_opts ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# skip-frames (optional),
[ ! -z "$skip_opts" ] && lstm_opts="$lstm_opts $skip_opts $skip_frames_opts"

#
# Record the setup,
[ ! -z "$cmvn_opts" ] && echo $cmvn_opts >$dir/cmvn_opts
[ ! -z "$delta_opts" ] && echo $delta_opts >$dir/delta_opts
[ ! -z "$skip_opts" ] && echo "$skip_opts" >$dir/skip_opts
[ ! -z "$online" ] && echo "$online" >$dir/online
###
###
###


###
### Prepare the alignments
### 
# Assuming all alignments will fit into memory
#ali="ark:gunzip -c $alidir/ali.*.gz |"
ali="scp:$alidir/ali.scp"
[ -z $si_model ] && si_model=$nnet


###
### Prepare the lattices
###
# The lattices are indexed by SCP (they are not gziped because of the random access in SGD)
lats="scp:$denlatdir/lat.scp"


# Run several iterations of the MPE/sMBR training
cur_mdl=$nnet
x=1
while [ $x -le $num_iters ]; do
  echo "Pass $x (learnrate $learn_rate)"
  if [ -f $dir/$x.nnet ]; then
    echo "Skipped, file $dir/$x.nnet exists"
  else
    #train
    # --do-smbr=$do_smbr \
    $cmd $dir/log/mpe.$x.log \
     nnet-train-sequential-parallel $lstm_opts \
       --kld-scale=$kld_scale \
       --si-model=$si_model \
       --feature-transform=$feature_transform \
       --class-frame-counts=$class_frame_counts \
       --acoustic-scale=$acwt \
       --lm-scale=$lmwt \
       --criterion=$criterion \
       --num-threads=$num_threads \
       --learn-rate=$learn_rate \
       --verbose=$verbose \
       --one-silence-class=$one_silence_class \
       $mpe_silphones_arg \
       $cur_mdl $alidir/final.mdl "$feats" "$lats" "$ali" $dir/$x.nnet || exit 1
  fi
  cur_mdl=$dir/$x.nnet

  #report the progress
  grep -B 2 "Overall average frame-accuracy" $dir/log/mpe.$x.log | sed -e 's|.*)||'

  x=$((x+1))
  learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
  
done

(cd $dir; [ -e final.nnet ] && unlink final.nnet; ln -s $((x-1)).nnet final.nnet)


echo "MPE/sMBR training finished"

echo "Re-estimating priors by forwarding the training set."
. cmd.sh
nj=$(cat $alidir/num_jobs)
#steps/nnet/make_priors.sh --cmd "$train_cmd" --nj $nj $data $dir || exit 1

exit 0
