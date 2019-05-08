#!/bin/bash

# Copyright 2012/2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# Begin configuration.
config=            # config, which is also sent to all other scripts

# NETWORK INITIALIZATION
nnet_init=          # select initialized MLP (override initialization)
nnet_proto=         # select network prototype (initialize it)
proto_opts=        # non-default options for 'make_nnet_proto.py'
feature_transform= # provide feature transform (=splice,rescaling,...) (don't build new one)
network_type=dnn   # (dnn,cnn1d,cnn2d,lstm) select type of neural network
#
hid_layers=4       # nr. of hidden layers (prior to sotfmax or bottleneck)
hid_dim=1024       # select hidden dimension
momentum=0.9
#
init_opts=         # options, passed to the initialization script

# TRAINING SCHEDULER
learn_rate=0.008   # initial learning rate
train_opts=        # options, passed to the training script
train_tool=        # optionally change the training tool
frame_weights=     # per-frame weights for gradient weighting
vocab_file=
class_boundary=
var_penalty=
zt_mean_filename=
num_threads=1

# OTHER
seed=777    # seed value used for training data shuffling and initialization
skip_cuda_check=false
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


. parse_options.sh || exit 1;


if [ $# != 4 ]; then
   echo "Usage: $0 <data-train> <data-dev> <lang-dir> <ali-train> <ali-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang exp/mono_ali_train exp/mono_ali_cv exp/mono_nnet"
   echo ""
   echo " Training data : <data-train> (for optimizing cross-entropy)"
   echo " Held-out data : <data-dev> (for learn-rate/model selection based on cross-entopy)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo ""
   echo "  --feature-transform <file> # re-use this input feature transform"
   echo "  --vocab-file <file> # re-use this input feature transform"
   echo ""
   echo "  --learn-rate <float>     # initial leaning-rate"
   echo ""
   exit 1;
fi

data=$1
data_cv=$2
lang=$3
dir=$4

# Using alidir for supervision (default)
if [ -z "$labels" ]; then 
  #silphonelist=`cat $lang/phones/silence.csl` || exit 1;
  for f in $lang/vocab.txt ; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
fi

for f in $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

echo
echo "# INFO"
echo "$0 : Training Neural Network"
printf "\t dir       : $dir \n"
printf "\t Train-set : $data \n"
printf "\t CV-set    : $data_cv \n"

mkdir -p $dir/{log,nnet}

# skip when already trained
[ -e $dir/final.nnet ] && printf "\nSKIPPING TRAINING... ($0)\nnnet already trained : $dir/final.nnet ($(readlink $dir/final.nnet))\n\n" && exit 0

# check if CUDA is compiled in,
if ! $skip_cuda_check; then
  cuda-compiled || { echo 'CUDA was not compiled in, skipping! Check src/kaldi.mk and src/configure' && exit 1; }
fi

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
# shuffle the list
#cat $data/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/train.scp
rootdir=`pwd`
cd $dir
  unlink train.scp
  ln -s $rootdir/$data/feats.scp  train.scp
cd -
  cp $data_cv/feats.scp $dir/cv.scp

# print the list sizes
#wc -l $dir/train.scp $dir/cv.scp

###### INITIALIZE THE NNET ######
echo 
echo "# NN-INITIALIZATION"
[ ! -z "$nnet_init" ] && echo "Using pre-initialized network '$nnet_init'";
if [ ! -z "$nnet_proto" ]; then 
  echo "Initializing using network prototype '$nnet_proto'";
  nnet_init=$dir/nnet.init; log=$dir/log/nnet_initialize.log
  nnet-initialize $nnet_proto $nnet_init 2>$log || { cat $log; exit 1; } 
fi

###### PREPARE FEATURE PIPELINE ######
if [ -z "$vocab_file" ]; then
    echo "Vocabulary file vocab_file=$lang/vocab.txt (by force)";
    vocab_file=$lang/vocab.txt
fi
     
#feats_tr="ark:utils/sym2int.pl -f 2- $vocab_file $dir/train.scp |"
#feats_cv="ark:utils/sym2int.pl -f 2- $vocab_file $dir/cv.scp |"
feats_tr="ark:copy-align scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-align scp:$dir/cv.scp ark:- |"

###### PREPARE CLASS ######
false && \
{
if [ -z "$class_boundary" ]; then
    echo "Class boundary file $lang/class_boundary.txt (by force)";
    class_boundary=$lang/class_boundary.txt
fi
}

###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING SCHEDULER"
steps/nnet/train_scheduler_thread.sh \
  --learn-rate $learn_rate \
  --momentum $momentum \
  --randomizer-seed $seed \
  --num-threads $num_threads \
  ${train_opts} \
  ${train_tool:+ --train-tool "$train_tool"} \
  ${frame_weights:+ --frame-weights "$frame_weights"} \
  ${class_boundary:+ --class-boundary "$class_boundary"} \
  ${var_penalty:+ --var-penalty "$var_penalty"} \
  ${zt_mean_filename:+ "--zt-mean-filename=$zt_mean_filename"} \
  ${config:+ --config $config} \
  $nnet_init "$feats_tr" "$feats_cv" $dir || exit 1

echo "$0 successfuly finished.. $dir"

sleep 3
exit 0
