#!/bin/bash
# Copyright 2013  Brno University of Technology (Author: Karel Vesely)  
# Apache 2.0.

# Sequence-discriminative MMI/BMMI training of DNN.
# 4 iterations (by default) of Stochastic Gradient Descent with per-utterance updates.
# Boosting of paths with more errors (BMMI) gets activated by '--boost <float>' option.

# For the numerator we have a fixed alignment rather than a lattice--
# this actually follows from the way lattices are defined in Kaldi, which
# is to have a single path for each word (output-symbol) sequence.


# Begin configuration section.
cmd=run.pl
num_iters=4
boost=0.0 #ie. disable boosting 
acwt=0.1
lmwt=1.0
learn_rate=0.00001
halving_factor=0.6 #ie. disable halving
drop_frames=true
verbose=1

seed=777    # seed value used for training data shuffling
skip_cuda_check=false
# End configuration section

num_threads=2
frame_limit=10000
num_stream=10
lstm_opts=" --frame-limit=10000 --dump-time=250 --num-stream=5 --network-type=fsmn "
sort_by_len=false
online=false
skip_frames_opts=" --sweep-frames=0 "
sweep_frames_filename=
criterion=mmi
si_model=
kld_scale=
max_frames=1500

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: steps/$0 <data> <lang> <srcdir> <ali> <denlats> <exp>"
  echo " e.g.: steps/$0 data/train_all data/lang exp/tri3b_dnn exp/tri3b_dnn_ali exp/tri3b_dnn_denlats exp/tri3b_dnn_mmi"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --num-iters <N>                                  # number of iterations to run"
  echo "  --acwt <float>                                   # acoustic score scaling"
  echo "  --lmwt <float>                                   # linguistic score scaling"
  echo "  --learn-rate <float>                             # learning rate for NN training"
  echo "  --drop-frames <bool>                             # drop frames num/den completely disagree"
  echo "  --boost <boost-weight>                           # (e.g. 0.1), for boosted MMI.  (default 0)"
  
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
alidir=$4
denlatdir=$5
dir=$6

for f in $data/feats.scp $denlatdir/lat.scp $srcdir/{final.nnet,final.feature_transform}; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# check if CUDA is compiled in,
if ! $skip_cuda_check; then
  cuda-compiled || { echo 'CUDA was not compiled in, skipping! Check src/kaldi.mk and src/configure' && exit 1; }
fi

mkdir -p $dir/log

#cp $alidir/{final.mdl,tree} $dir

#silphonelist=`cat $lang/silence.csl` || exit 1;
silphonelist=1


#Get the files we will need
nnet=$srcdir/$(readlink $srcdir/final.nnet || echo final.nnet);
[ -z "$nnet" ] && echo "Error nnet '$nnet' does not exist!" && exit 1;
cp $nnet $dir/0.nnet; 
nnet=$dir/0.nnet

#class_frame_counts=$srcdir/ali_train_pdf.counts
class_frame_counts=$srcdir/label.counts
[ -z "$class_frame_counts" ] && echo "Error class_frame_counts '$class_frame_counts' does not exist!" && exit 1;
cp $class_frame_counts $dir

feature_transform=$srcdir/final.feature_transform
if [ ! -f $feature_transform ]; then
  echo "Missing feature_transform '$feature_transform'"
  exit 1
fi
cp $feature_transform $dir/final.feature_transform

model=$dir/final.mdl
#[ -z "$model" ] && echo "Error transition model '$model' does not exist!" && exit 1;


if $sort_by_len; then
  echo "Sort utterences by lengths :"
  [ ! -f $data/featlen.scp ] && echo "$0: no such file $data/featlen.scp" && exit 1;
  awk 'NR==FNR{ss[$1]=$2;}NR>FNR{if($1 in ss) print $0,ss[$1];}' $data/featlen.scp $data/feats.scp | sort -k3 -n - | awk '{print $1,$2}' > $dir/train.scp
else
  cp $data/feats.scp $dir/train.scp
fi

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
[ ! -z "$cmvn_opts" -a "$online" == "false" -a ! -f $data/cmvn.scp ] && echo "$0: Missing $data/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" -a "$online" == "false" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
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

# optionally sweep frames filename,
if [ -e $data/sweep_frames.scp ]; then
    cp $data/sweep_frames.scp $dir
    #sweep_frame_fn="ark,t:$dir/sweep_frames.scp" 
    sweep_frames_filename="$dir/sweep_frames.scp" 
fi

###
### Prepare the alignments
### 
# Assuming all alignments will fit into memory
#ali="ark:gunzip -c $alidir/ali.*.gz |"
#ali="scp:exp/tri_dnn_ali_train/ali.scp"
ali="scp:$alidir/ali.scp"
[ -z $si_model ] && si_model=$nnet


###
### Prepare the lattices
###
# The lattices are indexed by SCP (they are not gziped because of the random access in SGD)
lats="scp:$denlatdir/lat.scp"

# Optionally apply boosting
if [[ "$boost" != "0.0" && "$boost" != 0 ]]; then
  #make lattice scp with same order as the shuffled feature scp
  awk '{ if(r==0) { latH[$1]=$2; }
         if(r==1) { if(latH[$1] != "") { print $1" "latH[$1] } }
  }' $denlatdir/lat.scp r=1 $dir/train.scp > $dir/lat.scp
  #get the list of alignments
  awk '{print $1;}' $alidir/ali.scp > $dir/ali.lst
  #remove feature files which have no lattice or no alignment,
  #(so that the mmi training tool does not blow-up due to lattice caching)
  mv $dir/train.scp $dir/train.scp_unfilt
  awk '{ if(r==0) { latH[$1]="1"; }
         if(r==1) { aliH[$1]="1"; }
         if(r==2) { if((latH[$1] != "") && (aliH[$1] != "")) { print $0; } }
  }' $dir/lat.scp r=1 $dir/ali.lst r=2 $dir/train.scp_unfilt > $dir/train.scp
  #create the lat pipeline
  lats="ark,o:lattice-boost-ali-ctc --b=$boost --silence-phones=$silphonelist scp:$dir/lat.scp '$ali' ark:- |"
fi
###
###
###

# Run several iterations of the MMI/BMMI training
cur_mdl=$nnet
x=1
while [ $x -le $num_iters ]; do
  echo "Pass $x (learnrate $learn_rate)"
  if [ -f $dir/$x.nnet ]; then
    echo "Skipped, file $dir/$x.nnet exists"
  else
    $cmd $dir/log/mmi.$x.log \
     nnet-train-sequential-parallel $lstm_opts \
       --feature-transform=$feature_transform \
       --class-frame-counts=$class_frame_counts \
       --acoustic-scale=$acwt \
       --lm-scale=$lmwt \
       --learn-rate=$learn_rate \
       --max-frames=$max_frames \
       --drop-frames=$drop_frames \
       --num-threads=$num_threads \
       --verbose=$verbose \
       ${kld_scale:+ --kld-scale="$kld_scale"} \
       ${si_model:+ --si-model="$si_model"} \
       ${sweep_frames_filename:+ --sweep-frames-filename=$sweep_frames_filename} \
       $cur_mdl "$feats" "$lats" "$ali" $dir/$x.nnet || exit 1
  fi
  cur_mdl=$dir/$x.nnet

  #report the progress
  grep -B 2 MMI-objective $dir/log/mmi.$x.log | sed -e 's|^[^)]*)[^)]*)||'

  x=$((x+1))
  learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
  
done

(cd $dir; [ -e final.nnet ] && unlink final.nnet; ln -s $((x-1)).nnet final.nnet)

echo "MMI/BMMI training finished"

echo "Re-estimating priors by forwarding the training set."
. cmd.sh
nj=$(cat $alidir/num_jobs)
#steps/nnet/make_priors.sh --cmd "$train_cmd" --nj $nj $data $dir || exit 1

exit 0
