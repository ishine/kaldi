#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# Create denominator lattices for MMI/MPE training.
# Creates its output in $dir/lat.*.gz

# Begin configuration section.
nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
model=              # non-default location of transition model (optional)
class_frame_counts= # non-default location of PDF counts (optional)
srcdir=             # non-default location of DNN-dir (decouples model dir from decode dir)

nj=4
cmd=run.pl

acwt=0.1 # note: only really affects pruning (scoring is on lattices).
beam=13.0
lattice_beam=7.0
max_active=5000
max_mem=20000000 # This will stop the processes getting too large.
# This is in bytes, but not "real" bytes-- you have to multiply
# by something like 5 or 10 to get real bytes (not sure why so large)
# End configuration section.

use_gpu=yes # yes|no|optional
nnet_forward_opts="--no-softmax=true --prior-scale=1.0 --num-threads=4 --num-stream=40 --batch-size=20"
num_threads=3
# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

skip_opts=
sweep_frames_opts="--sweep-frames=1"
online=false

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/make_denlats_nnet.sh [options] <data-dir> <lang-dir> <src-dir> <exp-dir>"
   echo "  e.g.: steps/make_denlats.sh data/train data/lang exp/tri1 exp/tri1_denlats"
   echo "Works for (delta|lda) features, and (with --transform-dir option) such features"
   echo " plus transforms."
   echo ""
   echo "Main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --sub-split <n-split>                            # e.g. 40; use this for "
   echo "                           # large databases so your jobs will be smaller and"
   echo "                           # will (individually) finish reasonably soon."
   echo "  --transform-dir <transform-dir>   # directory to find fMLLR transforms."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

sdata=$data/split$nj
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
mkdir -p $dir/log
mkdir -p $dir
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data $nj || exit 1;
echo $nj > $dir/num_jobs


#Get the files we will need
cp $srcdir/{tree,final.mdl} $dir

# Select default locations to model files (if not already set externally)
[ -z "$nnet" ] && nnet=$srcdir/final.nnet
[ -z "$model" ] && model=$srcdir/final.mdl
[ -z "$feature_transform" ] && feature_transform=$srcdir/final.feature_transform

#
[ -z "$class_frame_counts" -a -f $srcdir/prior_counts ] && class_frame_counts=$srcdir/prior_counts # priority,
[ -z "$class_frame_counts" ] && class_frame_counts=$srcdir/ali_train_pdf.counts

# Check that files exist
for f in $sdata/1/feats.scp $nnet $model $feature_transform $class_frame_counts $lang/HCLG.fst; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done


###
### Prepare feature pipeline (same as for decoding)
###
# PREPARE FEATURE EXTRACTION PIPELINE
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
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" -a "$online" == "false" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
[ ! -z "$cmvn_opts" -a "$online" == "true" ] && feats="$feats apply-cmvn-sliding $cmvn_opts ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# skip-frames (optional),
[ ! -z "$skip_opts" ] && nnet_forward_opts="$nnet_forward_opts $skip_opts $sweep_frames_opts"

# Finally add feature_transform and the MLP
#feats="$feats nnet-forward --feature-transform=$feature_transform --no-softmax=true --use-gpu=$use_gpu --class-frame-counts=$class_frame_counts $nnet ark:- ark:- |"
feats="$feats nnet-forward-parallel $nnet_forward_opts --feature-transform=$feature_transform --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu $nnet ark:- ark:- |"
###
###
###

###
### We will produce lattices, where the correct path is not necessarily present
###

#1) We don't use reference path here...

echo "Generating the denlats"
#2) Generate the denominator lattices
$cmd JOB=1:$nj $dir/log/decode_den.JOB.log \
    latgen-faster-mapped$thread_string --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
      --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
      $lang/HCLG.fst "$feats" "ark,scp:$dir/lat.JOB.ark,$dir/lat.JOB.scp" || exit 1;
      

#3) Merge the SCPs to create full list of lattices (will use random access)
echo Merging to single list $dir/lat.scp
for ((n=1; n<=nj; n++)); do
  cat $dir/lat.$n.scp
done > $dir/lat.scp


echo "$0: doe generating denominator lattices."

