#!/bin/bash
# Copyright 2012-2013 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# Aligns 'data' to sequences of transition-ids using Neural Network based acoustic model.
# Optionally produces alignment in lattice format, this is handy to get word alignment.

# Begin configuration section.  
nj=4
cmd=run.pl
stage=0
# Begin configuration.
scale_opts="--acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
nnet_forward_opts=" --no-softmax=true --prior-scale=1.0 --num-threads=4 --copy-posterior=false "
skip_opts=
sweep_frames_opts="--sweep-frames=0"
sweep_frame_fn=
online=false

align_to_lats=false # optionally produce alignment in lattice format
 lats_decode_opts="--acoustic-scale=0.1 --beam=20 --lattice_beam=10"
 lats_graph_scales="--transition-scale=1.0 --self-loop-scale=0.1"

use_gpu="yes" # yes|no|optionaly
# End configuration options.

[ $# -gt 0 ] && echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: $0 <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  $0 data/train data/lang exp/tri1 exp/tri1_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data $nj || exit 1;

#cp $srcdir/{tree,final.mdl} $dir || exit 1;

# Select default locations to model files
nnet=$srcdir/final.nnet;
class_frame_counts=$srcdir/label.counts
feature_transform=$srcdir/final.feature_transform
model=$dir/final.mdl

# Check that files exist
for f in $sdata/1/feats.scp $sdata/1/text $lang/TL.fst $nnet $feature_transform $class_frame_counts; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done


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
[ ! -z "$cmvn_opts" -a "$online" == "false" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" -a "$online" == "false" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
[ ! -z "$cmvn_opts" -a "$online" == "true" ] && feats="$feats apply-cmvn-sliding $cmvn_opts ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# skip-frames (optional),
[ ! -z "$skip_opts" ] && nnet_forward_opts="$nnet_forward_opts $skip_opts $sweep_frames_opts"

# optionally sweep frames filename,
if [ -e $data/sweep_frames.scp ]; then
    cp $data/sweep_frames.scp $dir
    sweep_frame_fn="ark,t:$dir/sweep_frames.scp" 
fi

# nnet-forward,
feats="$feats nnet-forward-parallel $nnet_forward_opts --feature-transform=$feature_transform --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu $nnet ark:- $sweep_frame_fn ark:- |"
#

echo "$0: aligning data '$data' using nnet/model '$srcdir', putting alignments in '$dir'"

#wd007
# Map oovs in reference transcription, 
#oov=`cat $lang/oov.int` || exit 1;
#tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";
tra="ark:utils/sym2int.pl -f 2- $lang/words.txt $sdata/JOB/text|";
# We could just use align-mapped in the next line, but it's less efficient as it compiles the
# training graphs one by one.
if [ $stage -le 0 ]; then
  train_graphs="ark:compile-train-graphs-ctc $lang/TL.fst '$tra' ark:- |"
  $cmd JOB=1:$nj $dir/log/align.JOB.log \
    compile-train-graphs-ctc $lang/TL.fst "$tra" ark:- \| \
    align-compiled-mapped-ctc $scale_opts --beam=$beam --retry-beam=$retry_beam ark:- \
    "$feats" "ark,scp:$dir/ali.JOB.ark,$dir/ali.JOB.scp" || exit 1;
    #"$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" || exit 1;
   
   #Merge the SCPs to create full list of alignment (will use random access)
	echo Merging to single list $dir/ali.scp
	for ((n=1; n<=nj; n++)); do
  		cat $dir/ali.$n.scp
	done > $dir/ali.scp

fi

# Optionally align to lattice format (handy to get word alignment)
if [ "$align_to_lats" == "true" ]; then
  echo "$0: aligning also to lattices '$dir/lat.*.gz'"
  $cmd JOB=1:$nj $dir/log/align_lat.JOB.log \
    compile-train-graphs-ctc $lat_graph_scale $dir/tree $dir/final.mdl  $lang/L.fst "$tra" ark:- \| \
    latgen-faster-mapped $lat_decode_opts --word-symbol-table=$lang/words.txt $dir/final.mdl ark:- \
      "$feats" "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

echo "$0: done aligning data."
