#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# Create denominator lattices for MMI/MPE training.
# Creates its output in $dir/lat.*.gz

# Begin configuration section.
nj=4
cmd=run.pl
sub_split=1
beam=13.0
lattice_beam=7.0
acwt=0.1
use_gpu_id=-2
use_gpu=yes # yes|no|optional
nnet_forward_opts="--no-softmax=true --prior-scale=1.0 --num-threads=4 "
num_threads=3
# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"
max_active=5000
nnet=
max_mem=20000000 # This will stop the processes getting too large.
# This is in bytes, but not "real" bytes-- you have to multiply
# by something like 5 or 10 to get real bytes (not sure why so large)
# End configuration section.

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
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh --per-utt $data $nj || exit 1;
echo $nj > $dir/num_jobs

# fx310 #
#oov=`cat $lang/oov.int` || exit 1;

mkdir -p $dir

ln -s `pwd`/$lang $dir/
ln -s `pwd`/$lang $dir/dengraph

# Compute grammar FST which corresponds to unigram decoding graph.
new_lang="$dir/"$(basename "$lang")
echo "Making unigram grammar FST in $new_lang"
#cat $data/text | utils/sym2int.pl -f 2- $lang/words.txt | \
#cat $data/text | utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt | \
#  awk '{for(n=2;n<=NF;n++){ printf("%s ", $n); } printf("\n"); }' | \
#  utils/make_unigram_grammar.pl | fstcompile > $new_lang/G.fst \
#   || exit 1;

# mkgraph.sh expects a whole directory "lang", so put everything in one directory...
# it gets L_disambig.fst and G.fst (among other things) from $dir/lang, and
# final.mdl from $srcdir; the output HCLG.fst goes in $dir/graph.

echo "Compiling decoding graph in $dir/dengraph"
if [ -s $dir/dengraph/HCLG.fst ]; then
   echo "Graph $dir/dengraph/HCLG.fst already exists: skipping graph creation."
else
#  utils/mkgraph.sh $new_lang $srcdir $dir/dengraph || exit 1;
  local/mkgraph_uni_htk.sh
fi

echo $thread_string

#Get the files we will need
cp $srcdir/{tree,final.mdl} $dir

[ -z "$nnet" ] && nnet=$srcdir/final.nnet;
[ ! -f "$nnet" ] && echo "Error nnet '$nnet' does not exist!" && exit 1;

class_frame_counts=$srcdir/ali_train_pdf.counts
[ -z "$class_frame_counts" ] && echo "Error class_frame_counts '$class_frame_counts' does not exist!" && exit 1;

feature_transform=$srcdir/final.feature_transform
if [ ! -f $feature_transform ]; then
  echo "Missing feature_transform '$feature_transform'"
  exit 1
fi

model=$dir/final.mdl
[ -z "$model" ] && echo "Error transition model '$model' does not exist!" && exit 1;

# remove the softmax from the nnet
#nnet_i=$nnet; nnet=$dir/$(basename $nnet)_nosoftmax;
#nnet-trim-n-last-transforms --n=1 --binary=false $nnet_i $nnet 2>$dir/$(basename $nnet)_log || exit 1;


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
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $sdata/1/cmvn.scp ] && echo "$0: Missing $sdata/1/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"

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
if [ $sub_split -eq 1 ]; then 
  $cmd JOB=1:$nj $dir/log/decode_den.JOB.log \
    latgen-faster-mapped$thread_string --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
      --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
      $dir/dengraph/HCLG.fst "$feats" "ark,scp:$dir/lat.JOB.ark,$dir/lat.JOB.scp" || exit 1;
else
  for n in `seq $nj`; do
    if [ -f $dir/.done.$n ] && [ $dir/.done.$n -nt $alidir/final.mdl ]; then
      echo "Not processing subset $n as already done (delete $dir/.done.$n if not)";
    else
      sdata2=$data/split$nj/$n/split$sub_split;
      if [ ! -d $sdata2 ] || [ $sdata2 -ot $sdata/$n/feats.scp ]; then
        split_data.sh --per-utt $sdata/$n $sub_split || exit 1;
      fi
      mkdir -p $dir/log/$n
      mkdir -p $dir/part
      feats_subset=$(echo $feats | sed s:JOB/:$n/split$sub_split/JOB/:g)
      $cmd JOB=1:$sub_split $dir/log/$n/decode_den.JOB.log \
        latgen-faster-mapped$thread_string --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
          --max-mem=$max_mem --max-active=$max_active --word-symbol-table=$lang/words.txt $srcdir/final.mdl  \
          $dir/dengraph/HCLG.fst "$feats_subset" "ark,scp:$dir/lat.$n.JOB.ark,$dir/lat.$n.JOB.scp" || exit 1;
      echo Merging lists for data subset $n
      for k in `seq $sub_split`; do
        cat $dir/lat.$n.$k.scp
      done > $dir/lat.$n.all.scp
      echo Merge the ark $n
      lattice-copy scp:$dir/lat.$n.all.scp ark,scp:$dir/lat.$n.ark,$dir/lat.$n.scp || exit 1;
      #remove the data
      rm $dir/lat.$n.*.ark $dir/lat.$n.*.scp $dir/lat.$n.all.scp
      touch $dir/.done.$n
    fi
  done
fi

      

#3) Merge the SCPs to create full list of lattices (will use random access)
echo Merging to single list $dir/lat.scp
for ((n=1; n<=nj; n++)); do
  cat $dir/lat.$n.scp
done > $dir/lat.scp


echo "$0: doe generating denominator lattices."
