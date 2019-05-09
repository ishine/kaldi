#!/bin/bash

# Copyright 2012-2013 Karel Vesely, Daniel Povey
# Apache 2.0

# Begin configuration section. 
nnet=               # non-default location of DNN (optional)
feature_transform=  # non-default location of feature_transform (optional)
model=              # non-default location of transition model (optional)
class_frame_counts= # non-default location of PDF counts (optional)
srcdir=             # non-default location of DNN-dir (decouples model dir from decode dir)

stage=0 # stage=1 skips lattice generation
nj=4
cmd=run.pl

acwt=0.1 # note: only really affects pruning (scoring is on lattices). acoustic-scale
beam=13.0
min_active=20
max_active=7000 # limit of active tokens
max_mem=50000000 # approx. limit to memory consumption during minimization in bytes
nnet_forward_opts="--no-softmax=true --prior-scale=1.0 --num-stream=4 --batch-size=20 --num-threads=1 "

skip_scoring=false
skip_opts=
online=false

num_threads=10 # if >1, will use latgen-faster-parallel
#parallel_opts="-pe smp 5" #"-pe smp $((num_threads+1))" # use 2 CPUs (1 DNN-forward, 1 decoder)
use_gpu="yes" # yes|no|optionaly
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the DNN and transition model is."
   echo "e.g.: $0 exp/dnn1/graph_tgpr data/test exp/dnn1/decode_tgpr"
   echo ""
   echo "This script works on plain or modified features (CMN,delta+delta-delta),"
   echo "which are then sent through feature-transform. It works out what type"
   echo "of features you used from content of srcdir."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo ""
   echo "  --nnet <nnet>                                    # non-default location of DNN (opt.)"
   echo "  --srcdir <dir>                                   # non-default dir with DNN/models, can be different"
   echo "                                                   # from parent dir of <decode-dir>' (opt.)"
   echo ""
   echo "  --acwt <float>                                   # select acoustic scale for decoding"
   echo "  --scoring-opts <opts>                            # options forwarded to local/score.sh"
   echo "  --num-threads <N>                                # N>1: run multi-threaded decoder"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
[ -z $srcdir ] && srcdir=`dirname $dir`; # Default model directory one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Select default locations to model files (if not already set externally)
[ -z "$nnet" ] && nnet=$srcdir/final.nnet
[ -z "$model" ] && model=$srcdir/final.mdl
[ -z "$feature_transform" ] && feature_transform=$srcdir/final.feature_transform
#
[ -z "$class_frame_counts" -a -f $srcdir/prior_counts ] && class_frame_counts=$srcdir/prior_counts # priority,
[ -z "$class_frame_counts" ] && class_frame_counts=$srcdir/ali_train_pdf.counts

# Check that files exist
for f in $sdata/1/feats.scp $nnet $model $feature_transform $class_frame_counts $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "$0: missing file $f" && exit 1;
done

# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads" 


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
[ ! -z "$skip_opts" ] && nnet_forward_opts="$nnet_forward_opts $skip_opts"
#

# Run the decoding in the queue,
if [ $stage -le 0 ]; then
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode.JOB.log \
    nnet-forward-parallel $nnet_forward_opts --feature-transform=$feature_transform --class-frame-counts=$class_frame_counts --use-gpu=$use_gpu $nnet "$feats" ark:- \| \
    decode-faster-mapped --min-active=$min_active --max-active=$max_active --beam=$beam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $model $graphdir/HCLG.fst ark:- ark,t:$dir/trans.JOB || exit 1;
fi

ref_filtering_cmd=" sed 's/E/一/g' | sed 's/G/J/g' | sed 's/R/二/g' "
# Run the scoring
if ! $skip_scoring ; then
    mkdir -p $dir/scoring_kaldi
    mkdir -p $dir/scoring_kaldi/wer_details

    hyp=$dir/scoring_kaldi/hyps.txt
    ref=$dir/scoring_kaldi/test_filt.txt
    cat $data/text | sed 's/E/一/g' | sed 's/G/J/g' | sed 's/R/二/g' > $ref || exit 1;
    cat $dir/trans.* > $dir/scoring_kaldi/trans.wordid.txt || exit 1;
    cat $dir/scoring_kaldi/trans.wordid.txt | utils/int2sym.pl -f 2- $graphdir/words.txt > $hyp

    export LC_ALL=en_US.UTF-8
        mv $hyp $hyp".org"
        awk '{printf("%s", $1); for(j=2;j<=NF;j++){num=split($j,ss,"");for(i=1;i<=num;i++){c=ss[i];if(c~/[\000-\177]/){s=s""c; if(i==num){printf(" %s",s);s="";}} else if(c!~/[\000-\177]/){if(s!=""){printf(" %s",s);s="";}printf(" %s",c);}} } printf("\n");}' $hyp".org" | sed 's/E/一/g' | sed 's/G/J/g' | sed 's/R/二/g' > $hyp
    export LC_ALL=C

    cat $hyp | compute-wer --text --mode=present ark:$ref  ark,p:-  >& $dir/scoring_kaldi/best_wer || exit 1;

    cat $hyp | align-text --special-symbol="'***'" ark:$ref ark:- ark,t:- |\
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" | tee $dir/scoring_kaldi/wer_details/per_utt |\
      utils/scoring/wer_per_spk_details.pl $data/utt2spk > $dir/scoring_kaldi/wer_details/per_spk || exit 1;

fi

exit 0;
