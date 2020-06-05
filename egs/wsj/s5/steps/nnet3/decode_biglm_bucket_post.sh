#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#           2019  Hang Lyu
# Apache 2.0

# Begin configuration.
nj=4
cmd=run.pl
minactive=200
maxactive=7000
beam=15.0
lattice_beam=8.0
bucket_length=5

acwt=1.0
post_decode_acwt=10.0
skip_scoring=false
stage=0

online_ivector_dir=
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
frames_per_chunk=50
use_gpu=false
use_batch=false
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/nnet3/decode_biglm_bucket_post.sh [options] <graph-dir> \\"
   echo "<old-LM-fst> <new-LM-fst> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/nnet3/decode_biglm_bucket_post.sh exp/chain/tdnn/graph_tgsmall \\"
   echo "data/lang_tgsmall/G.fst data/lang_tglarge/G.fst data/test_dev93 \\"
   echo "exp/mono/decode_dev93_biglm_tgsmall_tglarge."
   echo ""
   echo "This script will generate the posteriors in first or use them directly"
   echo "if exists. Then it will decode from the posteriors"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi


graphdir=$1
oldlm_fst=$2
newlm_fst=$3
data=$4
dir=$5

srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`

utils/lang/check_phones_compatible.sh {$srcdir,$graphdir}/phones.txt || exit 1

for f in $srcdir/final.mdl $graphdir/HCLG.fst $oldlm_fst $newlm_fst; do
  [ ! -f $f ] && echo "decode_biglm_bucket_post.sh: no such file $f" && exit 1;
done

batch_string=
gpu=
if $use_gpu; then
  queue_opt="--gpu 1"
  gpu="yes"
else
  gpu="no"
fi
if $use_batch; then
  batch_string="-batch"
fi

#### Check data
mkdir -p $dir/log
data_ok=false
if [ -f $data/feats.scp ]; then
  [[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
  echo $nj > $dir/num_jobs
  data_ok=true
  stage=0
fi
complete_posterior=true
for i in `seq $nj`; do
  if [ ! -f $sdata/$i/posterior.scp ]; then
    complete_posterior=false
  fi
done
if $complete_posterior; then
  echo "Use the posteriors directly"
  stage=1
  data_ok=true
fi
if ! $data_ok; then
  echo "Neither posterior.scp nor feats.scp exists."
  exit 1;
fi

posteriors="ark,scp:$sdata/JOB/posterior.ark,$sdata/JOB/posterior.scp"
posteriors_rspecifier="scp:$sdata/JOB/posterior.scp"
if [ $stage -le 0 ]; then
  ## Set up features. Generate the posteriors.
  if [ -f $srcdir/online_cmvn ]; then online_cmvn=true
  else online_cmvn=false; fi

  if ! $online_cmvn; then
    echo "$0: feature type is raw"
    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
  else
    feats="ark,s,cs:apply-cmvn-online $cmvn_opts --spk2utt=ark:$sdata/JOB/spk2utt $srcdir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- |"
  fi

  if [ ! -z "$online_ivector_dir" ]; then
    ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
    ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
  fi

  frame_subsampling_opt=
  if [ -f $srcdir/frame_subsampling_factor ]; then
    # e.g. for 'chain' systems
    frame_subsampling_opt="--frame-subsampling-factor=$(cat $srcdir/frame_subsampling_factor)"
  fi

  $cmd $queue_opt JOB=1:$nj $dir/log/nnet_compute.JOB.log \
    nnet3-compute$batch_string $ivector_opts $frame_subsampling_opt \
    --acoustic-scale=$acwt \
    --extra-left-context=$extra_left_context \
    --extra-right-context=$extra_right_context \
    --extra-left-context-initial=$extra_left_context_initial \
    --extra-right-context-final=$extra_right_context_final \
    --frames-per-chunk=$frames_per_chunk \
    --use-gpu=$gpu --use-priors=true \
    $srcdir/final.mdl "$feats" "$posteriors"
fi

if [ $stage -le 1 ]; then
  if [ "$post_decode_acwt" == 1.0 ]; then
    lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
  else
    lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
  fi

  [ -f `dirname $oldlm_fst`/words.txt ] && ! cmp `dirname $oldlm_fst`/words.txt $graphdir/words.txt && \
    echo "Warning: old LM words.txt does not match with that in $graphdir .. probably will not work.";
  [ -f `dirname $newlm_fst`/words.txt ] && ! cmp `dirname $oldlm_fst`/words.txt $graphdir/words.txt && \
    echo "Warning: new LM words.txt does not match with that in $graphdir .. probably will not work.";

  # fstproject replaces the disambiguation symbol #0, which only appears on the
  # input side, with the <eps> that appears in the corresponding arcs on the output side.
  if [ -f `dirname $oldlm_fst`/G_sort.fst ]; then
    oldlm_cmd="`dirname $oldlm_fst`/G_sort.fst"
  else
    oldlm_cmd="fstproject --project_output=true $oldlm_fst | fstarcsort --sort_type=ilabel |"
  fi
  if [ -f `dirname $newlm_fst`/G_sort.fst ]; then
    newlm_cmd="`dirname $newlm_fst`/G_sort.fst"
  else
    newlm_cmd="fstproject --project_output=true $newlm_fst | fstarcsort --sort_type=ilabel |"
  fi

  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    latgen-biglm-bucket-faster-mapped --acoustic-scale=$acwt \
      --allow-partial=true --max-active=$maxactive --min-active=$minactive \
      --beam=$beam --lattice-beam=$lattice_beam --bucket-length=$bucket_length \
      --word-symbol-table=$graphdir/words.txt \
      $srcdir/final.mdl $graphdir/HCLG.fst "$oldlm_cmd" "$newlm_cmd" \
      "$posteriors_rspecifier" "$lat_wspecifier" || exit 1;
#  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
#    latgen-biglm-faster-mapped --acoustic-scale=$acwt \
#      --allow-partial=true --max-active=$maxactive --min-active=$minactive \
#      --beam=$beam --lattice-beam=$lattice_beam \
#      --word-symbol-table=$graphdir/words.txt \
#      $srcdir/final.mdl $graphdir/HCLG.fst "$oldlm_cmd" "$newlm_cmd" \
#      "$posteriors_rspecifier" "$lat_wspecifier" || exit 1;
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $data $graphdir $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;
