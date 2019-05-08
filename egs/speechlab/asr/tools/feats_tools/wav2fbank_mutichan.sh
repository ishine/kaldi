#!/bin/bash

if [ $# != 3 ]; then
   echo "usage: wav2fbank.sh <fbank_config> <wavlist> <path-to-fbankdir>";
   exit 1;
fi

fbank_config=$1
wavlist=$2
dir=$3

compress=false
num_chan=3
#cmd=run.pl
cmd="queue.pl -l hostname='(shenzhen|changsha|beijing)' "
#cmd="queue.pl"
nj=600
mj=20

[ -f path.sh ] && . path.sh 

#rm -rf $dir
[ ! -d $dir ]  && mkdir -p $dir

name=`basename $wavlist`
wavscp=$name".scp"
awk -F "/|\\\.wav" '{print $(NF-1), $0;}' $wavlist > $dir/$wavscp
#cp $wavlist $dir/$wavscp

split_scps=""
  for k in $(seq $nj); do
    split_scps="$split_scps $dir/wav.$k.scp"
  done

feats=""
  for((i=0;i<$num_chan;i++)); do
    feats="$feats \"ark:compute-fbank-feats --verbose=1 --channel=$i --config=$fbank_config scp,p:$dir/wav.JOB.scp ark:- |\" "
  done
echo $feats
#false && \
{
utils/split_scp.pl $dir/$wavscp $split_scps || exit 1;

 $cmd --max-jobs-run $mj JOB=1:$nj $dir/wav2fbank.JOB.log \
    paste-feats $feats ark,scp:`pwd`/$dir/raw_feat.JOB.ark,`pwd`/$dir/raw_feat.JOB.scp  || exit 1;

  for ((k=1; k<=nj; k++)); do
    cat $dir/raw_feat.${k}.scp
  done > $dir/raw_feat.scp
}

rootdir=`pwd`
cd $dir
ln -s raw_feat.scp feats.scp
cd $rootdir

feat-to-len scp:$dir/feats.scp ark,t:$dir/featlen.scp || exit 1;

#compute-cmvn-stats --spk2utt=ark:$dir/spk2utt scp:$dir/feats.scp \
compute-cmvn-stats  scp:$dir/feats.scp \
    ark,scp:`pwd`/$dir/cmvn.ark,`pwd`/$dir/cmvn.scp || exit 1;

# spk
awk '{print $1, $1;}' $dir/feats.scp > $dir/utt2spk
cp $dir/utt2spk $dir/spk2utt

exit 0

