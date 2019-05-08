#!/bin/bash

if [ $# != 3 ]; then
   echo "usage: make_fbank.sh <fbank_config> <data-dir> <path-to-fbankdir>";
   exit 1;
fi

fbank_config=$1
wav_dir=$2
dir=$3

compress=false
name=`basename $wav_dir`
wavlist=$name".wavlist"
cmd="run.pl" #run.pl
num_chan=3

[ -f path.sh ] && . path.sh 

#rm -rf $dir
[ ! -d $dir ]  && mkdir -p $dir

false && \
{
for fn in $wavlist $name".char.mlf" text utt2spk spk2utt; do
    [ ! -f $wav_dir/$fn ] && echo "$fn not found!" && exit 1;
    cp $wav_dir/$fn $dir/$fn
done
}

# text
#grep -v '#!MLF!#' $wav_dir/$name".char.mlf" |  awk -F "/|\.lab\"" '{if(NF==3)printf("%s",$2);else if($0==".")printf("\n");else printf(" %s",$0);}' > $dir/text || exit 1;


wavscp=$name".wav.scp"
awk -F "/|\\\.wav" '{print $(NF-1), $0;}' $wav_dir/$wavlist > $dir/$wavscp

feats=""
  for((i=0;i<$num_chan;i++)); do
    feats="$feats \"ark:compute-fbank-feats --verbose=1 --channel=$i --config=$fbank_config scp,p:$dir/$wavscp ark:- |\" "
  done
echo $feats

$cmd $dir/fbank.log \
    paste-feats $feats ark,scp:`pwd`/$dir/feats.ark,`pwd`/$dir/feats.scp || exit 1;

feat-to-len scp:$dir/feats.scp ark,t:$dir/featlen.scp || exit 1;

#compute-cmvn-stats --spk2utt=ark:$dir/spk2utt scp:$dir/feats.scp \
compute-cmvn-stats  scp:$dir/feats.scp \
    ark,scp:`pwd`/$dir/cmvn.ark,`pwd`/$dir/cmvn.scp || exit 1;

# spk
awk '{print $1, $1;}' $dir/feats.scp > $dir/utt2spk
cp $dir/utt2spk $dir/spk2utt

exit 0

