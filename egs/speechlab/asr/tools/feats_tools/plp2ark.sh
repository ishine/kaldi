#!/bin/bash

# Begin configuration section.
nj=5
#cmd=run.pl
cmd="queue.pl -l hostname=wuhan"
compress=false
htk=true
feat=plp
tool=/aifs/users/wd007/src/KALDI/kaldi-merge/src/featbin/copy-feats
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Usage: $0 <feats-scplist> <ark-dir> "	
   exit 1;
fi

scplist=$1
dir=$2
logdir=$dir/log
m=0
n=0

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log

[ -e $dir/.n ] &&  n=$(cat $dir/.n)

for featscp in `cat $scplist`; do 
  #skip finished scp
  m=$[$m+1]
  [ $m -le $n ] && echo -n "skipping... "$featscp && continue

  arkdir=$dir/$m
  [ ! -d $arkdir ] && mkdir $arkdir

  split_scps=""
  for k in $(seq $nj); do
    split_scps="$split_scps $arkdir/${feat}.$k.scp"
  done
  utils/split_scp.pl $featscp $split_scps || exit 1;

  $cmd JOB=1:$nj $logdir/${feat}2ark_${m}.JOB.log \
    copy-feats --compress=$compress --htk-in=$htk scp:$arkdir/${feat}.JOB.scp \
		ark,scp:`pwd`/$arkdir/raw_${feat}_${m}.JOB.ark,$arkdir/raw_${feat}_${m}.JOB.scp || exit 1;

  for ((k=1; k<=nj; k++)); do
    cat $arkdir/raw_${feat}_${m}.${k}.scp
  done > $arkdir/raw_${feat}_${m}.scp
  echo $m >$dir/.n
done

echo Merging to single list $dir/ark.scp
    for ((k=1; k<=m; k++)); do
        cat $dir/$k/raw_${feat}_${k}.scp
    done > $dir/raw_${feat}.scp

echo "$0: done copy data."
