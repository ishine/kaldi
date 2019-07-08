#!/bin/bash
#Copyright 2019.6.10 wangzhichao Apache 2.0

# This script generate a subset of all alignments based on the data dir.
cmd=run.pl
num_jobs=40

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <data> <dest-ali-dir> <src-ali-dir>"
  exit 1;
fi

data=$1
dest=$2
src=$3

if [ -d $dest ]; then
  echo "Error: $dest is already exists!" && exit 1;
fi

mkdir -p $dest
export LC_ALL=C

cat $src/ali.*.syllable >$dest/ali.total
cp $src/cmvn_opts $src/final.mdl $dest
echo $num_jobs >$dest/num_jobs || exit 1;

echo "$0: splitting data to get reference utt2spk for individual ali files."
utils/split_data.sh $data $num_jobs || exit 1;

echo "$0: generate alignments according to the reference utt2spk files"
utils/filter_scps.pl JOB=1:$num_jobs \
  $data/split$num_jobs/JOB/utt2spk $dest/ali.total $dest/ali.JOB.syllable

echo "$0: checking the alignment files generated have at least 90% of the utterances."
for i in `seq 1 $num_jobs`; do
  num_lines=`cat $dest/ali.$i.syllable |wc -l` || exit 1;
  num_lines_tot=`cat $data/split$num_jobs/$i/utt2spk |wc -l`  || exit 1;
  python -c "import sys;
percent = 100.0 * float($num_lines) / $num_lines_tot
if percent < 90 :
  print ('$dest/ali.$i.gz {0}% utterances missing.'.format(percent))"  || exit 1;
done

rm $dest/ali.total
echo "Generate alignments success!"
exit 0
