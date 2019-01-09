#!/bin/bash

# This is sometimes needed by higher-level scripts


if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  echo "Usage: $0 <data-dir>"
  echo "Prints the number of frames of data in the data-dir"
  exit 1;
fi
nj=30
data=$1
sdata=$data/split$nj
utils/split_data.sh $data $nj

if [ ! -f $data/utt2dur ]; then
  for n in `seq $nj`; do
    utils/data/get_utt2dur.sh $sdata/$n &
  done
  wait;
  for n in `seq $nj`; do
    cat $sdata/$n/utt2dur
  done > $data/utt2dur
fi


