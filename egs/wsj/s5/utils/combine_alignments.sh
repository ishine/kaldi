#!/bin/bash
# Copyright 2017 wang zhichao.  Apache 2.0.

# This script combines the lats from multiple source directories into
# a single destination directory.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 2 ]; then
  echo "Usage: combine_lats.sh <dest-data-dir> <src-data-dir1> <src-data-dir2> ..."
  echo "Note, files that don't appear in all source dirs will not be combined,"
  echo "with the exception of utt2uniq and segments, which are created where necessary."
  exit 1
fi

dest=$1;
shift;

first_src=$1;

rm -r $dest 2>/dev/null
mkdir -p $dest;

export LC_ALL=C

for dir in $*; do
  if [ ! -f $dir/ali.1.gz ]; then
    echo "$0: no such file $dir/ali.1.gz"
    exit 1;
  fi
done

for in_dir in $*; do
  if [ ! -f $in_dir/trans.1 ]; then
    echo "$0: no such file $in_dir/trans.1"
    exit 1;
  fi
done

num_leaves=$(tree-info $first_src/tree |grep num-pdfs|awk '{print $2}')

# we are going to check the consistency of tree info in each source
for in_dir in $*; do
  if [ ! -f $in_dir/tree ]; then
    echo "$0: no such file $in_dir/tree"
    exit 1;
  else 
    num_pdfs=$(tree-info $in_dir/tree |grep num-pdfs|awk '{print $2}')
    [ $num_leaves -ne $num_pdfs ] && echo "tree in $first_src does not equal to tree in $in_dir" && exit 1;
  fi
done 

cp $first_src/cmvn_opts $dest 2>/dev/null
cp $first_src/final.* $dest 2>/dev/null
cp $first_src/num_jobs $dest 2>/dev/null
cp $first_src/phones.txt $dest 2>/dev/null
cp $first_src/splice_opts $dest 2>/dev/null
cp $first_src/tree $dest 2>/dev/null
cp $first_src/full.mat $dest 2>/dev/null

num_jobs=$(cat $first_src/num_jobs) || exit 1;
for id in $(seq $num_jobs); do
  for in_dir in $*; do cat $in_dir/ali.$id.gz; done > $dest/ali.$id.gz
done

for id in $(seq $num_jobs); do
  for in_dir in $*; do cat $in_dir/trans.$id; done > $dest/trans.$id
done  

for id in $(seq $num_jobs); do
  for in_dir in $*; do cat $in_dir/fst.$id.gz; done > $dest/fst.$id.gz
done  
echo "Combine ali dir done!"

exit 0
