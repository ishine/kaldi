#!/bin/bash

# Begin configuration section.
cmd=run.pl
stage=0
extra_files=
num_jobs=20
max_job_run=20
# End configuration section.
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 [options] <data> <dest-lats-dir> <src-lats-dir1> <src-lats-dir2> ..."
  echo "e.g.: $0 --num-jobs 32 data/train exp/chain/lstm_denlats_combined exp/chain/lstm_denlats_1 exp/chain/lstm_denlats_2"
  echo "Options:"
  echo " --extra-files <file1 file2...>   # specify addtional files in 'src-denlats-dir1' to copy"
  echo " --num-jobs <nj>                  # number of jobs used to split the data directory."
  echo " Note, files that don't appear in the first source dir will not be added even if they appear in later ones."
  echo " Other than denlats, only files from the first src ali dir are copied."
  exit 1;
fi

data=$1;
shift;
dest=$1;
shift;
first_src=$1;

mkdir -p $dest;
rm $dest/{lat.*.gz,num_jobs} 2>/dev/null

export LC_ALL=C

for dir in $*; do
  if [ ! -f $dir/lat.1.gz ]; then
    echo "$0: check if alignments (ali.*.gz) are present in $dir."
    exit 1;
  fi
done

for dir in $*; do
  for f in tree; do
    diff $first_src/$f $dir/$f 1>/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "$0: Cannot combine lats directories with different $f files."
    fi
  done
done

for f in cmvn_opts num_jobs final.alimdl final.mat final.mdl final.occs full.mat phones.txt splice_opts tree $extra_files; do
  if [ ! -f $first_src/$f ]; then
    echo "combine_denlats_dir.sh: no such file $first_src/$f"
    exit 1;
  fi
  cp $first_src/$f $dest/
done

dest_temp_dir=$dest/temp
if [ $stage -le 0 ]; then
  echo "$0: dumping denlats in each source directory and combine the lat.scp"
  [ -d $dest_temp_dir ] && rm -r $dest_temp_dir;
  mkdir -p $dest_temp_dir
  [ -f $dest/lats.scp ] && rm -f $dest/lats.scp;
  for dir in $*; do
    temp_dir=$dir/temp
    [ -d $temp_dir ] && rm -r $temp_dir;
    mkdir -p $temp_dir
    cur_num_jobs=$(cat $dir/num_jobs) || exit 1;
    $cmd --max-jobs-run $max_job_run JOB=1:$cur_num_jobs $temp_dir/log/lattice_copy.JOB.log \
      lattice-copy "ark:gunzip -c $dir/lat.JOB.gz |" \
      ark,scp:$temp_dir/lat.JOB.ark,$temp_dir/lat.JOB.scp || exit 1;
    for id in $(seq $cur_num_jobs); do cat $temp_dir/lat.$id.scp; done >> $dest_temp_dir/combined_lats.scp
  done
fi

sort -u $dest_temp_dir/combined_lats.scp > $dest_temp_dir/combined_lats_sorted.scp || exit 1;

echo "$0: splitting data to get reference utt2spk for individual lat.JOB.gz files."
utils/split_data.sh $data $num_jobs || exit 1;

echo "$0: splitting the denlats to appropriate chunks according to the reference utt2spk files."
$cmd --max-jobs-run $max_job_run JOB=1:$num_jobs $dest_temp_dir/log/lattice_copy.JOB.log \
  lattice-copy --include=$data/split$num_jobs/JOB/feats.scp scp:$dest_temp_dir/combined_lats_sorted.scp "ark:|gzip -c >$dest/lat.JOB.gz" || exit 1;
echo $num_jobs > $dest/num_jobs  || exit 1

echo "$0: checking the denlats files generated have at least 90% of the utterances."
num_lines=`cat $dest_temp_dir/combined_lats_sorted.scp | wc -l` || exit 1;
num_lines_tot=`cat $data/utt2spk | wc -l` || exit 1;
python -c "import sys;
percent = 100.0 * float($num_lines) / $num_lines_tot
if percent < 90 :
  print ('$dest/lat.*.gz {0}% utterances missing.'.format(percent))"  || exit 1;
for dir in $*; do
  rm -r $dir/temp 2>/dev/null
done

rm -r $dest_temp_dir 2>/dev/null

echo "Combined denlats and stored in $dest"
exit 0
