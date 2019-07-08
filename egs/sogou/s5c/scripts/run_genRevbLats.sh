
set -e

# configs for 'chain'
affix=
stage=1
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cmd=queue.pl

data=data/train_sogou_fbank_2400h_farfield_ori_clean_SimuRIR_isotropic_noise
clean_lat_dir=exp/tri3b_2400h_farfield_ori_clean_lats
lat_dir=exp/tri3b_2400h_farfield_ori_clean_lats_SimuRIR
num_lat_jobs=$(cat $clean_lat_dir/num_jobs) || exit 1;
max_job_run=$num_lat_jobs
prefix=rev1

sdata=$data/split$num_lat_jobs
utils/split_data.sh $data $num_lat_jobs

if [ $stage -le 1 ]; then
  # Create the lattices for the reverberated data
  # We use the lattices/alignments from the clean data for the reverberated data.
  mkdir -p $lat_dir/temp/
  $cmd --max-jobs-run $max_job_run JOB=1:$num_lat_jobs $lat_dir/log/lattice_copy.JOB.log \
    lattice-copy "ark:gunzip -c $clean_lat_dir/lat.JOB.gz|" ark,scp:$lat_dir/temp/lat.JOB.ark,$lat_dir/temp/lat.JOB.scp || exit 1;

  for id in $(seq $num_lat_jobs); do cat $lat_dir/temp/lat.$id.scp; done > $lat_dir/temp/lats.scp
#  lattice-copy "ark:gunzip -c $clean_lat_dir/lat.*.gz |" ark,scp:$lat_dir/temp/lats.ark,$lat_dir/temp/lats.scp

  # copy the lattices for the reverberated data
  # Here prefix "rev0_" represents the clean set, "rev1_" represents the reverberated set
#  for i in `seq 1 1`; do
#    cat $lat_dir/temp/lats.scp | sed -e "s/^/rev${i}_/" >> $lat_dir/temp/combined_lats.scp
#  done
  cat $lat_dir/temp/lats.scp | sed -e "s/^/${prefix}_/" >> $lat_dir/temp/combined_lats.scp
  sort -u $lat_dir/temp/combined_lats.scp > $lat_dir/temp/combined_lats_sorted.scp

  $cmd --max-jobs-run $max_job_run JOB=1:$num_lat_jobs $lat_dir/log/lattice_copy.JOB.log \
    lattice-copy --include=$sdata/JOB/feats.scp scp:$lat_dir/temp/combined_lats_sorted.scp "ark:|gzip -c >$lat_dir/lat.JOB.gz" || exit 1;
  echo "$num_lat_jobs" > $lat_dir/num_jobs

  # copy other files from original lattice dir
  for f in cmvn_opts final.mdl splice_opts tree; do
    cp $clean_lat_dir/$f $lat_dir/$f
  done

  rm $lat_dir/temp -rf
fi


