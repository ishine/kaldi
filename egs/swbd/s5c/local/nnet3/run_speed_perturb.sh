#!/bin/bash

. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true
train_set=train_sogou_500h

. ./path.sh
. ./utils/parse_options.sh

mkdir -p nnet3
# perturbed data preparation

if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    #Although the nnet will be trained by high resolution data, we still have to perturbe the normal data to get the alignment
    # _sp stands for speed-perturbed

      utils/perturb_data_dir_speed.sh 1.3 data/$train_set data/${train_set}_tmp 
      utils/validate_data_dir.sh --no-feats data/${train_set}_tmp

      mfccdir=mfcc_sp_500h_1.3
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 \
       data/${train_set}_tmp exp/make_mfcc/${train_set}_tmp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh data/${train_set}_tmp exp/make_mfcc/${train_set}_tmp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${train_set}_tmp

#      utils/copy_data_dir.sh --spk-prefix sp1.0- --utt-prefix sp1.0- data/${train_set} data/temp0
#      utils/combine_data.sh data/${train_set}_sp data/${train_set}_tmp data/temp0
#      utils/fix_data_dir.sh data/${train_set}_sp
#      rm -r data/temp0 data/${datadir}_tmp
  fi


#  if [ $stage -le 2 ] && [ "$generate_alignments" == "true" ]; then
    #obtain the alignment of the perturbed data
#    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
#      data/train_nodup_sp data/lang exp/tri4 exp/tri4_ali_nodup_sp || exit 1
#  fi
  train_set=${train_set}_sp
fi

echo "Success perturb data by 1.1 speed."
exit 0;
# we do not do volume perturb as we will use cmvn in the training procedure

if [ $stage -le 3 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/swbd-$date/s5b/$mfccdir/storage $mfccdir/storage
  fi

  # the 100k_nodup directory is copied seperately, as
  # we want to use exp/tri2_ali_100k_nodup for lda_mllt training
  # the main train directory might be speed_perturbed
  for dataset in $train_set train_100k_nodup; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done

  for dataset in eval2000 train_dev $maybe_rt03; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done

fi

exit 0;
