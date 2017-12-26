#!/bin/bash

# This scripts is to train GMM and DNN(chain) models with sogou data
# This setup is modified from egs/swbd
# Date: Fri Sep 21 2017   -- WangZhichao
####################################################################################

. cmd.sh
. path.sh
set -e # exit on error

# Prepare sogou Acoustic data and Language data first:
# 1. Put original training data and test data(including: "utt2spk, wav.scp , text"; "segments" is optional) 
# under the path: data/local/train
# 2. Put dict data (including: "extra_questions.txt, lexicon.txt, nonsilence_phones.txt, optional_silence.txt, \\
# phones.txt, silence_phones.txt") under the path: data/local/dict
# check data/local/train dir
###utils/fix_data_dir.sh data/local/train


# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc/mfcc_1300h
# In this setup, we use MFCC feature to train GMM and FBANK to train NN  
  for x in train_mfcc_1300h ; do
    steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" \
      data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done
