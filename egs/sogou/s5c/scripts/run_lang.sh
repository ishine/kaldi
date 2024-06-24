#!/bin/bash

# This scripts is to train GMM and DNN(chain) models with sogou data
# This setup is modified from egs/swbd
# Date: Fri Sep 21 2017   -- WangZhichao
####################################################################################

. cmd.sh
. path.sh
set -e # exit on error

stage=1
train_nn_stage=12
train_LM=false
# Prepare sogou Acoustic data and Language data first:
# 1. Put original training data and test data(including: "utt2spk, wav.scp , text"; "segments" is optional) 
# under the path: data/local/train
# 2. Put dict data (including: "extra_questions.txt, lexicon.txt, nonsilence_phones.txt, optional_silence.txt, \\
# phones.txt, silence_phones.txt") under the path: data/local/dict
# check data/local/train dir
###utils/fix_data_dir.sh data/local/train

# Now prepare the "lang" data and train LM if train_LM=true. 
LM=data/lang_0528/lm.arpa.gz
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"

if [ $stage -le 1 ]; then
#  utils/prepare_lang.sh --position-dependent-phones false data/local/dict_online_if \
#  '!SIL'  data/local/lang data/lang_online_if
  # Compiles G for swbd trigram LM
  utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
    data/lang_0528 $LM data/local/dict_0528/lexicon.txt data/lang_0528_G
fi
