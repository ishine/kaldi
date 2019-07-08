#!/bin/bash

# This scripts is to train GMM and DNN(chain) models with sogou data
# This setup is modified from egs/swbd
# Date: Fri Sep 21 2017   -- WangZhichao
####################################################################################

. cmd.sh
. path.sh
set -e # exit on error

stage=3
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
LM=data/local/lm/sw1.o3g.kn.gz
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"

if [ $stage -le 1 ]; then
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
  '!SIL'  data/local/lang data/lang
 # if train_LM=true, we train the language models with the training transcription
  if $train_LM ; then 
    echo "train LM with local transcription"
    local/swbd1_train_lms.sh data/local/train/text \
      data/local/dict/lexicon.txt data/local/lm 
    # Compiles G for swbd trigram LM
    utils/format_lm_sri.sh --srilm-opts "$srilm_opts" \
      data/lang $LM data/local/dict/lexicon.txt data/lang_nosp_sw1_tg
  fi
fi

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
# In this setup, we use MFCC feature to train GMM and FBANK to train NN  
if [ $stage -le 2 ]; then
  cp -r data/local/train data/train_mfcc || exit 1;
  #We do not use test-set but dev-set to test 
#  cp -r data/local/not_on_screen data/not_on_screen_mfcc || exit 1;
#  cp -r data/local/test8000 data/test8000_mfcc || exit 1;
#  cp -r data/local/testIOS data/testIOS_mfcc || exit 1;
  # Modify conf/mfcc.conf to set the MFCC config 
  for x in train_mfcc ; do
    steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" \
      data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done
fi

# Monophone trainging
if [ $stage -le 3 ]; then
  # Use the first 100k sentences to start the monophone training.
  utils/subset_data_dir.sh --first data/train_mfcc 500000 data/train_mfcc_500k_mono || exit 1;
  steps/train_mono.sh --nj 40 --cmd "$train_cmd" \
    data/train_mfcc_500k_mono data/lang_0528_syllable exp/mono || exit 1;

  # use the last 2000 sentenses as dev set.
  utils/subset_data_dir.sh --last data/train_mfcc 2000 data/train_mfcc_dev || exit 1;
  (
    graph_dir=exp/mono/graph
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_0528_syllable exp/mono $graph_dir
    steps/decode_si.sh --nj 5 --cmd "$decode_cmd" --config conf/decode.config \
      $graph_dir data/train_mfcc_dev exp/mono/decode_dev
  ) &
fi

if [ $stage -le 4 ]; then
  #use 1000k sentences to train tri1
  utils/subset_data_dir.sh  data/train_mfcc 1000000 data/train_mfcc_1000k || exit 1;
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train_mfcc_1000k data/lang_0528_syllable exp/mono exp/mono_ali || exit 1;
  
  steps/train_deltas.sh --cmd "$train_cmd" 4000 50000 \
    data/train_mfcc_1000k data/lang_0528_syllable exp/mono_ali exp/tri1 || exit 1;

  (
    graph_dir=exp/tri1/graph
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_0528_syllable exp/tri1 $graph_dir
    steps/decode_si.sh --nj 5 --cmd "$decode_cmd" --config conf/decode.config \
      $graph_dir data/train_mfcc_dev exp/tri1/decode_dev
  ) &
fi

if [ $stage -le 5 ]; then
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train_mfcc_1000k data/lang_0528_syllable exp/tri1 exp/tri1_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
    5000 100000 data/train_mfcc_1000k data/lang_0528_syllable exp/tri1_ali exp/tri2
fi

# From now, we start using all of the data 
if [ $stage -le 6 ]; then 
  steps/align_si.sh --nj 40 --cmd "$train_cmd" \
    data/train_mfcc data/lang_0528_syllable exp/tri2 exp/tri2_ali

  # Do another iteration of LDA+MLLT training, on all the data.
  steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" \
    6000 240000 data/train_mfcc data/lang_0528_syllable exp/tri2_ali exp/tri2b
fi

# Train tri3b, which is LDA+MLLT+SAT, on all the data.
if [ $stage -le 7 ]; then 
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    data/train_mfcc data/lang_0528_syllable exp/tri2b exp/tri2b_ali

  steps/train_sat.sh  --cmd "$train_cmd" \
    10000 600000 data/train_mfcc data/lang_0528_syllable exp/tri2b_ali exp/tri3b
  
  # Get the alignment for NN training
  steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
    data/train_mfcc data/lang_0528_syllable exp/tri3b exp/tri3b_ali
  (
    graph_dir=exp/tri3b/graph
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_0528_syllable exp/tri3b $graph_dir
    steps/decode_fmllr.sh --nj 5 --cmd "$decode_cmd" \
      --config conf/decode.config \
      $graph_dir data/train_mfcc_dev exp/tri3b/decode_dev
  ) &
wait
fi
exit 0;
# Here is for MMI training (later will be added)
if [ $stage -le 8 ]; then
  steps/make_denlats.sh --nj 30 --cmd "$decode_cmd" \
    --config conf/decode.config --transform-dir exp/tri3b_ali \
    data/train_mfcc data/lang exp/tri3b exp/tri4_denlats

  # 4 iterations of MMI seems to work well overall. The number of iterations is
  # used as an explicit argument even though train_mmi.sh will use 4 iterations by
  # default.
fi

if [ $stage -le 9 ]; then
  num_mmi_iters=4
  steps/train_mmi.sh --cmd "$decode_cmd" \
    --boost 0.1 --num-iters $num_mmi_iters \
    data/train_mfcc data/lang exp/tri3b_ali exp/tri4_denlats exp/tri4_mmi_b0.1

  for iter in 1 2 3 4; do
    (
      graph_dir=exp/tri3b/graph
      decode_dir=exp/tri4_mmi_b0.1/decode_dev_${iter}
      steps/decode.sh --nj 6 --cmd "$decode_cmd" \
        --config conf/decode.config --iter $iter \
        --transform-dir exp/tri3b/decode_dev \
        $graph_dir data/train_mfcc_dev $decode_dir
    ) &
  done
fi



if [ $stage -le 10 ]; then 
# nnet3-chain LSTM recipe
# local/chain/run_lstm_sogou.sh

# nnet3-chain TDNN-LSTM recipe
#  local/chain/local/chain/run_tdnn_lstm_sogou_1c.sh --train-stage "$train_nn_stage" 
fi
# getting results (see RESULTS file)
