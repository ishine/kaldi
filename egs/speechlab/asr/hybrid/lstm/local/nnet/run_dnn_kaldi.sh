#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a DNN on top of fMLLR features. 
# The training is done in 3 stages,
#
# 1) RBM pre-training:
#    in this unsupervised stage we train stack of RBMs, 
#    a good starting point for frame cross-entropy trainig.
# 2) frame cross-entropy training:
#    the objective is to classify frames to correct pdfs.
# 3) sequence-training optimizing sMBR: 
#    the objective is to emphasize state-sequences with better 
#    frame accuracy w.r.t. reference alignment.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

# Config:
gmm=exp/tri3b
data_fmllr=data-fmllr-tri3b
stage=0 # resume training with --stage=N
# End of config.
. utils/parse_options.sh || exit 1;
#
false && \
{
if [ $stage -le 0 ]; then
  # Store fMLLR features, so we can train on them easily,
  # test
  dir=$data_fmllr/test
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir $gmm/decode \
     $dir data/test $gmm $dir/log $dir/data || exit 1
  # train
  dir=$data_fmllr/train
  steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmm}_ali \
     $dir data/train $gmm $dir/log $dir/data || exit 1
  # split the data : 90% train 10% cross-validation (held-out)
  utils/subset_data_dir_tr_cv.sh $dir ${dir}_tr90 ${dir}_cv10 || exit 1
fi
}



cuda_cmd=run.pl
data_fmllr=data/train300
ali=exp/tri_gmm_ali
dir=exp/dnn4b_pretrain-dbn

false && \
{
if [ $stage -le 1 ]; then
  # Pre-train DBN, i.e. a stack of RBMs (small database, smaller DNN)
  (tail --pid=$$ -F $dir/log/pretrain_dbn.log 2>/dev/null)& # forward log
  $cuda_cmd $dir/log/pretrain_dbn.log \
    steps/nnet/pretrain_dbn.sh --hid-dim 2048 --rbm-iter 2 $data_fmllr/train300_tr90 $dir || exit 1;
fi
}

ali=exp/tri_gmm_ali
dbndir=exp/dnn4b_pretrain-dbn
dir=exp/dnn4b_pretrain-dbn_dnn_baseline
#false && \
{
if [ $stage -le 2 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  #feature_transform=$dbndir/final.feature_transform
  dbn=$dbndir/6.dbn #--dbn $dbn
  feature_transform=$dbndir/final.feature_transform
  nnet=exp/tri_dnn/svd0.55_iter1.nnet
  (tail --pid=$$ -F $dir/log/train_nnet.log 2>/dev/null)& # forward log
  # Train --dbn $dbn --hid-layers 0
  #$cuda_cmd $dir/log/train_nnet.log \
  #  steps/nnet/train.sh --copy-feats false --feature-transform $feature_transform --nnet-init $nnet --learn-rate 0.00004 \
  #  $data_fmllr/train_tr90 $data_fmllr/train_cv10 data/lang $ali $ali $dir || exit 1;
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_asgd.sh --copy-feats false --feature-transform $feature_transform --dbn $dbn --hid-layers 0 --learn-rate 0.0008 --train-opts "--verbose 1" \
    $data_fmllr/train300_tr90 $data_fmllr/train300_cv10 data/lang $ali $ali $dir || exit 1;
fi
}

lang=/sgfs/users/wd007/asr/baseline_chn_7000h/sequence/s5/data/lang
#dir=exp/dnn4b_pretrain-dbn_dnn_htk
#dir=exp/tri_dnn
#ai_mcsnor_evl13jun_v1  ai_mcsntr_evl13mar_v1  ai_mcsntr_evl14jan_v1  ai_mcsntr_evl14mar_v1

false && \
{
  # Decode (reuse HCLG graph)
  for test in ai_mcsnor_evl13jun_v1  ai_mcsntr_evl13mar_v1  ai_mcsntr_evl14jan_v1 ai_mcsntr_evl14mar_v1; do
	for acwt in 0.05;do
  	steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir\
    	--nnet $dir/final.nnet $lang data/test/$test $dir/decode/$test"_"$acwt || exit 1;
	done
  done
}


false && \
{
# Sequence training using sMBR criterion, we do Stochastic-GD 
# with per-utterance updates. For RM good acwt is 0.2
dir=exp/dnn4b_pretrain-dbn_dnn_smbr
srcdir=exp/dnn4b_pretrain-dbn_dnn
acwt=0.2

if [ $stage -le 3 ]; then
  # First we generate lattices and alignments:
  steps/nnet/align.sh --nj 20 --cmd "$train_cmd" \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali || exit 1;
  steps/nnet/make_denlats.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
    $data_fmllr/train data/lang $srcdir ${srcdir}_denlats || exit 1;
fi
}

false && \
{
if [ $stage -le 4 ]; then
  # Re-train the DNN by 6 iterations of sMBR 
  steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 6 --acwt $acwt --do-smbr true \
    $data_fmllr/train data/lang $srcdir ${srcdir}_ali ${srcdir}_denlats $dir || exit 1
  # Decode
  for ITER in 1 2 3 4 5 6; do
    steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config \
      --nnet $dir/${ITER}.nnet --acwt $acwt \
      $gmm/graph $data_fmllr/test $dir/decode_it${ITER} || exit 1
  done 
fi
}

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done

# to see how model conversion to nnet2 works, run run_dnn_convert_nnet2.sh at this point.

