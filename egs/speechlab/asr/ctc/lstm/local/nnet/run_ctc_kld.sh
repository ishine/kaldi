#!/bin/bash

# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains a LSTM network on FBANK features.
# The LSTM code comes from Yiayu DU, and Wei Li, thanks!

. ./cmd.sh
. ./path.sh

stage=0
. utils/parse_options.sh || exit 1;

cuda_cmd=run.pl
dir=exp/plstm_digit_kld
ali=exp/digit_ctc_ali
train=data/train_digit_aug/trainctc

#false && \
{
if [ $stage -le 1 ]; then
  # Train the DNN optimizing per-frame cross-entropy.
  # Train
  nnet=$dir/nnet.init
  si_model=$dir/si.nnet
  # --nnet-proto $dir/nnet.proto
  # --nnet-init $nnet
  # --cmvn-opts "--norm-means=true --norm-vars=false"
  feature_transform=$dir/final.feature_transform
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train_ctc.sh --network-type lstm --sort-by-len false --learn-rate 0.000005 --nnet-init $nnet --si-model $si_model \
      --kld-scale 0.3 --splice 0 --online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false" \
      --feature-transform $feature_transform --skip-frames 3 --skip-inner true \
      --train-tool "nnet-train-ctc-parallel  --num-stream=15 --max-frames=15000 --batch-size=120 --objective-function=\"ctc\" " \
    ${train}_tr90 ${train}_cv10 data/lang $ali $ali $dir || exit 1;

  #--online true --cmvn-opts "--cmn-window=10000 --min-cmn-window=100 --norm-vars=false"
  #--delta_opts "--delta-order=2" --batch-size=20
  # --splice 0  --batch-size=20
fi
}

decode_cmd=run.pl

data=data/test_set
lang=data/lang_decode
#ai_mcsnor_evl13jun_v1  ai_mcsntr_evl13mar_v1  ai_mcsntr_evl14jan_v1  ai_mcsntr_evl14mar_v1
#acar_cloud_asr_160106_hud_vst  alihome_x1000_1601  mixed_add_noise
#acar_hud_comm_1512_v1  car_noise_shanghai afar_kinect_tune15mar_v1 acar_robot_tune15jun_v2
#afar_uca_tune15dec_v1 car_noise_shanghai


opts="--prior-scale=0.0"
false && \
{
  for test in ai_mcsntr_evl13mar_v1 ai_mcsnor_evl13jun_v1 ai_mcsntr_evl14jan_v1 mixed_add_noise alihome_x1000_1601 afar_uca_tune15dec_v1 acar_hud_comm_1512_v1;do
	for acwt in 0.5; do
  # Decode (reuse HCLG graph)
  steps/nnet/decode_ctc.sh --nj 30 --beam 17.0 --lattice_beam 8.0 --max-active 2000 --acwt $acwt \
     --cmd "$decode_cmd" --config conf/decode_dnn.config --srcdir $dir \
    $lang $data/$test $dir/decode/${test}_"$acwt" || exit 1;
  #steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir  \
  #  --nnet-forward-opts "--no-softmax=true --prior-scale=1.0 --time-shift=5" \
  #  $lang $data/test_lstm/$test $dir/decode_time-shift5/${test}_"$acwt" || exit 1;
	done
  done
}

echo Success
exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
