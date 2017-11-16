#!/bin/bash

# Created on 2017-09-05
# Author: Kaituo Xu
# Funciton: Train DNN with frame cross-entropy.

[ -f cmd.sh ] && . ./cmd.sh
[ -f path.sh ] && . ./path.sh

train=data_fbank/train
dev=data_fbank/dev
test=data_fbank/test

train_original=data/train
dev_original=data/dev
test_original=data/test

gmm=exp/tri5a # TODO: make it a argument
conf_dir=local/nnet1/
fbank_conf=$conf_dir/fbank.conf
decode_dnn_conf=$conf_dir/decode_dnn.config

stage=0
. utils/parse_options.sh || exit 1;

# Step 1: Make the FBANK features
[ ! -e $dev ] && if [ $stage -le 0 ]; then
  for data in train dev test; do 
    data_fbank=`eval echo '$'$data`
    data_original=`eval echo '$'${data}_original`
    utils/copy_data_dir.sh $data_original $data_fbank || exit 1; rm $data_fbank/{cmvn,feats}.scp
    steps/make_fbank.sh --nj 10 --cmd "$train_cmd" --fbank-config $fbank_conf \
        $data_fbank $data_fbank/log $data_fbank/data || exit 1;
    steps/compute_cmvn_stats.sh $data_fbank $data_fbank/log $data_fbank/data || exit 1;
  done
fi

# Step 2: Train DNN with frame cross-entropy
dir=exp/dnn-bn6x1024-lr.004 # TODO: argument
if [ $stage -le 2 ]; then
  ali=${gmm}_ali
  dev_ali=${gmm}_dev_ali

  [ ! -d $dir ] && mkdir -p $dir

  num_tgt=$(hmm-info --print-args=false $ali/final.mdl | grep pdfs | awk '{ print $NF }')

  nnet_proto=$dir/nnet.proto
  cat > $nnet_proto << EOF
<AffineTransform> <InputDim> 440 <OutputDim> 1024 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.037344 <MaxNorm> 0.000000
<BatchNormComponent> <InputDim> 1024 <OutputDim> 1024
<Sigmoid> <InputDim> 1024 <OutputDim> 1024
<AffineTransform> <InputDim> 1024 <OutputDim> 1024 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.109375 <MaxNorm> 0.000000
<BatchNormComponent> <InputDim> 1024 <OutputDim> 1024
<Sigmoid> <InputDim> 1024 <OutputDim> 1024
<AffineTransform> <InputDim> 1024 <OutputDim> 1024 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.109375 <MaxNorm> 0.000000
<BatchNormComponent> <InputDim> 1024 <OutputDim> 1024
<Sigmoid> <InputDim> 1024 <OutputDim> 1024
<AffineTransform> <InputDim> 1024 <OutputDim> 1024 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.109375 <MaxNorm> 0.000000
<BatchNormComponent> <InputDim> 1024 <OutputDim> 1024
<Sigmoid> <InputDim> 1024 <OutputDim> 1024
<AffineTransform> <InputDim> 1024 <OutputDim> 1024 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.109375 <MaxNorm> 0.000000
<BatchNormComponent> <InputDim> 1024 <OutputDim> 1024
<Sigmoid> <InputDim> 1024 <OutputDim> 1024
<AffineTransform> <InputDim> 1024 <OutputDim> 1024 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.109375 <MaxNorm> 0.000000
<BatchNormComponent> <InputDim> 1024 <OutputDim> 1024
<Sigmoid> <InputDim> 1024 <OutputDim> 1024
<AffineTransform> <InputDim> 1024 <OutputDim> $num_tgt <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.077845 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000
<Softmax> <InputDim> $num_tgt <OutputDim> $num_tgt
EOF

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --splice 5 --cmvn-opts "--norm-means=true --norm-vars=true" --copy-feats false \
      --nnet-proto $nnet_proto \
      --learn-rate 0.004 --scheduler-opts "--momentum 0.9 --halving-factor 0.5" \
      --train-tool "nnet-train-frmshuff" \
      --train-tool-opts "--minibatch-size=256" \
    $train $dev data/lang $ali $dev_ali $dir || exit 1;
fi

# Step 3: Decode
if [ $stage -le 3 ]; then
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config $decode_dnn_conf --acwt 0.1 \
    $gmm/graph $test $dir/decode_test || exit 1;
fi
