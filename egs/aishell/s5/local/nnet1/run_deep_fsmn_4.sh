#!/bin/bash

# Created on 2018-05-23
# Author: Kaituo Xu
# Funciton: Train DeepFsmn with frame cross-entropy.

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

lr=0.00001
backorder=20  # DeepFSMN lookback order, use --backorder N
aheadorder=20  # DeepFSMN lookahead order, use --aheadorder N

stage=0
. utils/parse_options.sh || exit 1;

dir=exp/DeepFsmn-3x40-4x_2048-512-${backorder}-${aheadorder}_2x2048-512-lr$lr

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

# Step 2: Train FSMN with frame cross-entropy
if [ $stage -le 2 ]; then
  ali=${gmm}_ali
  dev_ali=${gmm}_dev_ali

  mkdir -p $dir/log

  num_tgt=$(hmm-info --print-args=false $ali/final.mdl | grep pdfs | awk '{ print $NF }')

  # DeepFsmn config: 3x40-4x_2048-512-20-20_2x2048-512
  #
  # DeepFsmn = [input -> Affine+ReLU -> Affine -> vFSMN -> output]
  #               |                                 ^
  #               |---------------------------------|
  #

  nnet_proto=$dir/nnet.proto
  cat > $nnet_proto << EOF
<AffineTransform> <InputDim> 120 <OutputDim> 2048 <Xavier> 1
<ReLU> <InputDim> 2048 <OutputDim> 2048
<AffineTransform> <InputDim> 2048 <OutputDim> 512 <Xavier> 1
<BiCompactVfsmn> <InputDim> 512 <OutputDim> 512 <BackOrder> $backorder <AheadOrder> $aheadorder
<DeepFsmn> <InputDim> 512 <OutputDim> 512 <HiddenSize> 2048 <BackOrder> $backorder <AheadOrder> $aheadorder
<DeepFsmn> <InputDim> 512 <OutputDim> 512 <HiddenSize> 2048 <BackOrder> $backorder <AheadOrder> $aheadorder
<DeepFsmn> <InputDim> 512 <OutputDim> 512 <HiddenSize> 2048 <BackOrder> $backorder <AheadOrder> $aheadorder
<AffineTransform> <InputDim> 512 <OutputDim> 2048 <Xavier> 1
<ReLU> <InputDim> 2048 <OutputDim> 2048 
<AffineTransform> <InputDim> 2048 <OutputDim> 2048 <Xavier> 1
<ReLU> <InputDim> 2048 <OutputDim> 2048 
<AffineTransform> <InputDim> 2048 <OutputDim> 512 <Xavier> 1
<AffineTransform> <InputDim> 512 <OutputDim> $num_tgt <Xavier> 1
<Softmax> <InputDim> $num_tgt <OutputDim> $num_tgt
EOF

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --splice 1 --cmvn-opts "--norm-means=true --norm-vars=true" --copy-feats false \
      --nnet-proto $nnet_proto \
      --learn-rate $lr --scheduler-opts "--momentum 0.9 --halving-factor 0.5" \
      --train-tool "nnet-train-fsmn" \
      --train-tool-opts "--minibatch-size=4096" \
    $train $dev data/lang $ali $dev_ali $dir || exit 1;
fi

# Step 3: Decode
if [ $stage -le 3 ]; then
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config $decode_dnn_conf --acwt 0.1 \
    $gmm/graph $test $dir/decode_test || exit 1;
fi
