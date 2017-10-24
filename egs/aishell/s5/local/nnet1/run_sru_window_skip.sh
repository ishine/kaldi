#!/bin/bash

# Created on 2017-10-23
# Author: Kaituo Xu
# Function: Train SimpleRecurrentUnit using fbank features and truncated BPTT,
#   and skipping frame.

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

# Step 2: Train SRU with truncated BPTT
dir=exp/sru9x512+1024-l0r4-skip1
if [ $stage -le 2 ]; then
  ali=${gmm}_ali
  dev_ali=${gmm}_dev_ali

  mkdir -p $dir
  echo "<Splice> <InputDim> 40 <OutputDim> 200 <BuildVector> 0:4 </BuildVector>" > $dir/delay5.proto
  echo "<Splice> <InputDim> 200 <OutputDim> 200 <BuildVector> 5 </BuildVector>" >> $dir/delay5.proto

  nnet_proto=$dir/nnet.proto
  cat > $nnet_proto << EOF
<SimpleRecurrentUnit> <InputDim> 200 <OutputDim> 512 <CellDim> 512
<SimpleRecurrentUnit> <InputDim> 512 <OutputDim> 512 <CellDim> 512
<SimpleRecurrentUnit> <InputDim> 512 <OutputDim> 512 <CellDim> 512
<SimpleRecurrentUnit> <InputDim> 512 <OutputDim> 512 <CellDim> 512
<SimpleRecurrentUnit> <InputDim> 512 <OutputDim> 512 <CellDim> 512
<SimpleRecurrentUnit> <InputDim> 512 <OutputDim> 512 <CellDim> 512
<SimpleRecurrentUnit> <InputDim> 512 <OutputDim> 512 <CellDim> 512
<SimpleRecurrentUnit> <InputDim> 512 <OutputDim> 512 <CellDim> 512
<SimpleRecurrentUnit> <InputDim> 512 <OutputDim> 512 <CellDim> 512
<SimpleRecurrentUnit> <InputDim> 512 <OutputDim> 1024 <CellDim> 1024
<AffineTransform> <InputDim> 1024 <OutputDim> 3019 <BiasMean> 0.0 <BiasRange> 0.0
<Softmax> <InputDim> 3019 <OutputDim> 3019
EOF

  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" --copy-feats false \
      --feature-transform-proto $dir/delay5.proto \
      --nnet-proto $nnet_proto \
      --learn-rate 0.000001 --scheduler-opts "--momentum 0.9 --halving-factor 0.5" \
      --train-tool "nnet-train-multistream-skip" \
      --train-tool-opts "--num-streams=128 --batch-size=20 --num-skip-frames=1" \
    $train $dev data/lang $ali $dev_ali $dir || exit 1;
fi

# Step 3: Decode
if [ $stage -le 3 ]; then
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config $decode_dnn_conf --acwt 0.1 \
    --nnet-forward-opts "--no-softmax=true --prior-scale=1.0 --num-skip-frames=1" \
    $gmm/graph $test $dir/decode_test || exit 1;
fi
