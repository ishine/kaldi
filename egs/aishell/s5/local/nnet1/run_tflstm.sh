#!/bin/bash

# Created on 2018-04-02
# Author: Kaituo Xu
# Function: Train TF-LSTM using fbank features and truncated BPTT.

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

# Step 2: Train LSTM with truncated BPTT
if [ $stage -le 2 ]; then
  dir=exp/tflstm1-C24F8S1-lstm4x1024x512-lr0.00004-mb128
  ali=${gmm}_ali
  dev_ali=${gmm}_dev_ali

  mkdir -p $dir
  echo "<Splice> <InputDim> 40 <OutputDim> 40 <BuildVector> 5 </BuildVector>" > $dir/delay5.proto

  nnet_proto=$dir/nnet.proto
  cat > $nnet_proto << EOF
<TfLstm> <InputDim> 40 <OutputDim> 792 <CellDim> 24 <F> 8 <S> 1
<LstmProjected> <InputDim> 792 <OutputDim> 512 <CellDim> 1024
<LstmProjected> <InputDim> 512 <OutputDim> 512 <CellDim> 1024
<LstmProjected> <InputDim> 512 <OutputDim> 512 <CellDim> 1024
<LstmProjected> <InputDim> 512 <OutputDim> 512 <CellDim> 1024
<Tanh> <InputDim> 512 <OutputDim> 512
<AffineTransform> <InputDim> 512 <OutputDim> 2952 <BiasMean> 0.0 <BiasRange> 0.0
<Softmax> <InputDim> 2952 <OutputDim> 2952
EOF


  # Train
  $cuda_cmd $dir/log/train_nnet.log \
    steps/nnet/train.sh \
      --cmvn-opts "--norm-means=true --norm-vars=true" --copy-feats false \
      --feature-transform-proto $dir/delay5.proto \
      --nnet-proto $nnet_proto \
      --learn-rate 0.00004 --scheduler-opts "--momentum 0.9 --halving-factor 0.5" \
      --train-tool "nnet-train-multistream" \
      --train-tool-opts "--num-streams=128 --batch-size=20" \
    $train $dev data/lang $ali $dev_ali $dir || exit 1;
fi

# Step 3: Decode
if [ $stage -le 3 ]; then
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config $decode_dnn_conf --acwt 0.1 \
    $gmm/graph $test $dir/decode_test || exit 1;
fi
