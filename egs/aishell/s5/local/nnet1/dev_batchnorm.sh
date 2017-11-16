#!/bin/bash

# Created on 2017-10-27
# Author: Kaituo XU
# Function: Verify whether my Batch Normalization impl is right or not.
#   If my impl is right, in the 10 utterances dev set, this model
#   should overfit, i.e. the frame acc should approximate 100% and
#   the loss should approximate 0.
#   When set max_iters=70, I get frame acc = 97%+, loss = 0.097. The impl has
#   no big problem.

. ./cmd.sh
[ -f path.sh ] && . ./path.sh

DEV_DATA=/search/speech/xukaituo/pytorch-workspace/ASR-AM/data
DIR=exp/dev_batchnorm
max_iters=170

[ ! -d $DIR ] && mkdir -p $DIR $DIR/log $DIR/nnet

nnet_proto=$DIR/nnet.proto
cat > $nnet_proto <<EOF
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
<AffineTransform> <InputDim> 1024 <OutputDim> 3019 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.077845 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000
<Softmax> <InputDim> 3019 <OutputDim> 3019
EOF

nnet_init=$DIR/nnet.init
nnet-initialize $nnet_proto $nnet_init

nnet_last=$nnet_init
for iter in $(seq -w $max_iters); do
  nnet_next=$DIR/nnet_iter${iter}
  log=$DIR/log/tr.log  # iter${iter}.tr.log
  nnet-train-frmshuff --cross-validate=false --randomize=true --verbose=0 --minibatch-size=2 --learn-rate=0.0004 --momentum=0.9 --l1-penalty=0 --l2-penalty=0 --feature-transform=exp/dnn6x1024/final.feature_transform "scp:$DEV_DATA/dev.scp" "ark:ali-to-post ark,t:$DEV_DATA/tr_ali.txt ark:- |" $nnet_last $nnet_next 2>> $log
  nnet_last=$nnet_next
done
