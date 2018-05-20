#!/bin/bash

# Created on 2017-10-19
# Author: Kaituo XU
# Function: Verify whether my SRU impl is right or not.
#   If my impl is right, in the 10 utterances dev set, this SRU model
#   should overfit, i.e. the frame acc should approximate 100% and
#   the loss should approximate 0.
#   When set max_iters=70, I get frame acc = 97%+, loss = 0.097. The impl has
#   no big problem.

. ./cmd.sh
[ -f path.sh ] && . ./path.sh

DEV_DATA=/home/t-kax/kaldi-workspace/pytorch-ASR-AM/data_fbank
DIR=exp/dev_fsmn5
max_iters=300

[ ! -d $DIR ] && mkdir -p $DIR $DIR/log $DIR/nnet

nnet_proto=$DIR/nnet.proto
cat > $nnet_proto <<EOF
<AffineTransform> <InputDim> 40 <OutputDim> 512 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.028646 <MaxNorm> 0.000000
<CompactVfsmn> <InputDim> 512 <OutputDim> 512 <Order> 10
<AffineTransform> <InputDim> 512 <OutputDim> 2048 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.028646 <MaxNorm> 0.000000
<Sigmoid> <InputDim> 2048 <OutputDim> 2048
<AffineTransform> <InputDim> 2048 <OutputDim> 2952 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 0.070000 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000
<Softmax> <InputDim> 2952 <OutputDim> 2952
EOF

nnet_init=$DIR/nnet.init
nnet-initialize $nnet_proto $nnet_init

nnet_last=$nnet_init
for iter in $(seq -w $max_iters); do
  nnet_next=$DIR/nnet_iter${iter}
  log=$DIR/log/tr.log  # iter${iter}.tr.log
  nnet-train-fsmn --cross-validate=false --randomize=false --verbose=0 --minibatch-size=32 --learn-rate=0.00004 --momentum=0.9 --l1-penalty=0 --l2-penalty=0 "scp:$DEV_DATA/dev.scp" "ark:ali-to-post ark,t:$DEV_DATA/dev.ali ark:- |" $nnet_last $nnet_next 2>> $log
  nnet_last=$nnet_next
done
