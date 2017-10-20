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

DEV_DATA=/search/speech/xukaituo/pytorch-workspace/ASR-AM/data
DIR=exp/dev_sru
max_iters=70

[ ! -d $DIR ] && mkdir -p $DIR $DIR/log $DIR/nnet

nnet_proto=$DIR/nnet.proto
cat > $nnet_proto <<EOF
<SimpleRecurrentUnit> <InputDim> 40 <OutputDim> 400 <CellDim> 400
<SimpleRecurrentUnit> <InputDim> 400 <OutputDim> 400 <CellDim> 400 
<SimpleRecurrentUnit> <InputDim> 400 <OutputDim> 400 <CellDim> 400 
<AffineTransform> <InputDim> 400 <OutputDim> 3019 <BiasMean> 0.0 <BiasRange> 0.0
<Softmax> <InputDim> 3019 <OutputDim> 3019
EOF

nnet_init=$DIR/nnet.init
nnet-initialize $nnet_proto $nnet_init

nnet_last=$nnet_init
for iter in $(seq -w $max_iters); do
  nnet_next=$DIR/nnet_iter${iter}
  log=$DIR/log/tr.log  # iter${iter}.tr.log
  nnet-train-multistream --cross-validate=false --randomize=true --verbose=0 --num-streams=2 --batch-size=20 --learn-rate=0.0001 --momentum=0.9 --l1-penalty=0 --l2-penalty=0 "scp:$DEV_DATA/dev.scp" "ark:ali-to-post ark,t:$DEV_DATA/tr_ali.txt ark:- |" $nnet_last $nnet_next 2>> $log
  nnet_last=$nnet_next
done
