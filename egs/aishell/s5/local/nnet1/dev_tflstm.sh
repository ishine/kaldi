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
DIR=exp/dev_tflstm
max_iters=300

[ ! -d $DIR ] && mkdir -p $DIR $DIR/log $DIR/nnet

nnet_proto=$DIR/nnet.proto
cat > $nnet_proto <<EOF
<TfLstm> <InputDim> 40 <OutputDim> 792 <CellDim> 24 <F> 8 <S> 1
<AffineTransform> <InputDim> 792 <OutputDim> 2952 <BiasMean> 0.0 <BiasRange> 0.0
<Softmax> <InputDim> 2952 <OutputDim> 2952
EOF

nnet_init=$DIR/nnet.init
nnet-initialize $nnet_proto $nnet_init

nnet_last=$nnet_init
for iter in $(seq -w $max_iters); do
  nnet_next=$DIR/nnet_iter${iter}
  log=$DIR/log/tr.log  # iter${iter}.tr.log
  nnet-train-multistream --cross-validate=false --randomize=true --verbose=0 --num-streams=2 --batch-size=20 --learn-rate=0.00004 --momentum=0.9 --l1-penalty=0 --l2-penalty=0 "scp:$DEV_DATA/dev.scp" "ark:ali-to-post ark,t:$DEV_DATA/tr_ali.txt ark:- |" $nnet_last $nnet_next 2>> $log
  nnet_last=$nnet_next
done
