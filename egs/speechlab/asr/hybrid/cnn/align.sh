#!/bin/bash

. ./cmd.sh
. ./path.sh


. utils/parse_options.sh || exit 1;

decode_cmd=run.pl
cuda_cmd=run.pl
train_cmd=run.pl

train_cmd=queue.pl
false && \
{
steps/align_si.sh --nj 90 --cmd "$train_cmd" \
  data/train_plp data/lang data/lang exp/tri_gmm_ali
}


train_cmd=run.pl

lang=data/lang
acwt=0.1
boost=0.1
srcdir=exp/cnn_relu_baseline
testdir=data/alignment
test_set="comm_split9"

#testdir=/aifs1/users/wd007/asr/baseline_chn_2000h/data/test/test_set
#test_set="accent_test  assistant_test  mandarin_test"

# First we need to generate lattices and alignments:
#false && \
{
for test in $test_set; do
steps/align_nnet_htk.sh --nj 10 --cmd "$train_cmd" \
   $testdir/$test $lang $srcdir ${srcdir}/alignment/$test || exit 1;
. path.sh
ali-to-phones --write-lengths=true ${srcdir}/final.mdl scp:${srcdir}/alignment/$test/ali.scp ark,t:${srcdir}/alignment/$test/ali.phone.length.txt
done
}
