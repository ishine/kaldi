#!/bin/bash

. path.sh
. cmd.sh


decode_cmd="queue.pl -l hostname=chengdu"
decode_cmd="run.pl"
dir=exp/gmm_baseline


data=/aifs1/users/wd007/asr/baseline_chn_300h/data/test_plp
test_set="assist_test read_test"
lang=data/lang_trainlm


data=/aifs1/users/wd007/asr/baseline_chn_2000h/data/test/test_set_plp
test_set="accent_test"

#false && \
{
  # Decode (reuse HCLG graph)
  for test in $test_set; do
        for acwt in 0.06; do
        steps/decode_si_gmm.sh --nj 4 --cmd "$decode_cmd" --acwt $acwt --srcdir $dir \
        --model $dir/final.mdl $lang $data/$test $dir/decode/$test"_"$acwt || exit 1;
        done
  done
}

exit 0
