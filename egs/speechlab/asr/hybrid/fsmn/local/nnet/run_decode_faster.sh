#!/bin/bash

. path.sh
. cmd.sh


decode_cmd="queue.pl -l hostname=chengdu"
decode_cmd="run.pl"
dir=exp/tri_lstm_mpe2_nobad
data=/aifs1/users/wd007/asr/baseline_chn_2000h/nn/lstm/s1/data/test

test_set="aishell_test ai_mcsntr_test400 keywords_test accent_test assistant_test mandarin_test aishell_mandarin_test aishell_mobile_real_test oral_mobile_test room_3m_90 eworld_1m_90 aishell_xiaoyu_test aishell_asr0010_E_test aishell_asr0010_D_test chn30_clean_201710 meeting_normal_test washroom_speech-1m-90_1chan"
test_set="afar_x1000x_real_dec15v1"
test_set="ai_mcsntr_test400 room_3m_90 eworld_1m_90 aishell_xiaoyu_test aishell_asr0010_D_test accent_test"
test_set="digit_alphabet_fm_test"
lang=data/lang_keywords
test_set="ai_mcsntr_test400 room_3m_90 eworld_1m_90 aishell_xiaoyu_test aishell_asr0010_D_test accent_test"
test_set="ai_mcsntr_test400 room_3m_90 eworld_1m_90 aishell_xiaoyu_test"


lang=/aifs1/users/wd007/decode/grammar/digit/lang_kws
data=data
test_set="facepay_train"

data=/aifs1/users/wd007/asr/baseline_chn_2000h/nn/lstm/s1/data/test
lang=/aifs1/users/wd007/asr/facepay/nn/lstm/s7/data/lang
test_set="digit_alphabet_test"

#false && \
{
  # Decode (reuse HCLG graph)
  for test in $test_set; do
        for acwt in 0.08 0.1 0.065; do
        steps/nnet/decode_lstm_faster.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
        --beam 10.0 --min-active 20  --max-active 600 \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode_facepay/$test"_"$acwt || exit 1;
        done
  done
}
