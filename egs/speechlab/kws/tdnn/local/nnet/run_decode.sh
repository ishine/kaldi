#!/bin/bash

. path.sh
. cmd.sh


lang=data/lang_decode
data=/aifs1/users/wd007/asr/baseline_chn_2000h/nn/lstm/s1/data/test
decode_cmd="run.pl"

test_set="aishell_test ai_mcsntr_test400 keywords_test accent_test assistant_test mandarin_test aishell_mandarin_test aishell_mobile_real_test oral_mobile_test room_3m_90 eworld_1m_90 aishell_xiaoyu_test aishell_asr0010_E_test aishell_asr0010_D_test"
test_set="aishell_test ai_mcsntr_test400 keywords_test accent_test assistant_test mandarin_test aishell_mandarin_test aishell_mobile_real_test oral_mobile_test room_3m_90 eworld_1m_90 aishell_xiaoyu_test aishell_asr0010_E_test aishell_asr0010_D_test chn30_clean_201710 meeting_normal_test washroom_speech-1m-90_1chan afar_x1000x_real_dec15v1"
test_set="chn30_clean_201710 meeting_normal_test washroom_speech-1m-90_1chan afar_x1000x_real_dec15v1"
dir=exp/dnn_relu_baseline

#false && \
{
  # Decode (reuse HCLG graph)
  for test in $test_set ; do
        for acwt in 0.05 ;do
        steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode_comm/$test"_"$acwt || exit 1;
        done
  done
}

dir=exp/tri_dnn_mmi

false && \
{
  # Decode (reuse HCLG graph)
  for test in accent_test assistant_test mandarin_test; do
        for acwt in 0.1 ;do
        steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
        --nnet $dir/0.nnet_1_45000048 $lang $data/$test $dir/decode_1/$test"_"$acwt || exit 1;
        done
  done
}
