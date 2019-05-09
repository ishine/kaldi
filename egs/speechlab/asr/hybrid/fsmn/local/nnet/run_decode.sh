#!/bin/bash

. path.sh
. cmd.sh


lang=data/lang_decode
data=/aifs1/users/wd007/asr/baseline_chn_300h/nn/lstm/s1/data
decode_cmd="run.pl"

data=data/test
dir=exp/fsmn_baseline
lang=data/lang_mix_kws

test_set="ailab_record_mandarin_clean_201807 aishell_asr0009_mandarin aishell_electric_car aishell_gasoline_car aishell_real_assistant datatang_children_read_300h datatang_mandarin_car_cellphone_245h datatang_mandarin_chat_cellphone_300h datatang_mandarin_english_mix_250h datatang_real_assistant_1175_600h datatang_real_assistant_3125_600h MDT2016S004_mandarin_cellphone MDT2016S005_chinese_english MDT2017S014_chinese_english_mix_read MDT2017S014_mandarin_chat_cellphone MDT20180112_mandarin_chat_cellphone MDT2018S001_mandarin_car MDT2018S010C1_children_read"

test_set="ailab_newsig_tesla_manual_20181118 ailab_tmjl_step10 asr_chn_tmjl_2017_dec12 asr_chn_tmjl_2017_dec12_2 aishell_electric_car aishell_gasoline_car MDT2018S001_mandarin_car datatang_mandarin_english_mix_250h datatang_children_read_300h ai_mcsntr_test400 afar_tc_test aishell_asr0010_E_test assistant_test oral_mobile_test accent_test ali_crowd_test asr_chn_yt_2018_jun10 keywords_test "

#false && \
{
  # Decode (reuse HCLG graph)
  for test in $test_set; do
        for acwt in 0.05;do
        steps/nnet/decode_fsmn.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode_mix_kws/$test"_"$acwt || exit 1;
        done
  done
}

dir=exp/tri_dnn_mmi

false && \
{
  # Decode (reuse HCLG graph)
  for test in assist_test read_test; do
        for acwt in 0.1 ;do
        steps/nnet/decode.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode/$test"_"$acwt || exit 1;
        done
  done
}
