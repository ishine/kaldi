#!/bin/bash

. path.sh
. cmd.sh


decode_cmd="queue.pl -l hostname=chengdu"
decode_cmd="run.pl"
data=data/test
lang=data/lang_mix_kws

test_set="ailab_record_mandarin_clean_201807 aishell_asr0009_mandarin aishell_electric_car aishell_gasoline_car aishell_real_assistant datatang_children_read_300h datatang_mandarin_car_cellphone_245h datatang_mandarin_chat_cellphone_300h datatang_mandarin_english_mix_250h datatang_real_assistant_1175_600h datatang_real_assistant_3125_600h MDT2016S004_mandarin_cellphone MDT2016S005_chinese_english MDT2017S014_chinese_english_mix_read MDT2017S014_mandarin_chat_cellphone MDT20180112_mandarin_chat_cellphone MDT2018S001_mandarin_car MDT2018S010C1_children_read"

test_set="asr_chn_tmjl_2017_dec12 ai_mcsntr_test400 afar_tc_test aishell_asr0010_E_test assistant_test oral_mobile_test accent_test ali_crowd_test asr_chn_yt_2018_jun10 keywords_test asr_chn_tmjl_2017_dec12_2 "

dir=exp/plstm_baseline
#false && \
{
  # Decode (reuse HCLG graph)
  for test in $test_set ; do
        for acwt in 0.05; do
        steps/nnet/decode_lstm.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode_mix_kws/$test"_"$acwt || exit 1;
        done
  done
}

false && \
{
  # Decode (reuse HCLG graph)
  for test in $test_set; do
        for acwt in 0.065; do
        steps/nnet/decode_lstm.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode_comm/$test"_"$acwt || exit 1;
        done
  done
}

#--beam 10.0 --lattice_beam 6.0 --min-active 20  --max-active 600 --scoring-opts "--min-lmwt 5 --max-lmwt 25" \
