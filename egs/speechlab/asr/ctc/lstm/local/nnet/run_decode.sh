#!/bin/bash

. path.sh
. cmd.sh


decode_cmd="run.pl"
lang=data/lang_mix_kws
data=data/test
decode_cmd="run.pl"

dir=exp/tri_ctc_mmi2

test_set="ailab_tmjl_step10 ai_mcsntr_test400 asr_chn_tmjl_2017_dec12 afar_tc_test aishell_asr0010_E_test assistant_test oral_mobile_test accent_test ali_crowd_test asr_chn_yt_2018_jun10 keywords_test asr_chn_tmjl_2017_dec12_2 "

test_set="ailab_record_mandarin_clean_201807 aishell_asr0009_mandarin aishell_electric_car aishell_gasoline_car aishell_real_assistant datatang_children_read_300h datatang_mandarin_car_cellphone_245h datatang_mandarin_chat_cellphone_300h datatang_mandarin_english_mix_250h datatang_real_assistant_1175_600h datatang_real_assistant_3125_600h MDT2016S004_mandarin_cellphone MDT2016S005_chinese_english MDT2017S014_chinese_english_mix_read MDT2017S014_mandarin_chat_cellphone MDT20180112_mandarin_chat_cellphone MDT2018S001_mandarin_car MDT2018S010C1_children_read"

test_set="chn_tmjl_own_manual afar_tc_test chn_tmjl_own_nov25 ailab_newsig_tesla_manual_20181118 ailab_newsig_tulanduo_20181204 ailab_tmjl_step10 asr_chn_tmjl_2017_dec12 asr_chn_tmjl_2017_dec12_2 aishell_electric_car aishell_gasoline_car MDT2018S001_mandarin_car datatang_mandarin_english_mix_250h datatang_children_read_300h ai_mcsntr_test400 aishell_asr0010_E_test assistant_test oral_mobile_test accent_test ali_crowd_test asr_chn_yt_2018_jun10 keywords_test "

#--blank-posterior-scale 0.11 
acwt=0.6
#false && \
{
  # Decode (reuse HCLG graph)
  for test in $test_set; do
        for acwt in 0.6; do
        steps/nnet/decode_ctc.sh --nj 4 --cmd "$decode_cmd" --config conf/decode_ctc.config --max-active 2000 --acwt $acwt --srcdir $dir \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode_mix_kws/$test"_"$acwt || exit 1;
        done
  done
}

exit 0
