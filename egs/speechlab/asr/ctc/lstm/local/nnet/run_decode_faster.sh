#!/bin/bash

. path.sh
. cmd.sh


decode_cmd="queue.pl -l hostname=chengdu"
decode_cmd="run.pl"
dir=exp/plstm_ce_ctc_aug

data=/aifs1/users/wd007/asr/baseline_chn_2000h/data/test/test_set
test_set="accent_test assistant_test mandarin_test "

#data=/aifs1/users/wd007/asr/baseline_chn_300h/nn/lstm/s1/data/test
#test_set="eworld_1m_0   eworld_1m_135 eworld_1m_180 eworld_1m_45  eworld_1m_90 room_3m_30  room_3m_60  room_3m_90"

data=/aifs1/users/wd007/asr/baseline_chn_2000h/nn/lstm/s1/data/test
test_set="aishell_test ai_mcsntr_test400 keywords_test accent_test assistant_test mandarin_test aishell_mandarin_test aishell_mobile_real_test oral_mobile_test room_3m_90 eworld_1m_90 aishell_xiaoyu_test aishell_asr0010_E_test aishell_asr0010_D_test chn30_clean_201710 meeting_normal_test"

lang=data/lang_decode
dir=exp/plstm_digit_kld

test_set="iflytek_split1_10  iflytek_split1_5  iflytek_split2_10  iflytek_split2_5  iflytek_fast_split1_0-5  iflytek_fast_split1_5-10  iflytek_fast_split2_0-5  iflytek_fast_split2_5-10"
test_set="digit_fast_new_test digit_test digit_fast_test digit_snr10_test digit_snr15_test digit_snr20_test xiaoyu_fast_split2_5-10 iflytek_fast_split1_5-10"
test_set="digit_fast_new_test digit_test digit_fast_test digit_snr15_test xiaoyu_fast_split2_5-10 iflytek_fast_split1_5-10"
test_set="digit_snr10_test digit_snr15_test digit_snr20_test xiaoyu_fast_split2_5-10 iflytek_fast_split1_5-10"
test_set="digit_fast_snr10_iflytek1_test digit_fast_snr10_xiaoyu_test "
lang=data/lang_digit
#--blank-posterior-scale 0.11 
acwt=0.6
#false && \
{
  # Decode (reuse HCLG graph)
  for test in $test_set; do
        for acwt in 1.5; do
            for post in 0.11; do
        steps/nnet/decode_ctc_faster.sh --nj 8 --cmd "$decode_cmd" --config conf/decode_ctc.config --acwt $acwt --srcdir $dir \
        --blank-posterior-scale $post --beam 10.0 --min-active 20  --max-active 600 \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode_prior/${test}_${acwt}_${post} || exit 1;
            done
        done
  done
}

exit 0
