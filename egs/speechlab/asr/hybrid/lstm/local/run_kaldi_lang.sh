#!/bin/bash


. cmd.sh

dir=lang_seq

model=/aifs/users/wd007/asr/baseline_chn_7000h/gmm/s6/exp/tri3b
lm=/aifs/users/wd007/asr/baseline_500h/nn/ctc/phone/s6/data/lms/final.ch.en-3-gram.gz
lm=/aifs/users/wd007/asr/baseline_chn_7000h/nn/hybrid/lstm/s6/data/lms/final.bg.lm.gz
dict=/aifs/users/wd007/asr/baseline_chn_7000h/data/resource/lexicon.txt

    echo =====================================================================
    echo "             FST Construction                 "
    echo =====================================================================

#false && \
{
    # G compilation, check LG composition
    utils/format_lm.sh lang $lm $dict $dir || exit 1;
}

    # HCLG
    utils/mkgraph.sh $dir $model $dir || exit 1;
