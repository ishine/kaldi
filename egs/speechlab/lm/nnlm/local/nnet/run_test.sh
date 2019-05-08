#!/bin/bash

. path.sh
. cmd.sh

data=/aifs/users/wd007/lm/baseline/web/nnlm/s5/data/raw_data
data=/aifs/users/wd007/lm/baseline/test_set/wseg
test_set="ai_mcsnor_evl13dec_fast_v1  ai_mcsnor_evl13jan_v1  ai_mcsntr_evl13mar_v1  ai_mcsntr_test400 ai_mcsnor_evl13dec_slow_v1  ai_mcsnor_evl13jun_v1  ai_mcsntr_evl14mar_v1  ai_mcsnur_evl13jan_v1  comm_carrbt_real_dec15v1 comm_appipt_real_sep16v1"


lang=data/lang
dir=exp/plstm_baseline
test_set="ai_mcsnor_evl13dec_fast_v1 ai_mcsntr_evl13mar_v1 ai_mcsntr_test400 comm_appipt_real_sep16v1"

#false && \
{
for test in $test_set; do
    outdir=$dir/test
    [ -d $outdir ] || mkdir -p $outdir
    awk 'NR==FNR{ss[$1]=$2;}NR>FNR{printf("utt_%d",NR);printf(" %s",ss["<s>"]); for(i=1;i<=NF;i++){if($i in ss)printf(" %s",ss[$i]);else printf(" %s",ss["<unk>"]);} printf(" %s\n",ss["</s>"]);}' $lang/vocab.txt $data/${test}.wseg | copy-align ark:- ark:$outdir/${test}.ark || exit 1;
    lm-lstm-sentence-ppl --use-gpu=yes --num-stream=40 --batch-size=12 --class-zt=$dir/zt.txt --class-boundary=$lang/class_boundary.txt $dir/final.nnet ark:$outdir/${test}.ark >$outdir/${test}.log 2>&1 || exit 1;

    echo $test done;
    done
}
