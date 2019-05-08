#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage $0 <result.txt> <labele.txt> <outfile.txt>"
    exit -1;
fi

. path.sh

result=$1
label=$2
outfile=$3

cat $result | align-text --special-symbol="'***'" ark:$label ark:- ark,t:- |\
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" | tee $outfile || exit -1;

cat $result | compute-wer --text --mode=present ark:$label ark,p:-
