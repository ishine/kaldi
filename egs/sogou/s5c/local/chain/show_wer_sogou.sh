#!/bin/bash

for l in $*; do
  grep WER ${l}/decode_not_on_screen_bigG/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_bigG/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_bigG/wer_* | utils/best_wer.sh
done
