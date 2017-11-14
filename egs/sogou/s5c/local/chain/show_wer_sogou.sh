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
for l in $*; do
  grep WER ${l}/decode_testset_testND_bigG/wer_* | utils/best_wer.sh
done

for l in $*; do
  grep WER ${l}/decode_not_on_screen_sogou_offline/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_sogou_offline/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_sogou_offline/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testset_testND_sogou_offline/wer_* | utils/best_wer.sh
done

for l in $*; do
  grep WER ${l}/decode_not_on_screen_sogou_online/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_sogou_online/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_sogou_online/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testset_testND_sogou_online/wer_* | utils/best_wer.sh
done
