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
  grep WER ${l}/decode_not_on_screen_offline/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_offline/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_offline/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testset_testND_offline/wer_* | utils/best_wer.sh
done

for l in $*; do
  grep WER ${l}/decode_not_on_screen_online/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_online/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_online/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testset_testND_online/wer_* | utils/best_wer.sh
done


for l in $*; do
  grep WER ${l}/decode_not_on_screen_looped/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_looped/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_looped/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testset_testND_looped/wer_* | utils/best_wer.sh
done
