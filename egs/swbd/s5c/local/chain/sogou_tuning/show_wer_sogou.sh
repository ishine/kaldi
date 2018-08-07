#!/bin/bash

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
  grep WER ${l}/decode_not_on_screen_sogou_online_chunk150/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_sogou_online_chunk150/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_sogou_online_chunk150/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testset_testND_sogou_online_chunk150/wer_* | utils/best_wer.sh
done

for l in $*; do
  grep WER ${l}/decode_not_on_screen_sogou_offline_bin/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_sogou_offline_bin/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_sogou_offline_bin/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testSogouTranslatorOutEnhance_sogou_offline_bin/wer_* | utils/best_wer.sh
done
