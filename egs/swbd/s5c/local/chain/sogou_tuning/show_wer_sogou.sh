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
  grep WER ${l}/decode_not_on_screen_kaldi_d71_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_kaldi_d71_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_kaldi_d71_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testset_testND_kaldi_d71_0528/wer_* | utils/best_wer.sh
done

for l in $*; do
  grep WER ${l}/decode_not_on_screen_sogou_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_test8000_sogou_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testIOS_sogou_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testset_testND_sogou_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testSmallRoom1hours_sogou_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testMiddleRoom1hours_sogou_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testBigRoom1hours_sogou_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testmvdr_sogou_0528/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER ${l}/decode_testfarfiled_sogou_0528/wer_* | utils/best_wer.sh
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
