lattice-to-nbest --acoustic-scale=0.1 --n=20 "ark:gunzip CarNoisyDatong_tdnn.lat.gz |" ark:lat.20best.ark
lattice-best-path --acoustic-scale=0.1 --word-symbol-table=/public/speech/wangzhichao/kaldi/kaldi-wzc/egs/sogou/s5c/data/lang_0528/words.txt ark:lat.20best.ark ark,t:lat.20best.txt >CarNoisyDatong_tdnn.lat.20best.txt
