#export KALDI_ROOT=`pwd`/../../..
export KALDI_ROOT=/aifs/users/wd007/src/KALDI/kaldi-develop
#PWD=/aifs/users/wd007/asr/baseline_chn_5000h/s7
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/nnet0bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$PWD:$PATH
export LC_ALL=C
export MKL_NUM_THREADS=1

export LD_LIBRARY_PATH=/aifs1/tools/MPI/mpich-3.2/mpich-install/lib:$LD_LIBRARY_PATH

export PATH=/aifs1/tools/MPI/mpich-3.2/mpich-install/bin:$PATH
