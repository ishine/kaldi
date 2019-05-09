#export KALDI_ROOT=`pwd`/../../..
#export KALDI_ROOT=/aifs/users/wd007/src/KALDI/kaldi-develop
#export KALDI_ROOT=/aifs/users/wd007/decoder/tools/kaldi-release
#export KALDI_ROOT=/aifs/users/wd007/decoder/tools/kaldi-test
export KALDI_ROOT=/aifs/users/wd007/speaker/data/tools/kaldi-debug
#PWD=/aifs/users/wd007/asr/baseline_chn_5000h/s7
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/nnet0bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$PWD:$PATH
export LC_ALL=C
export MKL_NUM_THREADS=1

export LD_LIBRARY_PATH=/aifs/tools/MPI/mpich-install/lib:$LD_LIBRARY_PATH

export PATH=/aifs/tools/MPI/mpich-install/bin:$PATH
