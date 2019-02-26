
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

BINFILES = pid-test online-nnet-ipc-forward online-nnet-ipc-forward1 \
		   online-feature-extractor online-decoder-test \
		   online-ivector-extractor-test online-ivector-enroll-test \
		   online-xvector-extractor-test online-xvector-score-test

OBJFILES =

TESTFILES =

ADDLIBS = ../online0/kaldi-online0.a ../ivector/kaldi-ivector.a \
		  ../nnet0/kaldi-nnet0.a ../nnet2/kaldi-nnet2.a ../nnet3/kaldi-nnet3.a\
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 
include ../makefiles/default_rules.mk
