
all:

include ../kaldi.mk

CXXFLAGS += -Wno-invalid-offsetof

TESTFILES = memory-pool-test link-test list-test phone-set-test vocab-test lexicon-test word-loop-test

OBJFILES = common.o memory-pool.o link.o list.o graph.o \
           phone-set.o vocab.o lexicon.o \
           word-loop.o wfst.o \
           dec-core.o online-decoder.o

LIBNAME = kaldi-iot

#ADDLIBS = ../base/kaldi-base.a 

ADDLIBS = ../ivector/kaldi-ivector.a ../nnet3/kaldi-nnet3.a \
          ../chain/kaldi-chain.a ../nnet2/kaldi-nnet2.a \
          ../cudamatrix/kaldi-cudamatrix.a \
          ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

