all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

TESTFILES =

OBJFILES = online-util.o online-nnet-faster-decoder.o online-feature.o \
		   online-nnet-feature-pipeline.o online-fst-decoder.o

LIBNAME = kaldi-online0

ADDLIBS = ../nnet0/kaldi-nnet0.a ../cudamatrix/kaldi-cudamatrix.a ../fstext/kaldi-fstext.a \
		  ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../feat/kaldi-feat.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../thread/kaldi-thread.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
