all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
include ../speechlab.mk

ifeq ($(USE_KENLM), true)
    LDLIBS += -L../../tools/kenlm/build/lib -lkenlm_util -lkenlm -lkenlm_builder -lkenlm_filter -lkenlm_interpolate -lkenlm_util -lbz2 -lz
	CXXFLAGS += -DHAVE_KENLM -I/aifs/users/wd007/speaker/data/tools/kaldi-debug/src/kenlm
endif


TESTFILES =

OBJFILES = training-graph-compiler.o lattice-simple-decoder.o lattice-faster-decoder.o \
   lattice-faster-online-decoder.o simple-decoder.o faster-decoder.o \
   decoder-wrappers.o grammar-fst.o decodable-matrix.o decoder-ctc-wrappers.o \
   rnnt-decoder.o ctc-decoder.o ctc-decoder-word.o

LIBNAME = kaldi-decoder

ADDLIBS = ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../lm/kaldi-lm.a ../nnet0/kaldi-nnet0.a

include ../makefiles/default_rules.mk
