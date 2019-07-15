EXTRA_CXXFLAGS = -Wno-sign-compare

all:

include ../kaldi.mk
include ../speechlab.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

TESTFILES = arpa-file-parser-test arpa-lm-compiler-test

OBJFILES = arpa-file-parser.o arpa-lm-compiler.o const-arpa-lm.o \
			kaldi-rnnlm.o mikolov-rnnlm-lib.o kaldi-nnlm.o kaldi-lstmlm.o \
			kaldi-lm.o kaldi-lmtable.o lm-model-sync.o lm-compute-lstm-parallel.o rnnt-compute-lstm-parallel.o \
			seqlabel-compute-lstm-parallel.o slu-compute-lstm-parallel.o

LIBNAME = kaldi-lm

ADDLIBS = ../fstext/kaldi-fstext.a ../util/kaldi-util.a \
          ../base/kaldi-base.a ../nnet0/kaldi-nnet0.a ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a

ifeq ($(USE_WARP_TRANSDUCER), true)
    ADDLIBS += ../lib/warprnnt.so
endif

include ../makefiles/default_rules.mk
