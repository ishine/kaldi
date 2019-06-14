
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
include ../speechlab.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

BINFILES = arpa2fst arpa-to-const-arpa \
			lm-train-lstm-parallel lm-train-lstm-parallel-mpi \
			rnnt-train-lstm-parallel \
			seqlabel-train-lstm-parallel slu-train-lstm-parallel \
			lm-lstm-sentence-ppl lm-multi-lstm-sentence-ppl

OBJFILES =

TESTFILES =

ADDLIBS = ../lm/kaldi-lm.a ../nnet0/kaldi-nnet0.a ../nnet2/kaldi-nnet2.a ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

ifeq ($(USE_WARP_TRANSDUCER), true)
    ADDLIBS += ../lib/warprnnt.so
endif

include ../makefiles/default_rules.mk
