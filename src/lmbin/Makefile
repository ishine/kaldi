
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
include ../speechlab.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

BINFILES = arpa2fst arpa-to-const-arpa \
			lm-train-lstm-parallel lm-train-lstm-parallel-mpi \
			am-train-frmshuff-parallel am-train-lstm-parallel am-train-ctc-parallel rnnt-train-lstm-parallel \
			seqlabel-train-lstm-parallel slu-train-lstm-parallel \
			lm-lstm-sentence-ppl lm-multi-lstm-sentence-ppl

OBJFILES =

TESTFILES =

ADDLIBS = ../lm/kaldi-lm.a ../nnet0/kaldi-nnet0.a ../cudamatrix/kaldi-cudamatrix.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

ifeq ($(USE_WARP_CTC), true)
    ADDLIBS += ../warp-ctc/build/warpctc.a
endif

ifeq ($(USE_WARP_TRANSDUCER), true)
    ADDLIBS += ../lib/warprnnt.so
endif

include ../makefiles/default_rules.mk
