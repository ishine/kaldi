

all:

include ../kaldi.mk
include ../speechlab.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
LDLIBS += $(MPICH_LDLIBS)

TESTFILES = nnet-randomizer-test nnet-component-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o \
           nnet-pdf-prior.o nnet-randomizer.o \
	   	   nnet-example.o nnet-model-sync.o \
           nnet-compute-sequential-parallel.o nnet-compute-parallel.o \
           nnet-model-merge-function.o \
	       nnet-compute-lstm-parallel.o nnet-compute-lstm-asgd.o \
	       nnet-compute-forward.o nnet-compute-ctc-parallel.o  \
		   nnet-compute-crfctc-parallel.o nnet-compute-lstm-lm-parallel.o 

ifeq ($(CUDA), true)
  OBJFILES += nnet-kernels.o
endif

LIBNAME = kaldi-nnet0

ADDLIBS = ../hmm/kaldi-hmm.a ../util/kaldi-util.a ../lat/kaldi-lat.a ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

ifeq ($(USE_WARP_CTC), true)
    ADDLIBS += ../warp-ctc/build/warpctc.a
endif

ifeq ($(USE_WARP_TRANSDUCER), true)
    ADDLIBS += ../lib/warprnnt.so
endif

# Implicit rule for kernel compilation,
%.o : %.cu
	$(CUDATKDIR)/bin/nvcc -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I../

include ../makefiles/default_rules.mk
