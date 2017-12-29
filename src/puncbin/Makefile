
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
SCWS_ROOT = /Users/kaituoxu/Tools/scws-1.2.3/scws
CXXFLAGS += -I$(SCWS_ROOT)/include

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = punc-forward punc-forward-oop

OBJFILES =



TESTFILES =

ADDLIBS = ../nnet/kaldi-nnet.a ../punc/kaldi-punc.a \
          ../cudamatrix/kaldi-cudamatrix.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
