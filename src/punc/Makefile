

all:

include ../kaldi.mk

SCWS_ROOT = /Users/kaituoxu/Tools/scws-1.2.3/scws
CXXFLAGS += -I$(SCWS_ROOT)/include
LDFLAGS += $(CUDA_LDFLAGS) -L$(SCWS_ROOT)/lib
LDLIBS += $(CUDA_LDLIBS) -lscws

TESTFILES = 

BINFILES = test-scws

OBJFILES = punctuator.o

LIBNAME = kaldi-punc

ADDLIBS = ../nnet/kaldi-nnet.a ../cudamatrix/kaldi-cudamatrix.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a \
		  ../util/kaldi-util.a $(SCWS_ROOT)/lib/scws.la

include ../makefiles/default_rules.mk

