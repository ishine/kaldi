
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES =

OBJFILES = mace-computer.o mace-am-decodable-simple.o mace-decodable-simple-looped.o \
           mace-decodable-online-looped.o mace-online-nnet2-decodable.o \
           mace-online-nnet2-decoding.o mace-online-nnet3-decoding.o


LIBNAME = kaldi-mace

ADDLIBS = ../cudamatrix/kaldi-cudamatrix.a \
          ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a \
          ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../online2/kaldi-online2.a

include ../makefiles/default_rules.mk
