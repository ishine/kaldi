
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

LDLIBS += $(MACE_LDLIBS)
LDFLAGS += $(MACE_LDFLAGS)

BINFILES = mace-latgen-faster mace-compute \
           mace-latgen-faster-looped mace-online2-wav-nnet2-am-compute \
           mace-online2-wav-nnet3-latgen-faster mace-online2-wav-nnet2-latgen-faster

OBJFILES =

# Add this dependency to force cuda-compiled.o to be rebuilt when we reconfigure.
cuda-compiled.o: ../kaldi.mk

TESTFILES =

ADDLIBS = ../base/kaldi-base.a ../matrix/kaldi-matrix.a ../cudamatrix/kaldi-cudamatrix.a \
          ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a \
          ../hmm/kaldi-hmm.a ../util/kaldi-util.a  ../feat/kaldi-feat.a \
          ../ivector/kaldi-ivector.a ../online2/kaldi-online2.a ../mace/kaldi-mace.a

include ../makefiles/default_rules.mk
