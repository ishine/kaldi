all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES =

OBJFILES = online-util.o online-nnet-faster-decoder.o online-feature.o \
		   online-nnet-feature-pipeline.o online-fst-decoder-cfg.o online-fst-decoder.o \
		   online-ivector-feature.o online-ivector-extractor.o online-xvector-extractor.o \
		   decoder_export.o ivector_export.o xvector_export.o

LIBNAME = kaldi-online0

ADDLIBS = ../ivector/kaldi-ivector.a ../nnet3/kaldi-nnet3.a ../nnet0/kaldi-nnet0.a \
		  ../cudamatrix/kaldi-cudamatrix.a ../fstext/kaldi-fstext.a \
		  ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../feat/kaldi-feat.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
