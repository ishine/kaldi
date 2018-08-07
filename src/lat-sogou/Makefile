all: libkaldilat.a

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

LIBNAME = latdet

ADDLIBS = ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../util/kaldi-util.a \

libkaldilat.a: lattice-determinizer.o
	-rm $@
	ar rcsT $@ $^ $(ADDLIBS) $(filter %.a,$(LDLIBS))
	printf "create $@\naddlib $@\nsave\nend"|ar -M

# LINK.o = $(CXX) $(LDFLAGS) $(TARGET_ARCH)
# clean:
# 	-rm *.o
