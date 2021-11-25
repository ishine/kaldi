# This directory is not compiled automatically when you do make under src/
# to compile the TensorFlow related code,

# you would need to run tools/extras/install_tensorflow_cc.sh first to generate
# the static library files required for compilation, and then

# cd ${KALDI_ROOT}/src
# cd tfnet
# make
# cd ../tfnetbin/
# make

include ../kaldi.mk

TENSORFLOW = /asrfs/users/wd007/asr/tools/src/opensource/tensorflow/tensorflow-gpu-1.14.0/build

all:

EXTRA_CXXFLAGS = -Wno-sign-compare \
				 -I/asrfs/users/wd007/tools/protobuf-3.7.0/src \
                 -I$(TENSORFLOW)/bazel-genfiles -I$(TENSORFLOW) \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/absl \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/eigen \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/nsync/public \
                 -I$(TENSORFLOW)/tensorflow/contrib/makefile/downloads/protobuf/src

OBJFILES = tensorflow-transformer.o

TESTFILES =

LIBNAME = kaldi-tfnet

ADDLIBS = ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a 
LDLIBS +=  -lz -ldl -fPIC -lrt
LDLIBS += -L$(TENSORFLOW)/lib -ltensorflow_cc -ltensorflow_framework

include ../makefiles/default_rules.mk
