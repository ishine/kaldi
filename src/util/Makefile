

all:


include ../kaldi.mk

TESTFILES = const-integer-set-test stl-utils-test text-utils-test \
    edit-distance-test hash-list-test kaldi-io-test parse-options-test \
    kaldi-table-test simple-options-test kaldi-thread-test circular-queue-test

OBJFILES = text-utils.o kaldi-io.o kaldi-holder.o kaldi-table.o \
           parse-options.o simple-options.o simple-io-funcs.o \
           kaldi-semaphore.o kaldi-mutex.o kaldi-barrier.o kaldi-thread.o trie-tree.o

LIBNAME = kaldi-util

ADDLIBS = ../matrix/kaldi-matrix.a ../base/kaldi-base.a 

include ../makefiles/default_rules.mk
