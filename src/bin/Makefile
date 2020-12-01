
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk
include ../speechlab.mk

ifeq ($(USE_KENLM), true)
    LDLIBS += -L../../tools/kenlm/build/lib -lkenlm_util -lkenlm -lkenlm_builder -lkenlm_filter -lkenlm_interpolate -lkenlm_util -lbz2 -lz
	CXXFLAGS += -DHAVE_KENLM -I/aifs/users/wd007/speaker/data/tools/kaldi-debug/src/kenlm
endif


BINFILES = align-equal align-equal-compiled acc-tree-stats \
        show-alignments compile-questions cluster-phones \
        compute-wer compute-wer-bootci make-h-transducer \
        add-self-loops convert-ali \
        compile-train-graphs compile-train-graphs-fsts \
        make-pdf-to-tid-transducer make-ilabel-transducer show-transitions \
        ali-to-phones ali-to-post weight-silence-post acc-lda est-lda \
        ali-to-pdf est-mllt build-tree build-tree-two-level decode-faster decode-faster-ctc \
        decode-faster-mapped vector-scale copy-transition-model \
        phones-to-prons prons-to-wordali copy-gselect copy-tree scale-post \
        post-to-weights sum-tree-stats weight-post post-to-tacc copy-matrix \
        copy-vector copy-int-vector sum-post sum-matrices draw-tree \
        align-mapped align-compiled-mapped latgen-faster-mapped latgen-faster-mapped-parallel \
        hmm-info analyze-counts post-to-phone-post \
        post-to-pdf-post logprob-to-post prob-to-post copy-post \
        matrix-sum build-pfile-from-ali get-post-on-ali tree-info am-info \
        vector-sum matrix-sum-rows est-pca sum-lda-accs sum-mllt-accs \
        transform-vec align-text matrix-dim copy-align ali-join-to-post latgen-faster-ctc \
		compile-train-graphs-ctc align-compiled-mapped-ctc copy-fst arpa2fst post-to-smat \
		latgen-faster-ctc-mapped-parallel analyze-pdf-acc triphone-to-pdf compile-questions-phone \
        compile-graph compare-int-vector decode-faster-rnnt decode-ctc-beam decode-ctc-beam-word


OBJFILES =

ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a ../lm/kaldi-lm.a \
          ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../nnet0/kaldi-nnet0.a ../cudamatrix/kaldi-cudamatrix.a ../matrix/kaldi-matrix.a


TESTFILES =

include ../makefiles/default_rules.mk
