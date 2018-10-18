. ./path.sh

KALDI_ROOT=/home/dophist/work/CodeForNoOne/kaldi
IOT_ROOT=${KALDI_ROOT}/src/iotbin

SRILM_ROOT=/home/dophist/work/git/srilm-1.7.2/bin/i686-m64
#KENLM_ROOT=/home/dophist/work/git/kenlm

dir=iot
mkdir -p $dir

stage=10

acwt=1.0
post_acwt=10.0
beam=15.0
lat_beam=8.0
min_active=200
max_active=7000
num_threads=10

if [ $stage -le 1 ]; then
  # copy resource into working dir
  cp exp/chain/tdnn_1a/graph/HCLG.fst ${dir}/
  cp exp/chain/tdnn_1a/final.mdl ${dir}/
  cp exp/chain/tree/tree ${dir}/
  cp data/lang_test/words.txt ${dir}/
  cp -r data/lang_test $dir/ && rm -rf $dir/lang_test/tmp && rm $dir/lang_test/G.fst

  cp data/test_mfcc_hires/{wav.scp,spk2utt,feats.scp} $dir/

  awk '{$1=""; print $0}' data/train/text > $dir/trn_text
  awk '{$1=""; print $0}' data/test/text  > $dir/tst_text
fi

if [ $stage -le 2 ]; then
  # offline decoding
  nnet3-latgen-faster-parallel \
    --verbose=1 \
    --num-threads=$num_threads \
    --frame-subsampling-factor=3 \
    --frames-per-chunk=50 \
    --extra-left-context=0 --extra-right-context=0 \
    --extra-left-context-initial=-1 --extra-right-context-final=-1 \
    --minimize=false \
    --min-active=$min_active --max-active=$max_active \
    --beam=$beam --lattice-beam=$lat_beam \
    --acoustic-scale=$acwt \
    --word-symbol-table=${dir}/words.txt \
    ${dir}/final.mdl \
    ${dir}/HCLG.fst \
    scp:${dir}/feats.scp \
    "ark:|lattice-scale --acoustic-scale=$post_acwt ark:- ark:${dir}/lat.ark" \
    &> $dir/log.offline
fi


if [ $stage -le 6 ]; then
  # Build ARPA
  cat $dir/words.txt | awk '{print $1}' | grep -v '#0' | grep -v '<eps>' | grep -v '<UNK>' > $dir/ngram.wlist

  # build 2gram for LA
  ${SRILM_ROOT}/ngram-count -order 2 -limit-vocab -vocab $dir/ngram.wlist -kndiscount -interpolate -text $dir/trn_text -lm $dir/g.arpa

  arpa2fst --disambig-symbol=#0 --read-symbol-table=${dir}/words.txt ${dir}/g.arpa - \
      | fstarcsort --sort_type=ilabel \
      > $dir/g.fst

  # build 3gram LM
  ${SRILM_ROOT}/ngram-count -order 3 -limit-vocab -vocab $dir/ngram.wlist -kndiscount -interpolate -text $dir/trn_text -lm $dir/G.arpa

  arpa2fst --disambig-symbol=#0 --read-symbol-table=${dir}/words.txt ${dir}/G.arpa - \
      | fstarcsort --sort_type=ilabel \
      > ${dir}/G.fst

  arpa2fst --disambig-symbol=#0 --read-symbol-table=${dir}/words.txt ${dir}/G.arpa - \
      | fstproject --project_output=true - \
      | fstarcsort --sort_type=ilabel \
      > ${dir}/G_proj.fst

#  # build in-domain LM for online interpolation
#  ${SRILM_ROOT}/ngram-count -order 3 -vocab $dir/ngram.wlist -text $dir/tst_text -lm $dir/in_domain.arpa
#  
#  arpa2fst --disambig-symbol=#0 --read-symbol-table=${dir}/words.txt ${dir}/in_domain.arpa - \
#      | fstproject --project_output=true - \
#      | fstarcsort --sort_type=ilabel \
#      > ${dir}/in_domain.fst
fi

if [ $stage -le 7 ]; then
  rm -rf $dir/lang_test/tmp
  rm $dir/lang_test/G.fst
  cp $dir/g.fst $dir/lang_test/G.fst
  utils/mkgraph.sh $dir/lang_test $dir $dir/graph_HCLg

  rm -rf $dir/lang_test/tmp
  rm $dir/lang_test/G.fst
  cp $dir/G.fst $dir/lang_test/G.fst
  utils/mkgraph.sh $dir/lang_test $dir $dir/graph_HCLG
fi

if [ $stage -le 8 ]; then
  cd $dir/graph_HCLg
  fstprint HCLG.fst > HCLG.fst.txt
  ${IOT_ROOT}/fst-copy HCLG.fst.txt HCLG.bin
  cd -

  cd $dir/graph_HCLG
  fstprint HCLG.fst > HCLG.fst.txt
  ${IOT_ROOT}/fst-copy HCLG.fst.txt HCLG.bin
  cd -
fi

if [ $stage -le 9 ]; then
  # kaldi online wav decoding
  online2-wav-nnet3-latgen-faster \
    --online=true \
    --do-endpointing=false --endpoint.silence-phones=1 \
    --feature-type=mfcc --mfcc-config=conf/mfcc_hires.conf \
    --frame-subsampling-factor=3 \
    --extra-left-context-initial=0 \
    --frames-per-chunk=50 \
    --min-active=$min_active --max-active=$max_active \
    --beam=$beam --lattice-beam=$lat_beam \
    --acoustic-scale=$acwt \
    --word-symbol-table=${dir}/words.txt \
    --verbose=1 \
    ${dir}/final.mdl \
    ${dir}/graph_HCLG/HCLG.fst \
    ark:${dir}/spk2utt \
    scp:${dir}/wav.scp.200 \
    ark:/dev/null \
    >& $dir/log.kaldi_online2 &
fi

if [ $stage -le 10 ]; then
  ${IOT_ROOT}/iot \
      --online=true  \
      --do-endpointing=false --silence-phones=1 \
      --feature-type=mfcc --mfcc-config=conf/mfcc_hires.conf \
      --frame-subsampling-factor=3 \
      --extra-left-context-initial=0 \
      --frames-per-chunk=50 \
      --min-active=$min_active --max-active=$max_active \
      --beam=$beam --lattice-beam=$lat_beam \
      --acoustic-scale=$acwt \
      --word-symbol-table=${dir}/words.txt \
      --verbose=1 \
      --interp-lm=$dir/G_proj.fst \
      --interp-lm-scale=0.0 \
      $dir/final.mdl \
      $dir/graph_HCLg/HCLG.bin \
      ark:$dir/spk2utt \
      scp:$dir/wav.scp.200 \
      ark:/dev/null \
      >& $dir/log.iot &
fi

if [ $stage -le 11 ]; then
  # build resources for iot experiment
  ${IOT_ROOT}/iot \
      --online=true  \
      --do-endpointing=false --silence-phones=1 \
      --feature-type=mfcc --mfcc-config=conf/mfcc_hires.conf \
      --frame-subsampling-factor=3 \
      --extra-left-context-initial=0 \
      --frames-per-chunk=50 \
      --min-active=$min_active --max-active=$max_active \
      --beam=$beam --lattice-beam=$lat_beam \
      --acoustic-scale=$acwt \
      --word-symbol-table=${dir}/words.txt \
      --verbose=1 \
      --interp-lm=$dir/G_proj.fst \
      --interp-lm-scale=0.1 \
      $dir/final.mdl \
      $dir/graph_HCLg/HCLG.bin \
      ark:$dir/spk2utt \
      scp:$dir/wav.scp.200 \
      ark:/dev/null \
      >& $dir/log.iot_dynamic &
fi

wait

if [ $stage -le 12 ]; then
  sort -k1,1 data/test/text > $dir/ref

  cd $dir
  cat log.kaldi_online2 | grep -v 'final.mdl' | grep -v 'LOG' | grep -v 'WARNING' | sort -k1,1 > rec.kaldi
  cat log.iot           | grep -v 'final.mdl' | grep -v 'LOG' | grep -v 'WARNING' | sort -k1,1 > rec.iot
  cat log.iot_dynamic   | grep -v 'final.mdl' | grep -v 'LOG' | grep -v 'WARNING' | sort -k1,1 > rec.iot_dynamic

  compute-wer --text=true --mode=present ark,t:ref ark,t:rec.kaldi
  compute-wer --text=true --mode=present ark,t:ref ark,t:rec.iot
  compute-wer --text=true --mode=present ark,t:ref ark,t:rec.iot_dynamic
  
  #vimdiff rec.iot rec.kaldi
  cd -
fi

#sort -k1,1 /Users/jerry/Documents/CodeForNoOne/kaldi/egs/aishell2/s5/data/dev/text > ref
#
#cat log.kaldi | grep -v 'LOG' | grep -v 'HCLG' | grep -v 'WARNING' | sort -k1,1 > rec.kaldi
#cat log.iot   | grep -v 'LOG' | grep -v 'HCLG' | grep -v 'WARNING' | sort -k1,1 > rec.iot
#
#compute-wer --text=true --mode=present ark,t:ref ark,t:rec.kaldi
#compute-wer --text=true --mode=present ark,t:ref ark,t:rec.iot

###vimdiff rec.iot rec.kaldi
