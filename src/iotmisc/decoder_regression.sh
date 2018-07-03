KALDI_ROOT=/Users/jerry/Documents/CodeForNoOne/kaldi
IOT_ROOT=${KALDI_ROOT}/src/iotbin
assets=assets

#${KALDI_ROOT}/tools/openfst/bin/fstprint ${assets}/HCLG.fst > ${assets}/HCLG.fst.txt
#${IOT_ROOT}/fst-copy ${assets}/HCLG.fst.txt ${assets}/HCLG.bin

${KALDI_ROOT}/src/online2bin/online2-wav-nnet3-latgen-faster \
    --online=true \
    --do-endpointing=false --endpoint.silence-phones=1 \
    --frames-per-chunk=20 --extra-left-context-initial=0 \
    --min-active=200 --max-active=7000 --beam=11.0 --lattice-beam=6.0 --acoustic-scale=0.1 \
    --feature-type=fbank --fbank-config=${assets}/fbank.conf \
    --word-symbol-table=${assets}/words.txt \
    --verbose=2 \
    ${assets}/final.mdl ${assets}/HCLG.fst \
    ark:/Users/jerry/Documents/CodeForNoOne/kaldi/egs/iot/s5/data/dev/spk2utt \
    scp:/Users/jerry/Documents/CodeForNoOne/kaldi/egs/iot/s5/data/dev/wav.scp \
    ark:/dev/null >& log.kaldi &

${IOT_ROOT}/iot \
    --online=true  \
    --do-endpointing=false --endpoint.silence-phones=1 \
    --frames-per-chunk=20 --extra-left-context-initial=0 \
    --min-active=200 --max-active=7000 --beam=11.0 --lattice-beam=6.0 --acoustic-scale=0.1 \
    --feature-type=fbank --fbank-config=${assets}/fbank.conf \
    --word-symbol-table=${assets}/words.txt \
    --verbose=2 \
    ${assets}/final.mdl ${assets}/HCLG.bin \
    ark:/Users/jerry/Documents/CodeForNoOne/kaldi/egs/iot/s5/data/dev/spk2utt \
    scp:/Users/jerry/Documents/CodeForNoOne/kaldi/egs/iot/s5/data/dev/wav.scp \
    ark:/dev/null >& log.iot &

wait

cat log.kaldi | grep -v 'LOG' | grep -v 'HCLG' > log.x
cat log.iot   | grep -v 'LOG' | grep -v 'HCLG' > log.y

vimdiff log.x log.y

