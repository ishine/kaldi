KALDI_ROOT=/Users/jerry/Documents/CodeForNoOne/kaldi
IOT_ROOT=${KALDI_ROOT}/src/iotbin
assets=assets

${IOT_ROOT}/realtime-demo \
    --do-endpointing=true --endpoint.silence-phones=1 \
    --frames-per-chunk=20 --extra-left-context-initial=0 --online=true  \
    --min-active=200 --max-active=7000 --beam=11.0 --lattice-beam=6.0 --acoustic-scale=0.1 \
    --feature-type=fbank \
    --fbank-config=${assets}/fbank.conf \
    --word-symbol-table=${assets}/words.txt \
    --verbose=2 \
    ${assets}/final.mdl ${assets}/HCLG.fst ark:/dev/null
