#!/bin/bash


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)



# fbank-dnn alignments first:
alidir=exp/tri
dir=exp/tri_dnn
acwt=0.1

false && \
{
#count
awk 'NR==FNR{if(NF>=3){w[$3]+=($2-$1)/100000;}}NR!=FNR{t[FNR]=$1;}END{printf "[";for(i=1;i<=FNR;i++){printf " %d",w[t[i]];}printf " ]\n";}' data/train/ref.mlf data/lang/statesyms.txt > $dir/ali_train_pdf.counts
#copy the old transition model, will be needed by decoder
copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl 2>$dir/final.mdl_log || exit 1
cp $alidir/tree $dir/tree || exit 1
}


### Finally we train using MMI criterion.
### We do Stochastic-GD with per-utterance updates. 
###
### To get faster convergence, we will re-generate 
### the lattices after 1st epoch of MMI.
###



srcdir=exp/tri_ctc
traindir=data/train/trainctc_tr90
acwt=0.6
boost=0.1
decode_cmd=run.pl
cuda_cmd=run.pl
train_cmd=run.pl

# First we need to generate lattices and alignments:

#false && \
{
steps/align_ctc_htk.sh --nj 10 --cmd "$train_cmd" \
   $traindir/split3/3 data/lang $srcdir ${srcdir}_ali_train/3 || exit 1;
}

#false && \
{
steps/make_denlats_ctc_htk.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_ctc.config --acwt $acwt \
   $traindir/split3/3 data/lang $srcdir ${srcdir}_denlats_train/3 || exit 1;
}

traindir=data/train_select/cm80
cuda_cmd=run.pl
dir=${srcdir}_mmi
# Now we re-train the hybrid by single iteration of MMI
false && \
{
# asgd
steps/nnet/train_mmi.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt \
  --learn-rate 0.000002 \
  $traindir data/lang  ${srcdir} \
  ${srcdir}_ali_train \
  ${srcdir}_denlats_train \
  $dir || exit 1;
}

dir=${srcdir}_smbr
# Now we re-train the hybrid by single iteration of MMI
false && \
{
# asgd
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt\
  --learn-rate 0.000001 \
  $traindir data/lang $srcdir \
  ${srcdir}_ali_train \
  ${srcdir}_denlats_train \
  $dir || exit 1;
}

dir=${srcdir}_mmi
decode_cmd=run.pl

# re-lattice
dir=exp/tri_ctc_mmi2
srcdir=exp/tri_ctc_mmi
traindir=data/train_select/cm80
acwt=0.6

# First we need to generate lattices and alignments:

false && \
{
steps/align_ctc_htk.sh --nj 5 --cmd "$train_cmd" \
   $traindir/split3/3 data/lang $srcdir ${srcdir}_ali_train/3 || exit 1;
}

false && \
{
steps/make_denlats_ctc_htk.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_ctc.config --acwt $acwt \
   $traindir/split3/3 data/lang $srcdir ${srcdir}_denlats_train/3 || exit 1;
}

# Now we re-train the hybrid by single iteration of MMI
false && \
{
steps/nnet/train_mmi.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt\
  --learn-rate 0.000001 \
  $traindir data/lang $srcdir \
  ${srcdir}_ali_train \
  ${srcdir}_denlats_train \
  $dir || exit 1;
}

false && \
{
# asgd
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt\
  --learn-rate 0.000001 \
  $traindir data/lang $srcdir \
  ${srcdir}_ali_train \
  ${srcdir}_denlats_train \
  $dir || exit 1;
}

