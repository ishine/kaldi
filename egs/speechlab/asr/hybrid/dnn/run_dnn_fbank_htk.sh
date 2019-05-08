#/bin/bash


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



srcdir=exp/tri_dnn
traindir=data/train/train_cv10
acwt=0.1
boost=0.1

decode_cmd=run.pl
cuda_cmd=run.pl
train_cmd=run.pl

# First we need to generate lattices and alignments:

#false && \
{
steps/align_nnet_htk.sh --nj 10 --cmd "$train_cmd" \
   $traindir data/lang $srcdir ${srcdir}_ali_cv || exit 1;
}

false && \
{
steps/make_denlats_nnet_htk.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
   $traindir data/lang $srcdir ${srcdir}_denlats_train  || exit 1;
}

dir=exp/tri_dnn_mmi
srcdir=exp/tri_dnn
traindir=data/train_select/cm90
# Now we re-train the hybrid by single iteration of MMI
false && \
{
# asgd
steps/nnet/train_mmi.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt\
  $traindir data/lang $srcdir \
  ${srcdir}_ali_train \
  ${srcdir}_denlats_train \
  $dir || exit 1;
}

false && \
{
# multi-machine
steps/nnet/train_mmi_mpi.sh --cmd "$cuda_cmd" --num-iters 1 --use-psgd true --acwt $acwt\
  $traindir data/lang $srcdir \
  ${srcdir}_ali_train \
  ${srcdir}_denlats_train \
  $dir || exit 1;
}

dir=exp/tri_dnn_mpe
# Now we re-train the hybrid by single iteration of MMI
false && \
{
# single
steps/nnet/train_mpe.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt\
  $traindir data/lang $srcdir \
  ${srcdir}_ali_train \
  ${srcdir}_denlats_train \
  $dir || exit 1;
}

decode_cmd=run.pl

data=/aifs/users/wd007/asr/test_set/aicloud
lang=data/lang_decode
#ai_mcsnor_evl13jun_v1  ai_mcsntr_evl13mar_v1  ai_mcsntr_evl14jan_v1  ai_mcsntr_evl14mar_v1

false && \
{
  # Decode (reuse HCLG graph)
  for test in ai_mcsnor_evl13jun_v1 ai_mcsntr_evl13mar_v1 ai_mcsntr_evl14jan_v1  ai_mcsntr_evl14mar_v1; do
        for acwt in 0.05;do
        steps/nnet/decode.sh --nj 2 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt --srcdir $dir \
        --nnet $dir/final.nnet $lang $data/$test $dir/decode/$test"_"$acwt || exit 1;
        done
  done
}


# re-lattice
dir=exp/tri_dnn_mmi2
srcdir=exp/tri_dnn_mmi
acwt=0.1
traindir=data/train_select/cm_fixed

# First we need to generate lattices and alignments:

#echo "sleep =========================="
#sleep 12h
false && \
{
steps/align_nnet_htk.sh --nj 10 --cmd "$train_cmd" \
   $traindir data/lang $srcdir ${srcdir}_ali_train || exit 1;
}

false && \
{
steps/make_denlats_nnet_htk.sh --nj 10 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt $acwt \
   $traindir data/lang $srcdir ${srcdir}_denlats_train || exit 1;
}


# Now we re-train the hybrid by single iteration of MMI
false && \
{
steps/nnet/train_mmi.sh --cmd "$cuda_cmd" --num-iters 1 --acwt $acwt\
  $traindir data/lang $srcdir \
  ${srcdir}_ali_train \
  ${srcdir}_denlats_train \
  $dir || exit 1;
}



