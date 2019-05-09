#!/bin/bash

# Copyright 2012  Karel Vesely (Brno University of Technology)
# Apache 2.0

# Train neural network

# Begin configuration.

# training options
learn_rate=0.008
momentum=0.9
l1_penalty=0
l2_penalty=1e-5
num_stream=40
batch_size=20
targets_delay=5
kld_scale=0.3
si_model=
# data processing
minibatch_size=40
randomizer_size=32768
randomizer_seed=777
feature_transform=
# learn rate scheduling
max_iters=20
min_iters=0 # keep training, disable weight rejection, start learn-rate halving as usual,
keep_lr_iters=0 # fix learning rate for N initial epochs,
#start_halving_inc=0.5
#end_halving_inc=0.1
start_halving_impr=0.5
end_halving_impr=0.1
halving_factor=0.5
# misc.
verbose=1
# tool
train_tool="nnet-train-frmshuff-parallel"
frame_weights=



criteria_type=TOKEN_ACCURACY
use_psgd=false
num_threads=4
asgd_lock=true
max_frame=25000 
keep_learnrate=0
block_shuffle=0
num_blocks=200

skip_frames=
sweep_frames=
skip_inner=false
sweep_loop=false
 
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Usage: $0 <mlp-init> <feats-tr> <feats-cv> <labels-tr> <labels-cv> <exp-dir>"
   echo " e.g.: $0 0.nnet scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark exp/dnn1"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

mlp_init=$1
feats_tr=$2
feats_cv=$3
labels_tr=$4
labels_cv=$5
dir=$6

tr_file=`echo $feats_tr | awk -F "scp:| " '{print $3;}'`
cv_file=`echo $feats_cv | awk -F "scp:| " '{print $3;}'`

sweep_loop=false
[ -z $sweep_frames ] && sweep_frames=`seq --separator=":" 0 $[$skip_frames-1]`
echo "--sweep-frames=$sweep_frames --sweep-loop=$sweep_loop"


[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet


# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

##############################
#start training

# choose mlp to start with
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
# optionally resume training from the best epoch
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)


# cross-validation on original network
false && \
{
 #--num-stream=$num_stream --batch-size=$batch_size --targets-delay=$targets_delay \
log=$dir/log/iter00.initial.log; hostname>$log
$train_tool --cross-validate=true \
 --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=false --verbose=$verbose \
 --skip-frames=$skip_frames --sweep-frames=$sweep_frames --sweep-loop=true --skip-inner=$skip_inner \
 --num-threads=$num_threads --use-psgd=$use_psgd --asgd-lock=$asgd_lock \
 ${feature_transform:+ --feature-transform=$feature_transform} \
 ${frame_weights:+ "--frame-weights=$frame_weights"} \
 "$feats_cv" "$labels_cv" $mlp_best \
 2>> $log || exit 1;

 loss=$(cat $dir/log/iter00.initial.log | grep "TOKEN_ACCURACY" | tail -n 1 | awk -F " |%" '{ print $3; }')
}
 loss=0.2
 loss_type=TOKEN_ACCURACY
 echo "CROSSVAL PRERUN $loss_type $(printf "%.2f%%" $loss)"
	
tr_sds=`mktemp $dir/tr_sds_XXXX.scp`
feats_tr=`echo $feats_tr | awk  -v f1=$tr_file -v f2=$tr_sds 'BEGIN{FS=f1}{print $1f2$2;}'`

# resume lr-halving
halving=0
sds_rate=1.0
min_sds_rate=0.5
max_minibatch_size=40
n=1

[ -e $dir/.halving ] && halving=$(cat $dir/.halving)
[ -e $dir/.sds ] &&  sds_rate=$(cat $dir/.sds)
[ -e $dir/.minibatch ] &&  minibatch_size=$(cat $dir/.minibatch)
[ -e $dir/.n ] &&  n=$(cat $dir/.n)

num_stream=$minibatch_size
num_blocks=1000
sweep_frame=$[($n-1)%2]

# training
for iter in $(seq -w $max_iters); do
  
  # sds
  {
     [ -e $tr_sds ] && rm $tr_sds
     #awk -v file=$tr_sds 'BEGIN{srand()}{if(rand() < '$sds_rate'){print $0 >> file}}' $tr_file
     awk -v SR=$sds_rate -v SK=$skip_frames 'BEGIN{srand();flag=0;}{if((NR-1)%SK!=0 && flag==1) print; if((NR-1)%SK==0){if(rand()<SR){print; flag=1;}else flag=0;}}' $tr_file > $tr_sds
     if [ 1 == $block_shuffle ]; then
        tmp_sds=`mktemp $dir/tr_sds_XXXX.scp`
        steps/nnet/block_shuffle.sh $num_blocks $tr_file > $tmp_sds
        mv $tmp_sds $tr_file
        echo block_shuffle: $block_shuffle
     fi

     echo sds_rate: $sds_rate
  }

  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  # skip iteration if already done
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue 
  
  # training
  log=$dir/log/iter${iter}.tr.log; hostname>$log
  $train_tool \
   --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
   --skip-frames=$skip_frames --sweep-frames=$sweep_frames --sweep-loop=$sweep_loop --skip-inner=$skip_inner \
   --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true --verbose=$verbose \
   --num-threads=$num_threads --use-psgd=$use_psgd --asgd-lock=$asgd_lock \
   --binary=true \
   ${kld_scale:+ --kld-scale="$kld_scale"} \
   ${si_model:+ --si-model="$si_model"} \
   ${feature_transform:+ --feature-transform=$feature_transform} \
   ${frame_weights:+ "--frame-weights=$frame_weights"} \
   ${randomizer_seed:+ --randomizer-seed=$randomizer_seed} \
   "$feats_tr" "$labels_tr" $mlp_best $mlp_next \
   2>> $log || exit 1; 

   tr_loss=$(cat $dir/log/iter${iter}.tr.log | grep "TOKEN_ACCURACY" | tail -n 1 | awk -F " |%" '{ print $3; }')
   echo -n "TRAIN TOKEN_ACCURACY $(printf "%.2f%%" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "
  
  # cross-validation
  log=$dir/log/iter${iter}.cv.log; hostname>$log
  $train_tool --cross-validate=true \
   --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=false --verbose=$verbose \
   --skip-frames=$skip_frames --sweep-frames=$sweep_frames --sweep-loop=true --skip-inner=$skip_inner \
   --num-threads=$num_threads --use-psgd=$use_psgd  --asgd-lock=$asgd_lock \
   ${feature_transform:+ --feature-transform=$feature_transform} \
   ${frame_weights:+ "--frame-weights=$frame_weights"} \
   "$feats_cv" "$labels_cv" $mlp_next \
   2>>$log || exit 1;
  
   loss_new=$(cat $dir/log/iter${iter}.cv.log | grep "TOKEN_ACCURACY" | tail -n 1 | awk -F " |%" '{ print $3; }')     
   echo -n "CROSSVAL TOKEN_ACCURACY $(printf "%.2f%%" $loss_new), "

  # accept or reject new parameters (based on objective function)
  loss_prev=$loss
  if [ 1 == $(bc <<< "$loss_new >= $loss") -o $iter -le $keep_lr_iters -o $iter -le $min_iters ]; then
    loss=$loss_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
    [ $iter -le $min_iters ] && mlp_best=${mlp_best}_min-iters-$min_iters
    [ $iter -le $keep_lr_iters ] && mlp_best=${mlp_best}_keep-lr-iters-$keep_lr_iters
    mv $mlp_next $mlp_best
    echo "nnet accepted ($(basename $mlp_best))"
    echo $mlp_best > $dir/.mlp_best 
    #minibatch
    minibatch_size=$[$minibatch_size+8]
    [ $minibatch_size -gt $max_minibatch_size ] && minibatch_size=$max_minibatch_size
    num_stream=$minibatch_size
    sds_rate=$(awk 'BEGIN{print( cos('$n'*13.0/180*3.1415) )}')
    if [[ $n -gt 6  || 1 == $(awk 'BEGIN{print('$sds_rate' < '$min_sds_rate' )}') ]]; then
     	sds_rate=$min_sds_rate
    fi
    n=$[$n+1]
    sweep_frames=`echo $sweep_frames | awk -F [:] '{for(i=1;i<NF;i++)printf("%s:",$(i+1));printf("%s",$1);}' `
    echo $sds_rate >$dir/.sds
    echo $minibatch_size >$dir/.minibatch
    echo $n >$dir/.n
    echo $sweep_frames >$dir/.sweep_frames
  else
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
    mv $mlp_next $mlp_reject
    echo "nnet rejected ($(basename $mlp_reject))"
  fi

  # create .done file as a mark that iteration is over
  touch $dir/.done_iter$iter
  
  # no learn-rate halving yet, if keep_lr_iters set accordingly
  [ $iter -le $keep_lr_iters ] && continue 

  # stopping criterion
  #rel_impr=$(bc <<< "scale=10; ($loss_prev-$loss)/$loss_prev")
  rel_impr=$(awk 'BEGIN{print('$loss'-('$loss_prev'))}' ) 
  if [ 1 == $halving -a 1 == $(awk 'BEGIN{print('$rel_impr' < '$end_halving_impr')}') ]; then
    if [ $iter -le $min_iters ]; then
      echo we were supposed to finish, but we continue as min_iters : $min_iters
      continue
    fi
    echo finished, too small rel. improvement $rel_impr
    break
  fi

  # start annealing when improvement is low
  if [ 1 == $(awk 'BEGIN{print( '$rel_impr' < '$start_halving_impr')}') ]; then
    halving=1
  elif [ 1 == $keep_learnrate ]; then
    halving=0
  fi
  echo $halving >$dir/.halving
  
  # do annealing
  if [ 1 == $halving ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    halving_factor=$(awk "BEGIN{print($halving_factor-0.025)}")
    if [ 1 == $(bc <<< "$halving_factor < 0.5") ]; then
	halving_factor=0.5
    fi
    echo $learn_rate >$dir/.learn_rate
  fi
done

# select the best network
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  "Error training neural network..."
  exit 1
fi
