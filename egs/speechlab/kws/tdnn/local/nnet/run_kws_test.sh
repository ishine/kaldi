#!/bin/bash

. path.sh


dir=exp/tdnn_baseline

data=/aifs/users/wd007/kws/test_set/Adult
data=/aifs/users/wd007/kws/test_set/Online
test_set="whx_online_0122"



for test in $test_set; do

cmvn_opts=
delta_opts=
D=$dir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
[ -e $D/skip_opts ] && skip_opts=$(cat $D/skip_opts)
[ -e $D/online ] && online=$(cat $D/online)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$data/$test/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a "$online" == "false" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$data/$test/utt2spk scp:$data/$test/cmvn.scp ark:- ark:- |"
[ ! -z "$cmvn_opts" -a "$online" == "true" ] && feats="$feats apply-cmvn-sliding $cmvn_opts ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# skip-frames (optional),
[ ! -z "$skip_opts" ] && nnet_forward_opts="$nnet_forward_opts $skip_opts"


    workdir=$dir/decode/$test
    workdir=$dir/decode/$test"_1"
    [ -d $workdir ] || mkdir -p $workdir;
    
    nnet-forward-parallel --use-gpu=yes --feature-transform=$dir/final.feature_transform $dir/final.nnet "$feats" ark,scp:$workdir/mlpoutput.ark,$workdir/mlpoutput.scp > $workdir/forward.log 2>&1 || exit -1
    #348:2:328:355  #348:369 #218:115:348:369
    keywords="324|193|137:138|177:178"
    keywords="218|115|324|193"
    for thd in 0.4 ; do
        nnet-kws-confidence --verbose=1 --keywords-id=$keywords --wakeup-threshold=$thd --smooth-window=7 --sliding-window=80 --word-interval=35 ark:$workdir/mlpoutput.ark ark,t:$workdir/smooth.txt ark,t:$workdir/confidence.txt > $workdir/rescore.$thd.log 2>&1 || exit -1
    done

    echo $test done
done

