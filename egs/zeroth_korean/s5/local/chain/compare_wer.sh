#!/bin/bash

# this script is used for comparing decoding results between systems.
# e.g. local/chain/compare_wer.sh exp/chain/tdnn_{c,d}_sp
# For use with discriminatively trained systems you specify the epochs after a colon:
# for instance,
# local/chain/compare_wer.sh exp/chain/tdnn_c_sp exp/chain/tdnn_c_sp_smbr:{1,2,3}


if [ $# == 0 ]; then
  echo "Usage: $0: <dir1> [<dir2> ... ]"
  echo "e.g.: $0 exp/chain/tdnn_{b,c}_sp"
  echo "or (with epoch numbers for discriminative training):"
  echo "$0 exp/chain/tdnn_b_sp_disc:{1,2,3}"
  exit 1
fi

echo "# $0 $*"

used_epochs=false

# this function set_names is used to separate the epoch-related parts of the name
# [for discriminative training] and the regular parts of the name.
# If called with a colon-free directory name, like:
#  set_names exp/chain/tdnn_lstm1e_sp_bi_smbr
# it will set dir=exp/chain/tdnn_lstm1e_sp_bi_smbr and epoch_infix=""
# If called with something like:
#  set_names exp/chain/tdnn_d_sp_smbr:3
# it will set dir=exp/chain/tdnn_d_sp_smbr and epoch_infix="_epoch3"


set_names() {
  if [ $# != 1 ]; then
    echo "compare_wer_general.sh: internal error"
    exit 1  # exit the program
  fi
  dirname=$(echo $1 | cut -d: -f1)
  epoch=$(echo $1 | cut -s -d: -f2)
  if [ -z $epoch ]; then
    epoch_infix=""
  else
    used_epochs=true
    epoch_infix=_epoch${epoch}
  fi
}



echo -n "# System               "
for x in $*; do   printf "% 10s" " $(basename $x)";   done
echo

strings=(
  "#WER test_clean (tgsmall)          "
  "#WER test_clean (fglarge)            ")

for n in 0 1 ; do
   echo -n "${strings[$n]}"
   for x in $*; do
     set_names $x  # sets $dirname and $epoch_infix
     decode_names=(tgsmall_test_clean fglarge_test_clean)

     wer=$(grep WER ${dirname}_online/decode_${decode_names[$n]}/wer_* | utils/best_wer.sh | awk '{print $2}')
     printf "% 10s" $wer
   done
   echo
done


if $used_epochs; then
  exit 0;  # the diagnostics aren't comparable between regular and discriminatively trained systems.
fi


echo -n "# Final train prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final train prob (xent)"
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob (xent)"
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "# Num-params              "
for x in $*; do
  printf "% 10s" $(grep num-parameters $x/log/progress.1.log | awk '{print $2}')
done
echo
