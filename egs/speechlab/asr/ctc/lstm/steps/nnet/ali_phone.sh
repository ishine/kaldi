#!/bin/bash
#set -x

if [ $# -ne 3 ];then
    echo "Usage: $0 gmmali lang dir."
    exit;
fi

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


gmmali=$1
lang=$2
dir=$3

[ -d $dir ] || mkdir -p $dir

for f in $gmmali/final.mdl $gmmali/ali.scp $lang/phonelist.txt $lang/statelist.txt ; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

awk 'BEGIN{i=0;}{print i,$0;i++;}' $lang/statelist.txt > $dir/statelist
awk 'BEGIN{i=1;}{if($0 != "sil"){print i,$0;i++;} else print 0, "sil";}' $lang/phonelist.txt > $dir/phonelist
awk 'NR==FNR{phone[$2]=$1;}NR>FNR{split($2,ss,"_"); print $1,phone[ss[1]], ss[1], $2;}' $dir/phonelist $dir/statelist > $dir/state2phone

ali-to-pdf $gmmali/final.mdl scp:$gmmali/ali.scp ark:- | copy-align ark:- ark,t:$dir/ali.state.txt || exit 1

awk 'NR==FNR{state[$1]=$2;}NR>FNR{printf("%s", $1); for(i=2;i<=NF;i++)printf(" %s", state[$i]); printf("\n");}' $dir/state2phone $dir/ali.state.txt > $dir/ali.phone.txt

cat $dir/ali.phone.txt | copy-align ark:- ark,scp:$dir/ali.phone.ark,$dir/ali.scp || exit 1

awk 'NR==FNR{ss[$1]=$0;}NR>FNR{if($1 in ss) print ss[$1];}'  $dir/ali.scp $gmmali/ali.cv.scp > $dir/ali.cv.scp
awk 'NR==FNR{ss[$1]=$0;}NR>FNR{if($1 in ss) print ss[$1];}'  $dir/ali.scp $gmmali/ali.tr.scp > $dir/ali.tr.scp
