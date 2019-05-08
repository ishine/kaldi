#!/bin/bash

if [ $# -ne 3 ];then
    echo "Usage: $0 aliphonedir alieesendir dir."
    exit;
fi

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


aliphone=$1
alieesen=$2
dir=$3


for f in $aliphone/ali.phone.txt ; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

[ ! -d $dir ] && mkdir -p $dir;


awk '{printf("%s", $1); for (i=2;i<=NF;i++){if($i != "0")printf(" %s", $i);}printf("\n");}' $aliphone/ali.phone.txt > $dir/ali.phone.txt
awk '{printf("%s %s", $1, $2); for(i=3;i<=NF;i++){if($i != $(i-1))printf(" %s", $i);} printf("\n");}' $dir/ali.phone.txt > $dir/ali.phone.unique.txt 

awk 'NR==FNR{ss[$1]=$0;}NR>FNR{ if($1 in ss){printf("%s",$1);n=split(ss[$1],arr," ");i=2;j=2; while(i<=n && j<=NF){if(arr[i] != $j && arr[i]==arr[i-1]){printf(" %s",arr[i]);i++;}else {printf(" %s",$j); i++;j++;}} while(j<=NF){printf(" %s",$j);j++;} printf("\n");}}' $alieesen/labels  $dir/ali.phone.unique.txt > $dir/ali.phone.ce.txt

cat $dir/ali.phone.ce.txt | copy-align ark:- ark,scp:$dir/ali.phone.ce.ark,$dir/ali.scp || exit 1
