#!/bin/bash

if [ $# -ne 3 ];then
    echo "Usage: $0 aligmmdir lang dir."
    exit;
fi

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 


aligmm=$1
lang=$2
dir=$3
silid=1


for f in $aligmm/ali.scp ; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

[ ! -d $dir ] && mkdir -p $dir;



#awk 'BEGIN{i=1;j=1;}{if($0 != "sil"){print i,$0,j;i++;} else {print 0, "sil", j;} j++;}' $lang/phonelist.txt > $dir/phonelist
grep -v -E "\#|<" $lang/phones.txt | awk 'BEGIN{i=1;}{if($1 != "sil"){print i,$0;i++;} else {print 0, $0;} }' - > $dir/phonelist

ali-to-phones --write-lengths $aligmm/final.mdl scp:$aligmm/ali.scp ark,t:$dir/ali.phone.length.txt || exit 1;

#awk -v sil=$silid '{printf("%s",$1); if($2!=sil)printf(" %s 1 %s %s",sil,$2,$3-1); else printf(" %s %s",$2,$3);for(i=5;i<=NF-2;i++) if($i != ";")printf(" %s",$i);  if($(NF-1)!=sil)printf(" %s %s %s 1",$(NF-1),$NF-1,sil); else printf(" %s %s",$(NF-1),$NF); printf("\n");}' $dir/ali.phone.length.txt > $dir/ali.phone.format.txt 

#awk  'NR==FNR{map[$3]=$1;}NR>FNR{printf("%s",$1); for(i=2;i<NF;i+=2){if($i in map){for(j=0;j<$(i+1);j++) printf(" %s",map[$i]);} else printf(" error"); }  printf("\n"); }' $dir/phonelist $dir/ali.phone.format.txt >  $dir/ali.phone.txt
awk  'NR==FNR{map[$3]=$1;}NR>FNR{printf("%s",$1); for(i=2;i<NF;i+=3){if($i in map){for(j=0;j<$(i+1);j++) printf(" %s",map[$i]);} else printf(" error"); }  printf("\n"); }' $dir/phonelist $dir/ali.phone.length.txt >  $dir/ali.phone.txt

cat $dir/ali.phone.txt | copy-align ark:- ark,scp:$dir/ali.phone.ark,$dir/ali.scp || exit 1

awk 'NR==FNR{ss[$1]=$0;}NR>FNR{if($1 in ss) print ss[$1];}'  $dir/ali.scp $aligmm/ali.cv.scp > $dir/ali.cv.scp
awk 'NR==FNR{ss[$1]=$0;}NR>FNR{if($1 in ss) print ss[$1];}'  $dir/ali.scp $aligmm/ali.tr.scp > $dir/ali.tr.scp
