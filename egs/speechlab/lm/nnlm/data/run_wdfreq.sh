#!/bin/bash
#set -x

if [ $# -lt 5 ];
then
    echo "Usage $0 rawdir traindir cvdir lang nclass dict(optinal)"
    exit 1; 
fi

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 

rawdir=$1
traindir=$2
cvdir=$3
lang=$4
cn=$5
dict=$6


for f in $rawdir/train.txt $rawdir/valid.txt; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

false && \
{
[ ! -d $lang ] && mkdir -p $lang

cat $rawdir/train.txt $rawdir/valid.txt > $rawdir/tmp.txt
if [ -f $dict ]; then
    awk 'NR==FNR{ss[$1]=0;}NR>FNR{for(i=1;i<=NF;i++){if($i in ss)ss[$i]++; else ss["<unk>"]++;} ss["<s>"]++; ss["</s>"]++;}END{for(c in ss){ if(ss[c]>4)print c, ss[c];}}' $dict $rawdir/tmp.txt | sort -n -k 2 > $lang/words_sort.txt 
else
    awk '{for(i=1;i<=NF;i++){if($i in ss)ss[$i]++; else ss[$i]=1;} ss["<s>"]++; ss["</s>"]++;}END{ss["<unk>"]=0; for(c in ss) print c, ss[c];}' | sort -n -k 2 > $lang/words_sort.txt
fi
    
awk 'BEGIN{i=0;}{print $1, i; i++;}' $lang/words_sort.txt > $lang/vocab.txt
}

wn=`wc -l $lang/vocab.txt | awk '{print $1;}'`
freq=`awk '{sum+=$2;}END{print sum;}' $lang/words_sort.txt ` 
echo wn=$wn, freq=$freq
awk -v cn=$cn -v wn=$wn -v freq=$freq 'BEGIN{clen=freq/cn;n=1;printf("[ 0");cf=0;}{cf+=$2; if(cf>n*clen){printf(" %s", NR-1);n++;} }END{printf(" %s ]\n", wn);}' $lang/words_sort.txt > $lang/class_boundary.txt


false && \
{
[ ! -d $traindir ] && mkdir -p $traindir
awk '{print "utt_"NR, "<s>", $0, "</s>";}' $rawdir/train.txt > $traindir/text
shuf $traindir/text > $traindir/text.random
awk 'NR==FNR{ss[$1]=$2;}NR>FNR{printf("%s",$1);for(i=2;i<=NF;i++){if($i in ss)printf(" %s",ss[$i]);else printf(" %s",ss["<unk>"]);} printf"\n";}' $lang/vocab.txt $traindir/text.random | copy-align ark:- ark,scp:`pwd`/$traindir/feats.ark,$traindir/feats.scp
#utils/sym2int.pl -f 2- $lang/vocab.txt $traindir/text.random | copy-align ark:- ark,scp:`pwd`/$traindir/feats.ark,$traindir/feats.scp
awk '{print $1, NF-2;}' $traindir/text.random > $traindir/featlen.scp

[ ! -d $cvdir ] && mkdir -p $cvdir
awk '{print "utt_"NR, "<s>", $0, "</s>";}' $rawdir/valid.txt > $cvdir/text
shuf $cvdir/text > $cvdir/text.random
awk 'NR==FNR{ss[$1]=$2;}NR>FNR{printf("%s",$1);for(i=2;i<=NF;i++){if($i in ss)printf(" %s",ss[$i]);else printf(" %s",ss["<unk>"]);} printf"\n";}' $lang/vocab.txt $cvdir/text.random | copy-align ark:- ark,scp:`pwd`/$cvdir/feats.ark,$cvdir/feats.scp
awk '{print $1, NF-2;}' $cvdir/text.random > $cvdir/featlen.scp
}


