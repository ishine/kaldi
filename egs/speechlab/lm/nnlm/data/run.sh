#!/bin/bash

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
cat $rawdir/train.txt $rawdir/valid.txt | awk '{for(i=1;i<=NF;i++)ss[$i]=1;}END{for(c in ss) print c;}' | sort -u > $lang/words_sort.txt
utils/shuffle.sh $lang/words_sort.txt | grep -v "<s>" | grep -v "</s>" | grep -v "<unk>" > $lang/words_random.txt

if [ ! -z $dict ]; then
    awk 'BEGIN{i=0;}NR==FNR{ss[$1]=1;}NR>FNR{if($0 in ss) {print $0,i; i++;} }END{print "<s>",i; print "</s>",i+1; print "<unk>",i+2;}' $dict $lang/words_random.txt > $lang/vocab.txt
else
    awk '{print $1,NR-1;}END{print "<s>",NR; print "</s>",NR+1; print "<unk>",NR+2;}' $lang/words_random.txt > $lang/vocab.txt
fi

wn=`wc -l $lang/vocab.txt | awk '{print $1;}'`
awk -v cn=$cn -v wn=$wn 'BEGIN{clen=wn/cn;n=1;printf("[ 0");}END{for(i=1;i<wn;i++){if(i>n*clen){printf(" %s", i);n++;}} printf(" %s ]\n", wn);}' $lang/vocab.txt > $lang/class_boundary.txt
}


#false && \
{
[ ! -d $traindir ] && mkdir -p $traindir
awk '{print "utt_"NR, "<s>", $0, "</s>";}' $rawdir/train.txt > $traindir/text
utils/shuffle.sh $traindir/text > $traindir/text.random
awk 'NR==FNR{ss[$1]=$2;}NR>FNR{printf("%s",$1);for(i=2;i<=NF;i++){if($i in ss)printf(" %s",ss[$i]);else printf(" %s",ss["<unk>"]);} printf"\n";}' $lang/vocab.txt $traindir/text.random | copy-align ark:- ark,scp:`pwd`/$traindir/feats.ark,$traindir/feats.scp
#utils/sym2int.pl -f 2- $lang/vocab.txt $traindir/text.random | copy-align ark:- ark,scp:`pwd`/$traindir/feats.ark,$traindir/feats.scp
awk '{print $1, NF-2;}' $traindir/text.random > $traindir/featlen.scp

[ ! -d $cvdir ] && mkdir -p $cvdir
awk '{print "utt_"NR, "<s>", $0, "</s>";}' $rawdir/valid.txt > $cvdir/text
utils/shuffle.sh $cvdir/text > $cvdir/text.random
awk 'NR==FNR{ss[$1]=$2;}NR>FNR{printf("%s",$1);for(i=2;i<=NF;i++){if($i in ss)printf(" %s",ss[$i]);else printf(" %s",ss["<unk>"]);} printf"\n";}' $lang/vocab.txt $cvdir/text.random | copy-align ark:- ark,scp:`pwd`/$cvdir/feats.ark,$cvdir/feats.scp
awk '{print $1, NF-2;}' $cvdir/text.random > $cvdir/featlen.scp
}


