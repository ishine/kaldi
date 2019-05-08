#! /bin/bash

if [ $# -lt 1 ];then
    echo "usage: $0 numpiece file"
    exit;
fi
len=`wc -l $2 | awk '{print $1}'`
if [ $len -lt $1 ];then
    echo "Too few files for splitting! Make sure the file number is larger than the piece number."
fi
suffix=${2##*.}
fn=${2%.*}
rm -f $fn"_"*.$suffix
awk -v L=$len -v N=$1 'BEGIN{PL=int(L/N);LL=L-(N-1)*PL;c=0;i=1;}
     i==N {print >> FN"_"N"."SF;}
     (c<PL) && (i<N)  {print >> FN"_"i"."SF; c++ ; if (c>=PL) {c=0; i++}}' SF=$suffix FN=$fn $2
