#!/bin/bash

if [ $# -ne 5 ];then
    echo "usage: $0 numjobs mergesize train.scp featslen.scp lat.scp"
    exit;
fi

#set -x

numjobs=$1
mergesize=$2
featscp=$3
lenscp=$4
latscp=$5

featlen=/sgfs/users/wd007/src/dnn/kaldi/src/featbin/feat-to-len


bn=`basename $featscp`
dir=`dirname $featscp`

suffix=${bn##*.}
fn=${bn%.*}

#$featlen scp:$3 ark,t:$dir/featlen.scp

awk 'NR==FNR{aa[$1]=$0;}NR>FNR{if($1 in aa) print aa[$1];}' $lenscp $featscp > $dir/featlen.scp

total_frames=`awk '{frames+=$2;}END{print frames;}' $dir/featlen.scp`

paste $featscp $dir/featlen.scp > $dir/tmp.scp

rm -f $dir/$fn"."*.$suffix

awk -v N=$total_frames -v JB=$numjobs -v MG=$mergesize -v SF=$suffix -v FN=$fn -v DR=$dir \
	'BEGIN{ num = int(N/(JB*MG)); frames=0; j=1;i=1;k=1;}
	 { 
		if (frames+$4>MG){i++;j=i%JB;frames=0;}

		if (i<=num*JB)
		{
			print $1,$2 >> DR"/"FN"."j"."SF;
			print $1,$4 >> DR"/"FN"."j"."SF".len"; 
			frames += $4;
		}
		else
		{
			print $1,$2 >> DR"/"FN"."k"."SF;
			print $1,$4 >> DR"/"FN"."k"."SF".len";
			k++; k=k%JB;
		}
		
	 }' $dir/tmp.scp

rm $dir/tmp.scp 
	
for i in `seq 0 $[numjobs-1]`;
do
	featfn=$dir/$fn.$i.$suffix
	awk 'NR==FNR{a[$1]=$0;}NR>FNR{if($1 in a) print a[$1];}' $latscp $featfn > $dir/lat.$i.$suffix
done
