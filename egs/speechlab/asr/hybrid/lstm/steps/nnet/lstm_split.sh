#!/bin/bash

if [ $# -ne 7 ];then
    echo "usage: $0 numjobs mergesize skipframes train.scp featlen.scp ali.scp flag(tr|cv)"
    exit;
fi

#set -x

numjobs=$1
mergesize=$2
skipframes=$3
featscp=$4
lenscp=$5
aliscp=$6
flag=$7

featlen=/sgfs/users/wd007/src/dnn/kaldi/src/featbin/feat-to-len


bn=`basename $featscp`
dir=`dirname $featscp`

suffix=${bn##*.}
fn=${bn%.*}

#$featlen scp:$3 ark,t:$dir/featlen.scp


awk 'NR==FNR{a[$1]=$2;}NR>FNR{if($1 in a) print $0, a[$1];}' $lenscp $featscp > $dir/tmp.scp

total_frames=`awk '{frames+=$3;}END{print frames;}' $dir/tmp.scp`

rm -f $dir/$fn"."*."$suffix"*

awk -v N=$total_frames -v JB=$numjobs -v MG=$mergesize -v SK=$skipframes -v SF=$suffix -v FN=$fn -v DR=$dir \
	'BEGIN{ job_frames=int(N/JB); for(i=0;i<JB;i++)count[i]=0; frames=0; i=0;j=-1;k=0;}
	 { 
        if((NR-1)%SK!=0)
        {
            print $1,$2 >> DR"/"FN"."j"."SF;
            print $1,$3 >> DR"/"FN"."j"."SF".len";
		    count[j] += $3;
            next;
        }

		flag=0;
		j = (j+1)%JB;
		for(i=0;i<JB;i++)
		{
			j=(j+i)%JB;
			if (count[j]<job_frames)
			{
				print $1,$2 >> DR"/"FN"."j"."SF;
				print $1,$3 >> DR"/"FN"."j"."SF".len"; 
				count[j] += $3;
				flag=1;
				break;
			}
		}
		
		if (flag==0)
		{
			print $1,$2 >> DR"/"FN"."j"."SF;
                       	print $1,$3 >> DR"/"FN"."j"."SF".len"; 			
			count[j] += $3;
			j = (j+1)%JB;
		}
			
	 }' $dir/tmp.scp

rm $dir/tmp.scp 

#false && \
{
for i in `seq 0 $[numjobs-1]`;
do
        featfn=$dir/$fn.$i.$suffix
        #awk 'NR==FNR{a[$1]=$0;}NR>FNR{if($1 in a) print a[$1];}' $latscp $featfn > $dir/lat.$i.$suffix
        #awk -v alidir=$alidir 'BEGIN{while("gunzip -c $alidir/ali.*.gz"|getline ss) {split(ss, arr, " "); a[arr[1]]=ss;} }{if($1 in a) print a[$1];}' $featfn | gzip -c > $dir/ali.$i.gz

        awk 'NR==FNR{a[$1]=$0;}NR>FNR{if($1 in a) print a[$1];}' $aliscp $featfn > $dir/ali.$flag.$i.$suffix
done
}

