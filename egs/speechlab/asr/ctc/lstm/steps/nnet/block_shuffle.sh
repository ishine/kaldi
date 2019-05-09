#!/bin/bash

if [ $# -ne 2 ];
then
    echo "Usage: block_shuffle.sh num_blocks input_file "
        exit -1; 
fi

bn=$1
in=$2

len=`wc -l $in | awk '{print $1;}'`
bz=`awk 'BEGIN{print int(('$len'+'$bn'-1)/'$bn');}'`

awk -v bz=$bz 'BEGIN{ i = 0; j = 0;srand()}
    { if(j<bz){a[i,j]=$0;j++;} if(j==bz){j=0;i++;} }
    END{ 
	for (m=0;m<=i;m++)
	    mask[m]=m;
        for (m=0;m<=i;m++)
        {
            t = int(rand()* (i-m) + m);
            tmp = mask[t];
            mask[t] = mask[m];
	    mask[m] = tmp;
        }
	for (m=0;m<=i;m++)
	{
	   if(mask[m]<i){
	   	for (n=0;n<bz;n++)
	           print a[mask[m],n];
	   }
	   else{
		for (n=0;n<j;n++)
		   print a[mask[m],n];
	   }
	}
    }' $in 
