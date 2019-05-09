#!/bin/bash

# usage input_file output_file

if [ $# -ne 2 ];
then
    echo "usage input_file output_file"
	exit -1;
fi

awk 'BEGIN{ i = 0;}
    {a[i]=$0; i++;}
    END{ 
        for (j=0;j<NR;j++)
        {
            t = int(rand()* (NR-j) + j);
            tmp = a[t];
            a[t] = a[j];
            print tmp;
        }
    }' $1 > $2

