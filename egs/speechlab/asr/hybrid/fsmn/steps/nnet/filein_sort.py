#!/usr/bin/python

import sys

#part-a6892fff-ae80-4b4e-8b72-284e945f075b raw_feat.1.ark 116
input=open(sys.argv[1])

block = {}
fn=""
for line in input:
    arr = line.split();
    if fn != arr[1]:
        dic = sorted(block.items(), key=lambda d:d[1]);
        for k,v in dic:
            print k,
        block = {};
        fn = arr[1];
        block[line] = int(arr[2]);
    else: 
        fn = arr[1];
        block[line] = int(arr[2]);

dic = sorted(block.iteritems(), key=lambda d:d[1]);
for k,v in dic: 
    print k,
    

