#!/bin/bash

dict=$1
label=$2

awk 'NR==FNR{ss[$1"+"$2]=$3;}
     NR>FNR{
     if(FNR%2==1){num=1;for(i=2;i<=NF;i++)words[num++]=$i;} 
     if(FNR%2==0){ 
        printf $1;
        m = 2; 
        for(i=1;i<num;i++){
            w=words[i]; 
            
            for(;m<NF;m+=3) {
                if($m=="sil")printf(" sil %d", $(m+1))
                else break;
            }

            str=$m; 
            len=0;
            flag=0;
            tag="";
            taglen=0;
            while(m<NF){
                key=w"+"str;
                len+=$(m+1);
                if(key in ss){tag=ss[key];taglen=len;flag=1;}
                else if(flag==1) break;
                m+=3;
                if(m<NF)str=str"_"$m;
            }
            
            if(flag==1)printf(" %s %d", tag, taglen);
            if(m>=NF&&flag==0)break;
        }

        for(;m<NF;m+=3)
            if($m=="sil")printf(" sil %d", $(m+1))

        if(i<num || m<NF) printf" error";
        printf "\n";
    }
}' $dict $label
