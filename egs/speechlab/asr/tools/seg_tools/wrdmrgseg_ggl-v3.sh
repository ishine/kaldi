#h/bin/bash
# 使用google工具进行分词
# 输入文件和词典以及输出文件都是utf8编码
# 输入词典带词频


dict=$1
infn=$2
outfn=$3
AddMark=$4 #AddMark NoMark
oov_percent=$5
single_word_percent=$6

if [[ "$AddMark" != ""  &&  "$AddMark" != "AddMark" ]];then
   echo "Usage: ./wrdmrgseg_ggl-v1.sh dict infn outfn AddMark [oov_percent] [single_word_percent]";
   exit;
fi


segtool=./tools/segmenter-v3.pl
gz=`echo $infn | awk '{if($1~/(gz|gzip)$/) print "gzip"; else print "text";}'`

outfn1=${outfn}.1
outfn2=${outfn}.2

if [ $gz == "gzip" ]
then
gzip -d -c $infn > $outfn1 
else
outfn1=$infn
fi

$segtool $dict < $outfn1 > $outfn2

# awk '{printf("%s",$1); if($1~/[aA-zZ]/) first=0; else first=1; for(i=2;i<=NF;i++) {if($i~/[aA-zZ]/) {if(first==1) {printf(" %s",$i);first=0;} else printf("%s",$i); } else {printf(" %s",$i);first=1;}} printf("\n");}' ${outfn}.2 > ${outfn}
#[AddMark] [oov_percent] [single_word_percent]
python2 ./tools/mergenewword.updated.py $dict $outfn2 ${outfn} ${AddMark} ${oov_percent} ${single_word_percent}


if [ $gz == "gzip" ]
then
rm -f $outfn1
fi

rm -f $outfn2


