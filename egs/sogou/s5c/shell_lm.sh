#!/bin/bash -u

if [ $# != 3 ]
then
	echo $0 word_txt lm_arpa out_dir
	exit 1;
fi

word_txt=$1
lm_arpa=$2
out_dir=$3

date

#ln -s /disk2/suxia/build_kclg_new/shells/kalditools/scripts utils
#export PATH=$PATH:/disk2/suxia/build_kclg_new/shells/kalditools

mkdir -p $out_dir

cp $word_txt $out_dir/words.txt

cat $lm_arpa | utils/find_arpa_oovs.pl $out_dir/words.txt \
> $out_dir/oovs_lm.txt


cat $lm_arpa \
| egrep -v '<s> <s>|</s> <s>|</s> </s>' \
| arpa2fst - | fstprint \
| utils/remove_oovs.pl $out_dir/oovs_lm.txt \
| utils/eps2disambig.pl | utils/s2eps.pl \
| fstcompile --isymbols=$out_dir/words.txt  --osymbols=$out_dir/words.txt \
 --keep_isymbols=false --keep_osymbols=false \
| fstrmepsilon | fstarcsort --sort_type=ilabel > $out_dir/G.fst

date
