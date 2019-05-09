

if [ $# != 3 ]; then
   echo "usage: make_plp.sh <plp_config> <data-dir> <path-to-plpdir>";
   exit 1;
fi

plp_config=$1
wav_dir=$2
dir=$3

compress=false
name=`basename $wav_dir`
wavlist=$name".wavlist"

[ -f path.sh ] && . path.sh 

#rm -rf $dir
[ ! -d $dir ]  && mkdir -p $dir

false && \
{
for fn in $wavlist $name".char.mlf" text utt2spk spk2utt; do
    [ ! -f $wav_dir/$fn ] && echo "$fn not found!" && exit 1;
    cp $wav_dir/$fn $dir/$fn
done
}

# text
#grep -v '#!MLF!#' $wav_dir/$name".char.mlf" |  awk -F "/|\.lab\"" '{if(NF==3)printf("%s",$2);else if($0==".")printf("\n");else printf(" %s",$0);}' > $dir/text || exit 1;
# spk
awk -F "/|\\\.wav" '{print $(NF-1), $(NF-1);}' $wav_dir/$wavlist > $dir/utt2spk || exit 1;
cp $dir/utt2spk $dir/spk2utt


wavscp=$name".wav.scp"
awk -F "/|\\\.wav" '{print $(NF-1), $0;}' $wav_dir/$wavlist > $dir/$wavscp

compute-plp-feats --verbose=1 --config=$plp_config scp,p:$dir/$wavscp ark:- | \
    copy-feats --compress=$compress ark:- \
     ark,scp:`pwd`/$dir/feats.ark,`pwd`/$dir/feats.scp  > $dir/plp".log" 2>&1 2>&1 || exit 1;

feat-to-len scp:$dir/feats.scp ark,t:$dir/feats.lengths || exit 1;

#compute-cmvn-stats --spk2utt=ark:$dir/spk2utt scp:$dir/feats.scp \
compute-cmvn-stats  scp:$dir/feats.scp \
    ark,scp:`pwd`/$dir/cmvn.ark,`pwd`/$dir/cmvn.scp || exit 1;

exit 0

