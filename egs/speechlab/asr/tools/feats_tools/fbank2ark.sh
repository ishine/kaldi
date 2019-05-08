
if [ $# != 2 ]; then
   echo "usage: fbank2ark.sh <fbankscp> <path-to-fbankdir>";
   exit 1;
fi

fbankscp=$1
dir=$2

compress=false
cmd="queue.pl -l hostname=shenzhen"
cmd=queue.pl
nj=1000
mj=50


[ -f path.sh ] && . path.sh 

#rm -rf $dir
[ ! -d $dir ]  && mkdir -p $dir

#awk -F "/|\\\.fbank" '{print $(NF-1), $0;}' $fbanklist > $dir/$fbankscp
cp $fbankscp $dir/fbank_org.scp

split_scps=""
  for k in $(seq $nj); do
    split_scps="$split_scps $dir/fbank_org.$k.scp"
  done

#false && \
{
utils/split_scp.pl $dir/fbank_org.scp $split_scps || exit 1;

$cmd --max-jobs-run $mj JOB=1:$nj $dir/fbank2ark.JOB.log \
    copy-feats --compress=$compress scp,p:$dir/fbank_org.JOB.scp ark,scp:`pwd`/$dir/raw_feat.JOB.ark,`pwd`/$dir/raw_feat.JOB.scp  || exit 1;

  for ((k=1; k<=nj; k++)); do
    cat $dir/raw_feat.${k}.scp
  done > $dir/raw_feat.scp
}

cd $dir
ln -s raw_feat.scp feats.scp
cd ..

(feat-to-len scp:$dir/feats.scp ark,t:$dir/featlen.scp > feat2len.log 2>&1 || exit 1)&

#compute-cmvn-stats --spk2utt=ark:$dir/spk2utt scp:$dir/feats.scp \
compute-cmvn-stats  scp:$dir/feats.scp \
    ark,scp:`pwd`/$dir/cmvn.ark,`pwd`/$dir/cmvn.scp || exit 1

# spk
awk '{print $1, $1;}' $dir/feats.scp > $dir/utt2spk
cp $dir/utt2spk $dir/spk2utt

exit 0

