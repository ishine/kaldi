$dir=exp/chain/
$iter=1800

if [ ! -d $dir ]
  echo "src dir is not exist!" && exit;
fi

rm $dir/accuracy.log $dir/norm_param.log $dir/param_change.log

for i in `seq 1 $iter`; do 
  grep "Norms of parameter matrices" progress.$i.log >>norm_param.log
  grep "Relative parameter differences" progress.$i.log >>param_change.log
done

for i in `seq 0 $iter-1`; do
  grep "'output'" compute_prob_valid.$i.log >>accuracy.log
done

