#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Computes training alignments; assumes features are (LDA+MLLT or delta+delta-delta)
# + fMLLR (probably with SAT models).
# It first computes an alignment with the final.alimdl (or the final.mdl if final.alimdl
# is not present), then does 2 iterations of fMLLR estimation.

# If you supply the --use-graphs option, it will use the training
# graphs from the source directory (where the model is).  In this
# case the number of jobs must match the source directory.


# Begin configuration section.
nj=30
cmd=run.pl

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "usage: steps/align_to_phone.sh <align-dir> <phone-dir>"
   echo "e.g.:  steps/align_to_phone.sh exp/tri1_ali exp/tri1_ali_phone"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --fmllr-update-type (full|diag|offset|none)      # default full."
   exit 1;
fi

srcdir=$1
dir=$2

mdl=$srcdir/final.mdl



$cmd JOB=1:$nj $dir/log/ali_to_phone.JOB.log \
  ali-to-phones --per-frame=true $srcdir/final.mdl "ark:gunzip -c $srcdir/ali.JOB.gz|" "ark,t:$dir/ali.JOB.phone"
