#!/bin/bash

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script compiles the ARPA-formatted language models into FSTs. Finally it composes the LM, lexicon
# and token FSTs together into the decoding graph. 

if [ $# != 3 ]; then
   echo "Usage: local/make_ctc_decode_graph.sh <lms-dir> <lang-dir> <graphdir>"
   echo "e.g.: local/make_ctc_decode_graph.sh lms data/lang_phn data/graph"
   exit 1;
fi

. ./path.sh || exit 1;

lmdir=$1
langdir=$2
graphdir=$3
tmpdir=$3/graph_tmp
mkdir -p $tmpdir

for f in $lmdir/lm.arpa.gz ${langdir}/words.txt ; do
   [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# These language models have been obtained when you run local/wsj_data_prep.sh
echo "Preparing language models for testing, may take some time ... "

    cp ${langdir}/words.txt $graphdir || exit 1;

    gunzip -c $lmdir/lm.arpa.gz | \
     utils/find_arpa_oovs.pl $graphdir/words.txt  > $tmpdir/oovs_lm.txt

    # grep -v '<s> <s>' because the LM seems to have some strange and useless
    # stuff in it with multiple <s>'s in the history.
    gunzip -c $lmdir/lm.arpa.gz | \
      grep -v '<s> <s>' | \
      grep -v '</s> <s>' | \
      grep -v '</s> </s>' | \
      arpa2fst - | fstprint | \
      utils/remove_oovs.pl $tmpdir/oovs_lm.txt | \
      utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$graphdir/words.txt \
        --osymbols=$graphdir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
       fstrmepsilon | fstarcsort --sort_type=ilabel > $graphdir/G.fst
    
    # Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
    # minimized.
    fsttablecompose ${langdir}/L.fst $graphdir/G.fst | fstdeterminizestar --use-log=true | \
      fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;
    fsttablecompose ${langdir}/T.fst $tmpdir/LG.fst > $graphdir/TLG.fst || exit 1;

echo "Composing decoding graph TLG.fst succeeded"
#rm -r $tmpdir
