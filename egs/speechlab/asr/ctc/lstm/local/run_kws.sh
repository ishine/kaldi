#!/bin/bash

workdir=kws
mkdir -p $workdir

srcdict=/aifs/users/wd007/asr/baseline_chn_10000h/data/resource/lexicon.txt
srclm=/aifs/users/wd007/asr/baseline_500h/nn/ctc/phone/s6/data/lms/kws.arpa

[ -f path.sh ] && . path.sh
cd $workdir

  echo =====================================================================
  echo "             Data Preparation and FST Construction                 "
  echo =====================================================================


#false && \
{

cat $srcdict | grep -v "<s>" | grep -v "</s>" | grep -v "sil" | LANG= LC_ALL= sort > lexicon.txt  || exit 1;

# Get the set of lexicon units without sil
#cut -d' ' -f2- lexicon.txt | tr ' ' '\n' | sort -u > units_nosil.txt
awk '{for(i=2;i<=NF;i++)if($i in ss){}else{ss[$i]=1;print $i;}}' lexicon.txt | sort -u > units_nosil.txt

# The complete set of lexicon units, indexed by numbers starting from 1
awk '{print $1 " " NR}' units_nosil.txt > units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
sym2int.pl -f 2- units.txt < lexicon.txt > lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"


tmpdir=token_tmp
[ -d $tmpdir ] || mkdir $tmpdir

# Add probabilities to lexicon entries. There is in fact no point of doing this here since all the entries have 1.0.
# But utils/make_lexicon_fst.pl requires a probabilistic version, so we just leave it as it is. 
perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < lexicon.txt > $tmpdir/lexiconp.txt || exit 1;

# Add disambiguation symbols to the lexicon. This is necessary for determinizing the composition of L.fst and G.fst.
# Without these symbols, determinization will fail. 
ndisambig=`add_lex_disambig.pl $tmpdir/lexiconp.txt $tmpdir/lexiconp_disambig.txt`
ndisambig=$[$ndisambig+1];

( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) > $tmpdir/disambig.list

# Get the full list of CTC tokens used in FST. These tokens include <eps>, the blank <blk>, the actual labels (e.g.,
# phonemes), and the disambiguation symbols. 
cat units.txt | awk '{print $1}' > $tmpdir/units.list
(echo '<eps>'; echo '<blk>';) | cat - $tmpdir/units.list $tmpdir/disambig.list | awk '{print $1 " " (NR-1)}' > tokens.txt

# Compile the tokens into FST
ctc_token_fst.py tokens.txt | fstcompile --isymbols=tokens.txt --osymbols=tokens.txt \
   --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > T.fst || exit 1;

# Encode the words with indices. Will be used in lexicon and language model FST compiling. 
cat $tmpdir/lexiconp.txt | awk '{print $1}' | sort | uniq  | awk '
  BEGIN {
    print "<eps> 0";
  } 
  {
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
  }' > words.txt || exit 1;

# Now compile the lexicon FST. Depending on the size of your lexicon, it may take some time. 
token_disambig_symbol=`grep \#0 tokens.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 words.txt | awk '{print $2}'`

make_lexicon_fst.pl --pron-probs $tmpdir/lexiconp_disambig.txt 0 "sil" '#'$ndisambig | \
       fstcompile --isymbols=tokens.txt --osymbols=words.txt \
       --keep_isymbols=false --keep_osymbols=false |   \
       fstaddselfloops  "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" | \
       fstarcsort --sort_type=olabel > L.fst || exit 1;

echo "Dict and token FSTs compiling succeeded"


tmpdir=graph_tmp
[ -d $tmpdir ] || mkdir $tmpdir

#false && \
{
    cat $srclm | find_arpa_oovs.pl words.txt  > $tmpdir/oovs_lm.txt

    # grep -v '<s> <s>' because the LM seems to have some strange and useless
    # stuff in it with multiple <s>'s in the history.
    # gunzip -c $srclm | \
    cat $srclm | \
      grep -v '<s> <s>' | \
      grep -v '</s> <s>' | \
      grep -v '</s> </s>' | \
      arpa2fst - | fstprint | \
      remove_oovs.pl $tmpdir/oovs_lm.txt | \
      eps2disambig.pl | s2eps.pl | fstcompile --isymbols=words.txt \
        --osymbols=words.txt  --keep_isymbols=false --keep_osymbols=false | \
       fstrmepsilon | fstarcsort --sort_type=ilabel > G.fst
}
    
    # Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
    # minimized.
    fsttablecompose L.fst G.fst | fstdeterminizestar --use-log=true | \
      fstminimizeencoded | fstarcsort --sort_type=ilabel > LG.fst || exit 1;
}

    fsttablecompose T.fst L.fst > TL.fst || exit 1;

    fsttablecompose T.fst LG.fst > TLG.fst || exit 1;

echo "Composing decoding graph TLG.fst succeeded"


