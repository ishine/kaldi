#!/bin/bash

# Copyright 2015  Yajie Miao  (Carnegie Mellon University) 

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

# This script prepares the phoneme-based lexicon from the CMU dictionary. It also generates
# the list of lexicon units and represents the lexicon using the indices of the units. 

echo "$0 $@"  # Print the command line for logging


if [ $# != 2 ]; then
  echo "Usage: local/chn_prepare_phn_dict.sh dict <dict-dir>"
  echo "e.g.: local/chn_prepare_phn_dict.sh src.dict data/dict_phn"
  exit 1;
fi

dir=$2
mkdir -p $dir

[ -f path.sh ] && . ./path.sh

srcdict=$1

[ ! -r $srcdict ] && echo "Missing $srcdict" && exit 1

# Join dicts and fix some troubles
cat $srcdict | grep -v "<s>" | grep -v "</s>" | grep -v "!SIL" | LANG= LC_ALL= sort > $dir/lexicon1_raw_nosil.txt || exit 1;

# We keep only one pronunciation for each word. Other alternative pronunciations are discarded.
cat $dir/lexicon1_raw_nosil.txt | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
  > $dir/lexicon2_raw_nosil.txt || exit 1;

# Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon2_raw_nosil.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt

cat  $dir/lexicon2_raw_nosil.txt | sort | uniq > $dir/lexicon.txt || exit 1;

# The complete set of lexicon units, indexed by numbers starting from 1
awk '{print $1 " " NR}' $dir/units_nosil.txt > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Phoneme-based dictionary preparation succeeded"

