#!/usr/bin/perl -w

# Copyright 2008 Google Inc. All Rights Reserved.
# Author: fliu@google.com (Fang Liu)
#
# This program is a perl version of a simple Chinese segmentor
#
# The algorithm for this segmenter is to search all possible segmentation,
# and choose the one with highest frequency product.
#
#
# Usage:
#  perl segmenter.pl [vocabularyFile] < unsegmentedFile > segmentedFile
#
#  if no vocabularyFileFile is given, will try to get the "vocab.txt" in
#  current directory.

use Encode;
use strict;
use utf8;

my %vocab;  # the vocabulary, which is word=>frequency mapping
my $max_word_len = 0;  # maximal length of the word among the vocabulary

# load the vocabulary file.
# The format of the vocabulary is:
# <word_1><tab><freq>
# <word_2><tab><freq>
sub load_vocabulary {
  open VOC_FILE, $_[0];
  binmode VOC_FILE, ":utf8";
  while(<VOC_FILE>) {
    chomp;
    my @entry = split(/\t/, $_);
    my $word = $entry[0];
    my $count = $entry[1];
    $vocab{$word} = $count;
    my $word_len = length($word);
    if ($word_len > $max_word_len) {
      $max_word_len = $word_len;
    }
  }
  close VOC_FILE;
  print STDERR "finished loading vocabulary\n";
}

# segment the text on the text as parameter, return the segmented text
# separated by space.
sub do_segment {
  my $DEFAULT_WEIGHT = 40;
  my $text = Encode::decode_utf8($_[0]);
  if ($text eq "") {
    return "";
  }
  my $text_len = length($text);
  my @best_prob = {};
  $best_prob[0] = 0;
  my @prev = {};
  for (my $i = 0; $i < $text_len; $i++) {
    for (my $j = $i + 1; $j <= $text_len; $j++) {
      # exceeds the maximal length
      if ($j - $i > $max_word_len) {
        last;
      }
      # matching an entry in the vocabulary
      my $current_word = substr($text, $i, $j - $i);
      my $iseng = 0;
      if (!exists($vocab{$current_word})) {
         if($current_word=~/^[aA-zZ,']+$/){
           $iseng=1;
           # printf("%s\n",$current_word);
         }
         else {
           next;
         }
      } else {
         if($current_word=~/^[aA-zZ,']+$/){
           $iseng=1;
           # printf("%s\n",$current_word);
         }

      }

      # get the previous found path, if not exists, use the default value,
      # which means we may take the previous token as the path.
      my $prev_weight = 0;
      if ($i > 0) {
        if (exists($best_prob[$i])) {
          $prev_weight = $best_prob[$i];
        } else {
          $prev_weight = $DEFAULT_WEIGHT*$i;
        }
      }
      # calculate weight for curent path.
      my $w = $vocab{$current_word};
      if ($iseng == 1){
         $w = 0.01;
      } else {
         $w = $vocab{$current_word};
      }
      my $current_weight = $prev_weight + $w;
      # update the path
      if (!exists($prev[$j]) || $best_prob[$j] > $current_weight) {
        $prev[$j] = $i;
        $best_prob[$j] = $current_weight;
      }
    }
  }

  # get boundaries
  my @boundaries;
  for (my $i = $text_len; $i > 0;) {
    $boundaries[$i] = 1;
    if (exists($prev[$i])) {
      $i = $prev[$i];
    } else {
      $i--;
    }
  }

  #fill the result string
  my $result;
  my $prev = 0;
  for (my $i = 1; $i < @boundaries; $i++) {
    if (defined($boundaries[$i])) {
      my $current_word = substr($text, $prev, $i - $prev);
      $result .= $current_word;
      $result .= " ";
      $prev = $i;
    }
  }
  chomp($result);
  return $result;
}

# Print the usage string
sub usage {
  print STDERR "\n-----------------------------------------------------------------------";
  print STDERR "\nUsage:";
  print STDERR "\n\tperl segmenter.pl [vocabularyFile] < unSegmentedFile > segmentedFile";
  print STDERR "\n-----------------------------------------------------------------------\n";
}

# the entry point function
sub main {
  if (@ARGV == 1 && $ARGV[0] eq "-help") {
    usage();
    exit;
  }
  my $voca_file;
  if (@ARGV == 1) {
    $voca_file = $ARGV[0];
  } else {
    $voca_file = "vocab.txt";
  }
  load_vocabulary($voca_file);

  $| = 1;
  binmode STDOUT, ":utf8";
  while(<STDIN>) {
    chomp;
    my $result = do_segment($_);
    print "$result\n";
  }
}

main();
