#!/bin/bash

# Copyright 2017 (author: Kaituo Xu)

echo "$0 $@"

nohup time bash local/nnet1/train_once_sparse_dnn.sh --prune-ratio "0.9,0.9,0.9,0.9,0.9,0" exp/sparse_dnn.9 exp/sparse_dnn.9_2 &
# local/nnet1/train_once_sparse_dnn.sh --prune-ratio "0.9,0.9,0.9,0.9,0.9,0" exp/sparse_dnn.9_2 exp/sparse_dnn.9_3

nohup time bash local/nnet1/train_once_sparse_dnn.sh --prune-ratio "0.8,0.8,0.8,0.8,0.8,0" exp/sparse_dnn.8 exp/sparse_dnn.8_2 &

nohup time bash local/nnet1/train_once_sparse_dnn.sh --prune-ratio "0.7,0.7,0.7,0.7,0.7,0" exp/sparse_dnn.7 exp/sparse_dnn.7_2 &

nohup time bash local/nnet1/train_once_sparse_dnn.sh --prune-ratio "0.6,0.6,0.6,0.6,0.6,0" exp/sparse_dnn.6 exp/sparse_dnn.6_2 &

nohup time bash local/nnet1/train_once_sparse_dnn.sh --prune-ratio "0.5,0.5,0.5,0.5,0.5,0" exp/sparse_dnn.5 exp/sparse_dnn.5_2 &
