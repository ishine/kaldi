set -e

# configs for 'chain'
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

stage=3
data=data/wav_fa

if [ $stage -le 1 ]; then
  # first make fbank features 
  # modify conf/fbank.conf to set fbank feature config
  for x in wav_fa; do
    steps/make_fbank.sh --nj 1 --cmd "$train_cmd" \
      data/$x data/$x/make_fbank/$x data/$x/fbank
    steps/compute_cmvn_stats.sh data/$x data/$x/make_fbank/$x data/$x/fbank
    utils/fix_data_dir.sh data/$x
  done
fi

if [ $stage -le 2 ]; then
  # Get the alignments as lattices
  steps/nnet3/align.sh --nj 1 --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' --cmd "$train_cmd" $data \
    data/lang exp/chain/lstm_6j_8job_ld5 exp/lstmCM_2000h_fb40_ali
fi

if [ $stage -le 3 ]; then
  # Get the alignments as lattices
  steps/ali_to_phone.sh --nj 1 --cmd "$train_cmd" \
    exp/lstmCM_2000h_fb40_ali exp/lstmCM_2000h_fb40_ali
fi
exit 0;
