#!/bin/bash

# Created on 2018-04-08
# Author: Kaituo Xu
# Function: Generate simulated test sets.

choice="both"  # both / noise / reverb

data_dir=test
foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"
num_reps=1

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

[ ! -f rirs_noises.zip ] && [ ! -d RIRS_NOISES ] && wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
[ ! -d RIRS_NOISES ] && unzip rirs_noises.zip

# Add Reverb and noise
if [ $choice == "both" ]; then
  python local/nnet1/data/reverberate_data_dir.py \
    --rir-set-parameters "0.3, RIRS_NOISES/simulated_rirs/smallroom/rir_list" \
    --rir-set-parameters "0.3, RIRS_NOISES/simulated_rirs/mediumroom/rir_list" \
    --rir-set-parameters "0.3, RIRS_NOISES/simulated_rirs/largeroom/rir_list" \
    --noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list \
    --prefix "rev_noise" \
    --foreground-snrs $foreground_snrs \
    --background-snrs $background_snrs \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 1 \
    --isotropic-noise-addition-probability 1 \
    --num-replications $num_reps \
    --max-noises-per-minute 1 \
    --source-sampling-rate 16000 \
    --random-seed 6666 \
    data/${data_dir} data/${data_dir}_rvb_noise
# Only add noise
elif [ $choice == "noise" ]; then
  python local/nnet1/data/reverberate_data_dir.py \
    --rir-set-parameters "0.3, RIRS_NOISES/simulated_rirs/smallroom/rir_list" \
    --noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list \
    --prefix "noise" \
    --foreground-snrs $foreground_snrs \
    --background-snrs $background_snrs \
    --speech-rvb-probability 0 \
    --pointsource-noise-addition-probability 1 \
    --isotropic-noise-addition-probability 0 \
    --num-replications $num_reps \
    --max-noises-per-minute 1 \
    --source-sampling-rate 16000 \
    --random-seed 6666 \
    data/${data_dir} data/${data_dir}_noise
  sed -i "s/ wav-reverberate --impulse.*start/\' --start/g" data/${data_dir}_noise/wav.scp
# Only add reverb
elif [ $choice == "reverb" ]; then
  python local/nnet1/data/reverberate_data_dir.py \
    --rir-set-parameters "0.3, RIRS_NOISES/simulated_rirs/smallroom/rir_list" \
    --rir-set-parameters "0.3, RIRS_NOISES/simulated_rirs/mediumroom/rir_list" \
    --rir-set-parameters "0.3, RIRS_NOISES/simulated_rirs/largeroom/rir_list" \
    --prefix "rev" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 1 \
    --num-replications $num_reps \
    --max-noises-per-minute 0 \
    --source-sampling-rate 16000 \
    --random-seed 6666 \
    data/${data_dir} data/${data_dir}_rvb
else
  echo "Unsupported choice. exit.";
  exit 1;
fi
