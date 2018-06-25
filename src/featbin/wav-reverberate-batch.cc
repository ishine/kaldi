// featbin/wav-reverberate-batch.cc

// Copyright 2015  Tom Ko

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "feat/signal.h"

namespace kaldi {

/*
   This function is to repeatedly concatenate signal1 by itself 
   to match the length of signal2 and add the two signals together.
*/
void AddVectorsOfUnequalLength(const VectorBase<BaseFloat> &signal1,
                                     Vector<BaseFloat> *signal2) {
  for (int32 po = 0; po < signal2->Dim(); po += signal1.Dim()) {
    int32 block_length = signal1.Dim();
    if (signal2->Dim() - po < block_length) block_length = signal2->Dim() - po;
    signal2->Range(po, block_length).AddVec(1.0, signal1.Range(0, block_length));
  }
}

void ElementwiseProductOfFft_local(const Vector<BaseFloat> &a, Vector<BaseFloat> *b) {
  int32 num_fft_bins = a.Dim() / 2;
  for (int32 i = 0; i < num_fft_bins; i++) {
    // do complex multiplication
    ComplexMul(a(2*i), a(2*i + 1), &((*b)(2*i)), &((*b)(2*i + 1)));
  }
}

void FFTbasedBlockConvolveSignals_KeepSampLength(const Vector<BaseFloat> &filter, Vector<BaseFloat> *signal) {
  int32 signal_length = signal->Dim();
  int32 filter_length = filter.Dim();
  // int32 output_length = signal_length + filter_length - 1;
  int32 output_length = signal_length;
  signal->Resize(output_length, kCopyData);

  KALDI_VLOG(1) << "Length of the filter is " << filter_length;

  int32 fft_length = RoundUpToNearestPowerOfTwo(4 * filter_length);
  KALDI_VLOG(1) << "Best FFT length is " << fft_length;

  int32 block_length = fft_length - filter_length + 1;
  KALDI_VLOG(1) << "Block size is " << block_length;
  SplitRadixRealFft<BaseFloat> srfft(fft_length);

  Vector<BaseFloat> filter_padded(fft_length);
  filter_padded.Range(0, filter_length).CopyFromVec(filter);
  srfft.Compute(filter_padded.Data(), true);

  Vector<BaseFloat> temp_pad(filter_length - 1);
  temp_pad.SetZero();
  Vector<BaseFloat> signal_block_padded(fft_length);

  for (int32 po = 0; po < output_length; po += block_length) {
    // get a block of the signal
    int32 process_length = std::min(block_length, output_length - po);
    signal_block_padded.SetZero();
    signal_block_padded.Range(0, process_length).CopyFromVec(signal->Range(po, process_length));

    srfft.Compute(signal_block_padded.Data(), true);

    ElementwiseProductOfFft_local(filter_padded, &signal_block_padded);

    srfft.Compute(signal_block_padded.Data(), false);
    signal_block_padded.Scale(1.0 / fft_length);

    // combine the block
    if (po + block_length < output_length) {       // current block is not the last block
      signal->Range(po, block_length).CopyFromVec(signal_block_padded.Range(0, block_length));
      signal->Range(po, filter_length - 1).AddVec(1.0, temp_pad);
      temp_pad.CopyFromVec(signal_block_padded.Range(block_length, filter_length - 1));
    } else {
      signal->Range(po, output_length - po).CopyFromVec(
                        signal_block_padded.Range(0, output_length - po));
      if (filter_length - 1 < output_length - po)
        signal->Range(po, filter_length - 1).AddVec(1.0, temp_pad);
      else
        signal->Range(po, output_length - po).AddVec(1.0, temp_pad.Range(0, output_length - po));
    }
  }
}


/*
   This function is to add signal1 to signal2 starting at the offset of signal2
   This will not extend the length of signal2.
*/
void AddVectorsWithOffset(const Vector<BaseFloat> &signal1, int32 offset,
                                             Vector<BaseFloat> *signal2) {
  int32 add_length = std::min(signal2->Dim() - offset, signal1.Dim());
  if (add_length > 0)
    signal2->Range(offset, add_length).AddVec(1.0, signal1.Range(0, add_length));
}



BaseFloat MaxAbsolute(const Vector<BaseFloat> &vector) {
  return std::max(std::abs(vector.Max()), std::abs(vector.Min()));
}

/* 
   Early reverberation component of the signal is composed of reflections 
   within 0.05 seconds of the direct path signal (assumed to be the peak of 
   the room impulse response). This function returns the energy in 
   this early reverberation component of the signal. 
   The input parameters to this function are the room impulse response, the signal
   and their sampling frequency respectively.
*/
BaseFloat ComputeEarlyReverbEnergy(const Vector<BaseFloat> &rir, const Vector<BaseFloat> &signal,
                                   BaseFloat samp_freq) {
  int32 peak_index = 0;
  rir.Max(&peak_index);
  KALDI_VLOG(1) << "peak index is " << peak_index;

  const float sec_before_peak = 0.001;
  const float sec_after_peak = 0.05;
  int32 early_rir_start_index = peak_index - sec_before_peak * samp_freq;
  int32 early_rir_end_index = peak_index + sec_after_peak * samp_freq;
  if (early_rir_start_index < 0) early_rir_start_index = 0;
  if (early_rir_end_index > rir.Dim()) early_rir_end_index = rir.Dim();

  int32 duration = early_rir_end_index - early_rir_start_index;
  Vector<BaseFloat> early_rir(rir.Range(early_rir_start_index, duration));
  Vector<BaseFloat> early_reverb(signal);
  FFTbasedBlockConvolveSignals(early_rir, &early_reverb);

  // compute the energy
  return VecVec(early_reverb, early_reverb) / early_reverb.Dim();
}

/*
   This is the core function to do reverberation on the given signal.
   The input parameters to this function are the room impulse response,
   the sampling frequency and the signal respectively.
   The length of the signal will be extended to (original signal length +
   rir length - 1) after the reverberation.
*/
float DoReverberation(const Vector<BaseFloat> &rir, BaseFloat samp_freq,
                        Vector<BaseFloat> *signal) {
  // float signal_power = ComputeEarlyReverbEnergy(rir, *signal, samp_freq);
  float signal_power = 0;
  FFTbasedBlockConvolveSignals_KeepSampLength(rir, signal);
  return signal_power;
}

/*
   The noise will be scaled before the addition
   to match the given signal-to-noise ratio (SNR).
*/
void AddNoise(Vector<BaseFloat> *noise, BaseFloat snr_db,
                BaseFloat noise_start_time, BaseFloat samp_freq,
                BaseFloat signal_power, Vector<BaseFloat> *signal) {
  float noise_power = VecVec(*noise, *noise) / noise->Dim();
  float scale_factor = sqrt(pow(10, -snr_db / 10) * signal_power / noise_power);
  noise->Scale(scale_factor);
  KALDI_VLOG(1) << "Noise signal is being scaled with " << scale_factor
                << " to generate output with SNR " << snr_db << "db\n";
  int32 offset = noise_start_time * samp_freq;
  AddVectorsWithOffset(*noise, offset, signal);
}


/*
   This function converts comma-spearted string into float vector.
*/
void ReadCommaSeparatedCommand(const std::string &s,
                                std::vector<BaseFloat> *v) {
  std::vector<std::string> split_string;
  SplitStringToVector(s, ",", true, &split_string);
  for (size_t i = 0; i < split_string.size(); i++) {
    float ret;
    ConvertStringToReal(split_string[i], &ret);
    v->push_back(ret);
  }
}


void LoadWavScp(const std::string &wav_rspecifier,
        std::vector<std::string> *wav_keys,
        std::vector<Vector<BaseFloat> > *wav_vector) {
  if (!wav_rspecifier.empty()) {
    SequentialTableReader<WaveHolder> *wav_reader =
      new SequentialTableReader<WaveHolder>(wav_rspecifier);
    for(; !wav_reader->Done(); wav_reader->Next()) {
      std::string wav_key = wav_reader->Key();
      WaveData wav = wav_reader->Value();
      // BaseFloat samp_freq = wav.SampFreq();
      int32 num_samp_input = wav.Data().NumCols();
      int32 num_input_channel = wav.Data().NumRows();

      KALDI_ASSERT(num_input_channel == 1);
      KALDI_ASSERT(num_samp_input > 1);

      // for (size_t i = 0; i < wav.Data().NumRows(); ++i) {
      //  Vector<BaseFloat> channnel_wav(wav.Data().Row(i));
      // }

      Vector<BaseFloat> channnel_wav(wav.Data().Row(0));
      wav_keys->push_back(wav_key);
      wav_vector->push_back(channnel_wav);
    }
    if (NULL != wav_reader) {
      delete wav_reader;
      wav_reader = NULL;
    }
  }
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Corrupts the wave files supplied via input pipe with the specified\n"
        "room-impulse response (rir_matrix) and additive noise distortions\n"
        "(specified by corresponding files).\n"
        "Usage:  wav-reverberate-batch [options...] <wav-in-rxfilename> "
        "<wav-out-wxfilename>\n"
        "e.g.\n"
        "wav-reverberate-batch --seed=seed_int --debug-file=ark,scp:debug.ark,debug.scp "
        "--impulse-response=scp:rir.scp --additive-signals=scp:noise.scp "
        "--snrs='20.0,10.0' scp:wav_input.scp ark,scp:out.ark,out.scp\n";

    ParseOptions po(usage);
    int32 seed = 0;
    std::string rir_file = "";
    std::string additive_signals = "";
    std::string snrs = "";
    std::string debug_out_file = "";


    bool multi_channel_output = false;
    // bool shift_output = true;
    int32 input_channel = 1;
    int32 rir_channel = 1;
    int32 noise_channel = 1;
    bool normalize_output = true;
    BaseFloat volume = 0;
    bool keep_key = false;

    po.Register("seed", &seed, "Seed for random number generator");
    po.Register("debug-file", &debug_out_file, "set the debug output wav file, will "
                "save every step wav for simu");
    po.Register("multi-channel-output", &multi_channel_output,
                "Specifies if the output should be multi-channel or not");
    po.Register("input-wave-channel", &input_channel,
                "Specifies the channel to be used from input as only a "
                "single channel will be used to generate reverberated output");
    po.Register("rir-channel", &rir_channel,
                "Specifies the channel of the room impulse response, "
                "it will only be used when multi-channel-output is false");
    po.Register("noise-channel", &noise_channel,
                "Specifies the channel of the noise file, "
                "it will only be used when multi-channel-output is false");
    po.Register("impulse-response", &rir_file,
                "File with the impulse response for reverberating the input wave"
                "It can be either a file in wav format or a piped command. "
                "E.g. --impulse-response='rir.wav' or 'sox rir.wav - |' ");
    po.Register("additive-signals", &additive_signals,
                "A comma separated list of additive signals. "
                "They can be either filenames or piped commands. "
                "E.g. --additive-signals='noise1.wav,noise2.wav' or "
                "'sox noise1.wav - |,sox noise2.wav - |'. "
                "Requires --snrs and --start-times.");
    po.Register("snrs", &snrs,
                "A comma separated list of SNRs(dB). "
                "The additive signals will be scaled according to these SNRs. "
                "E.g. --snrs='20.0,0.0,5.0,10.0' ");
    po.Register("normalize-output", &normalize_output,
                "If true, then after reverberating and "
                "possibly adding noise, scale so that the signal "
                "energy is the same as the original input signal. "
                "See also the --volume option.");
    po.Register("volume", &volume,
                "If nonzero, a scaling factor on the signal that is applied "
                "after reverberating and possibly adding noise. "
                "If you set this option to a nonzero value, it will be as "
                "if you had also specified --normalize-output=false.");
    po.Register("keep-key", &keep_key,
                "If true, the output-wav-key is same as input-wav-key "
                "If false, the output-wav-key will add snr* suffix.");

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    if (multi_channel_output) {
      if (rir_channel != 0 || noise_channel != 0)
        KALDI_WARN << "options for --rir-channel and --noise-channel"
                      "are ignored as --multi-channel-output is true.";
    }

    std::string input_wave_file = po.GetArg(1);
    std::string output_wave_file = po.GetArg(2);

    KALDI_ASSERT(multi_channel_output == false);
    KALDI_ASSERT(input_channel == 1);
    KALDI_ASSERT(rir_channel == 1);
    KALDI_ASSERT(noise_channel == 1);

    std::vector<std::string> addnoise_key_vector;
    std::vector<Vector<BaseFloat> > addnoise_data_vector;
    addnoise_key_vector.clear();
    addnoise_data_vector.clear();

    std::vector<std::string> rir_key_vector;
    std::vector<Vector<BaseFloat> > rir_data_vector;
    rir_key_vector.clear();
    rir_data_vector.clear();

    std::vector<BaseFloat> snr_vector;
    if (!snrs.empty()) {
      ReadCommaSeparatedCommand(snrs, &snr_vector);
    }
    // KALDI_ASSERT(snr_vector.size() >= 1);

    std::srand(seed);
    kaldi::RandomState rand_state;
    rand_state.seed = seed;

    LoadWavScp(additive_signals, &addnoise_key_vector, &addnoise_data_vector);
    LoadWavScp(rir_file, &rir_key_vector, &rir_data_vector);
    KALDI_ASSERT((!additive_signals.empty()) || (!rir_file.empty()));
    KALDI_VLOG(1) << "# addnoise noise num : " << addnoise_key_vector.size()
                  << "# rir num :" << rir_key_vector.size();


    SequentialTableReader<WaveHolder> *input_wav_reader =
      new SequentialTableReader<WaveHolder>(input_wave_file);
    TableWriter<WaveHolder> *out_wav_write =
      new TableWriter<WaveHolder>(output_wave_file);

    TableWriter<WaveHolder> *debug_out_wav_write = NULL;
    if (debug_out_file != "") {
      debug_out_wav_write = new TableWriter<WaveHolder>(debug_out_file);
    }


    for(; !input_wav_reader->Done(); input_wav_reader->Next()) {
      std::string wav_key = input_wav_reader->Key();
      WaveData input_wav = input_wav_reader->Value();
      BaseFloat samp_freq = input_wav.SampFreq();
      int32 num_samp_input = input_wav.Data().NumCols();
      int32 num_input_channel = input_wav.Data().NumRows();
      KALDI_VLOG(1) << "load wav file " << wav_key
                    << ", #sampling frequency: " << samp_freq
                    << ", #samples: " << num_samp_input
                    << ", #channel: " << num_input_channel;

      BaseFloat samp_freq_rir = samp_freq;

      KALDI_ASSERT(num_input_channel == 1);
      KALDI_ASSERT(num_samp_input > 1);

      // int32 shift_index = 0;
      // int32 num_output_channels = (multi_channel_output ? num_rir_channel : 1);
      int32 num_output_channels = 1;
      // int32 num_samp_output = num_samp_input;

      Matrix<BaseFloat> input_mat_wav = input_wav.Data();
      Matrix<BaseFloat> out_mat_wav(num_input_channel, num_samp_input);

      // get expect snr
      BaseFloat exp_snr_db = -1000;
      std::string out_wav_key = wav_key;
      if (0 != snr_vector.size()) {
        int32 snr_random_index = kaldi::RandInt(0, snr_vector.size() - 1, &rand_state);
        exp_snr_db = snr_vector[snr_random_index];
        char str_snr[128];
        memset(str_snr, 0 ,128);
        sprintf(str_snr, "%d", static_cast<int>(exp_snr_db));
        out_wav_key = keep_key ? wav_key : wav_key + "_noise" +
                                  std::string(str_snr);
        KALDI_VLOG(1) << "# snr_random_index :" << snr_random_index
                      << ", # expect snr: " << exp_snr_db;
      } else {
        out_wav_key = wav_key;
        KALDI_VLOG(1) << "only add the noise, not control the snr";
      }

      // get random rir
      Vector<BaseFloat> *local_rir_vec = NULL;
      if (rir_data_vector.size() != 0) {
        int32 rir_random_index = kaldi::RandInt(0, rir_data_vector.size() - 1, &rand_state);
        KALDI_VLOG(1) << "# rir_random_index : " << rir_random_index
                      << ", #rir key:" << rir_key_vector[rir_random_index]
                      << ", #rir sample num:" << rir_data_vector[rir_random_index].Dim();
        local_rir_vec = new Vector<BaseFloat>(rir_data_vector[rir_random_index]);
      }

      // get random noise
      Vector<BaseFloat> *local_addnoise_vec = NULL;
      if (addnoise_data_vector.size() != 0) {
        int32 addnoise_random_index = kaldi::RandInt(0, addnoise_data_vector.size() - 1, &rand_state);
        KALDI_VLOG(1) << "# addnoise_random_index : " << addnoise_random_index
                      << ", #addnoise key:" << addnoise_key_vector[addnoise_random_index]
                      << ", #noise sample num:" << addnoise_data_vector[addnoise_random_index].Dim();
        local_addnoise_vec = new Vector<BaseFloat>(addnoise_data_vector[addnoise_random_index]);
      }

      // do rir and add noise
      for (int32 output_channel = 0; output_channel < num_output_channels; output_channel++) {
        Vector<BaseFloat> input(num_samp_input);
        input.CopyRowFromMat(input_mat_wav, output_channel);
        float power_before_reverb = VecVec(input, input) / input.Dim();

        if(NULL != debug_out_wav_write) {
          WaveData tmp_input_wave(samp_freq, input_mat_wav);
          debug_out_wav_write->Write("input_" + wav_key, tmp_input_wave);
        }

        // int32 this_rir_channel = (multi_channel_output ? output_channel : rir_channel);
        // int32 this_rir_channel = 0;

        float power_after_reverb = power_before_reverb;
        if (NULL != local_rir_vec) {
          Vector<BaseFloat> rir;
          rir.Resize(local_rir_vec->Dim());
          rir.CopyFromVec(*local_rir_vec);
          rir.Scale(1.0 / (1 << 15));
          DoReverberation(rir, samp_freq_rir, &input);
          KALDI_VLOG(1) << "cur rir dim: " << rir.Dim()
                        << ", #after_rir.dim: " << input.Dim();

          if(NULL != debug_out_wav_write) {
            Matrix<BaseFloat> tmp_rir_mat(1, rir.Dim());
            tmp_rir_mat.CopyRowFromVec(rir, 0);
            WaveData tmp_rir_wave(samp_freq, tmp_rir_mat);
            debug_out_wav_write->Write("rir_" + wav_key, tmp_rir_wave);

            Matrix<BaseFloat> tmp_afterrir_mat(1, input.Dim());
            tmp_afterrir_mat.CopyRowFromVec(input, 0);
            WaveData tmp_afterrir_wave(samp_freq, tmp_afterrir_mat);
            debug_out_wav_write->Write("afterrir_" + wav_key, tmp_afterrir_wave);
          }
          power_after_reverb = VecVec(input, input) / input.Dim();
          input.Scale(sqrt(power_before_reverb / power_after_reverb));
          power_after_reverb = VecVec(input, input) / input.Dim();
          if(NULL != debug_out_wav_write) {
            Matrix<BaseFloat> tmp_afterrirnorm_mat(1, input.Dim());
            tmp_afterrirnorm_mat.CopyRowFromVec(input, 0);
            WaveData tmp_afterrirnorm_wave(samp_freq, tmp_afterrirnorm_mat);
            debug_out_wav_write->Write("afterrirnorm_" + wav_key, tmp_afterrirnorm_wave);
          }
        }


        if (NULL != local_addnoise_vec) {
          Vector<BaseFloat> noise(0);
          noise.Resize(input.Dim());
          int32 noise_start_offset = 0;
          // int32 this_noise_channel = (multi_channel_output ? output_channel : noise_channel);
          // int32 this_noise_channel = 0;
          if (local_addnoise_vec->Dim() <= input.Dim()) {
            // noise size < input wav size
            noise_start_offset = kaldi::RandInt(0,
              local_addnoise_vec->Dim() - 1, &rand_state);
            BaseFloat *p_noise_data = local_addnoise_vec->Data();
            for (int32 index_out = 0; index_out < input.Dim(); ++index_out) {
              int32 org_index = (noise_start_offset + index_out) %
                local_addnoise_vec->Dim();
              noise(index_out) = *(p_noise_data + org_index);
            }
          } else {
            // normal mode; random select a start offset
            noise_start_offset = kaldi::RandInt(0,
              local_addnoise_vec->Dim() - input.Dim(), &rand_state);
            SubVector<BaseFloat> subvector_noise = local_addnoise_vec->Range(
                            noise_start_offset, input.Dim());
            noise.CopyFromVec(subvector_noise);
          }
          KALDI_VLOG(1) << "# noise start samp offset : " << noise_start_offset;
          if(NULL != debug_out_wav_write) {
            Matrix<BaseFloat> tmp_noise_mat(1, noise.Dim());
            tmp_noise_mat.CopyRowFromVec(noise, 0);
            WaveData tmp_noise_wave(samp_freq, tmp_noise_mat);
            debug_out_wav_write->Write("noise_" + wav_key, tmp_noise_wave);
          }

          if (0 != snr_vector.size()) {
            AddNoise(&noise, exp_snr_db, 0,
                    samp_freq, power_after_reverb, &input);
          } else {
            input.AddVec(1.0f, noise);
          }
          if(NULL != debug_out_wav_write) {
            Matrix<BaseFloat> tmp_afternoise_mat(1, input.Dim());
            tmp_afternoise_mat.CopyRowFromVec(input, 0);
            WaveData tmp_afternoise_wave(samp_freq, tmp_afternoise_mat);
            debug_out_wav_write->Write("afternoise_" + wav_key, tmp_afternoise_wave);
          }
        }

        float power_after_addnoise = VecVec(input, input) / input.Dim();

        if (volume > 0)
          input.Scale(volume);
        else if (normalize_output)
          input.Scale(sqrt(power_before_reverb / power_after_addnoise));
        float power_out = VecVec(input, input) / input.Dim();
        KALDI_VLOG(1) << "# early_energy : " << power_before_reverb
                      << ", # after_reverb energy:" << power_after_reverb
                      << ", # after add noise energy:" << power_after_addnoise
                      << ", # output energy: " << power_out;

        if((normalize_output) && (NULL != debug_out_wav_write)) {
          Matrix<BaseFloat> tmp_afternorm_mat(1, input.Dim());
          tmp_afternorm_mat.CopyRowFromVec(input, 0);
          WaveData tmp_afternorm_wave(samp_freq, tmp_afternorm_mat);
          debug_out_wav_write->Write("afternorm_" + wav_key, tmp_afternorm_wave);
        }

        out_mat_wav.CopyRowFromVec(input, 0);
        /*if (num_samp_output <= num_samp_input) {
          // trim the signal from the start
          // out_mat_wav.CopyRowFromVec(input.Range(shift_index, num_samp_output), output_channel);
          out_mat_wav.CopyRowFromVec(input, 0);
        } else {
          // repeat the signal to fill up the duration
          Vector<BaseFloat> extended_input(num_samp_output);
          extended_input.SetZero();
          AddVectorsOfUnequalLength(input.Range(shift_index, num_samp_input), &extended_input);
          out_mat_wav.CopyRowFromVec(extended_input, output_channel);
        }*/
      }

      if (NULL != local_rir_vec) {
        delete local_rir_vec;
        local_rir_vec = NULL;
      }
      if (NULL != local_addnoise_vec) {
        delete local_addnoise_vec;
        local_addnoise_vec = NULL;
      }

      WaveData out_wave(samp_freq, out_mat_wav);
      out_wav_write->Write(out_wav_key, out_wave);
    }

    if (NULL != input_wav_reader) {
      delete input_wav_reader;
      input_wav_reader = NULL;
    }
    if (NULL != out_wav_write) {
      delete out_wav_write;
      out_wav_write = NULL;
    }
    if(NULL != debug_out_wav_write) {
      delete debug_out_wav_write;
      debug_out_wav_write = NULL;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

