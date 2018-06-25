stage=-1
wav_copy=/home/yjcelly/local_aishell_env/kaldi/src/featbin/wav-copy
#revert_bat_tool=/home/yjcelly/local_aishell_env/kaldi/src/featbin/wav-reverberate-batch
revert_bat_tool=/home/yjcelly/local_aishell_env/jiayu_kaldi/kaldi/src/featbin/wav-reverberate-batch
rir_wav_scp=rir_wav.scp
noise_wav_scp=noise_wav.scp
input_wav_scp=input_wav.scp


func_out_debug() {
  out_dir=$1
  #rm -rf ${out_dir}
  mkdir -p  ${out_dir}
  cat debug.scp  | awk -v awk_wav_copy=${wav_copy} -v awk_out_dir=${out_dir} \
    '{printf("%s %s %s/%s.wav\n",awk_wav_copy, $2, awk_out_dir, $1);}'  | sh
  cat  out.scp  | awk  -v awk_wav_copy=${wav_copy} -v awk_out_dir=${out_dir} \
    '{printf("%s %s %s/finalout_%s.wav\n", awk_wav_copy,$2,awk_out_dir,$1);}'  | sh
}

if [ $stage -le 0 ]; then
###rir + snr + add noise
  echo "demo for rir+snr+add noise"
  out_dir=out_dir_rir_snr_addnoise
  rm -rf $out_dir
#valgrind --tool=memcheck --leak-check=full  --show-leak-kinds=all -v \
     ${revert_bat_tool}    \
	 --debug-file=ark,scp:debug.ark,debug.scp \
	 --impulse-response=scp:${rir_wav_scp}  --additive-signals=scp:${noise_wav_scp}    --snrs='-20.0,-10.0'   \
 	 scp:${input_wav_scp}  ark,scp:out.ark,out.scp 
  func_out_debug $out_dir
fi

if [ $stage -le 1 ]; then
  echo "demo for rir+snr"
  out_dir=out_dir_rir_snr
  rm -rf $out_dir
       ${revert_bat_tool}    \
         --debug-file=ark,scp:debug.ark,debug.scp \
         --impulse-response=scp:${rir_wav_scp}    --snrs='-20.0,-10.0'   \
         scp:${input_wav_scp}  ark,scp:out.ark,out.scp
  func_out_debug $out_dir
fi

if [ $stage -le 2 ]; then
  echo "demo for snr+add noise"
  out_dir=out_dir_rir_noise
  rm -rf $out_dir
       ${revert_bat_tool}    \
         --debug-file=ark,scp:debug.ark,debug.scp \
         --snrs='-20.0,-10.0'  --additive-signals=scp:${noise_wav_scp}  \
         scp:${input_wav_scp}  ark,scp:out.ark,out.scp
  func_out_debug $out_dir
fi

if [ $stage -le 3 ]; then
  echo "demo for add noise"
  out_dir=out_dir_noise
  rm -rf $out_dir
       ${revert_bat_tool}    \
         --debug-file=ark,scp:debug.ark,debug.scp \
         --additive-signals=scp:${noise_wav_scp}  \
         scp:${input_wav_scp}  ark,scp:out.ark,out.scp
  func_out_debug $out_dir
fi

if [ $stage -le 4 ]; then
  echo "demo for not output norm"
  out_dir=out_dir_nonorm
  rm -rf $out_dir
       ${revert_bat_tool}  --normalize-output=false  \
         --debug-file=ark,scp:debug.ark,debug.scp \
         --additive-signals=scp:${noise_wav_scp}  \
         scp:${input_wav_scp}  ark,scp:out.ark,out.scp
  func_out_debug $out_dir
fi

if [ $stage -le 5 ]; then
  echo "demo for not debug"
  out_dir=out_dir_nodebug
  rm -rf $out_dir
  mkdir -p $out_dir
       ${revert_bat_tool}  --normalize-output=false  \
         --additive-signals=scp:${noise_wav_scp}  \
         scp:${input_wav_scp}  ark,scp:out.ark,out.scp
  cat  out.scp  | awk  -v awk_wav_copy=${wav_copy} -v awk_out_dir=${out_dir} \
    '{printf("%s %s %s/finalout_%s.wav\n", awk_wav_copy,$2,awk_out_dir,$1);}'  | sh
fi

