// cudamatrix/cu-device.cc

// Copyright 2009-2012  Karel Vesely
//                2013  Lucas Ondel
//           2013-2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Guoguo Chen

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



#if HAVE_CUDA == 1
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <algorithm>
#ifndef _MSC_VER
#include <dlfcn.h>
#endif

#include "cudamatrix/cu-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-matrix.h"
#include "base/kaldi-error.h"
#include "base/kaldi-utils.h"
#include "util/common-utils.h"
#include "util/kaldi-io.h"
// the following is for cuda_legacy_noop().
#include "cudamatrix/cu-kernels-ansi.h"

namespace kaldi {

/// This function attempts to get a CUDA device context on some available device
/// by doing 'cudaFree(0)'.  If it succeeds it returns true; if it fails, it
/// outputs some debugging information into 'debug_str' and returns false.
static bool GetCudaContext(int32 num_gpus, std::string *debug_str) {
  // Our first attempt to get a device context is: we do cudaFree(0) and see if
  // that returns no error code.  If it succeeds then we have a device
  // context.  Apparently this is the canonical way to get a context.
  if (cudaFree(0) == 0) {
    cudaGetLastError();  // Clear any error status.
    return true;
  }

  // The rest of this code represents how we used to get a device context, but
  // now its purpose is mainly a debugging one.
  std::ostringstream debug_stream;
  debug_stream << "num-gpus=" << num_gpus << ". ";
  for (int32 device = 0; device < num_gpus; device++) {
    cudaSetDevice(device);
    cudaError_t e = cudaFree(0);  // CUDA context gets created here.
    if (e == cudaSuccess) {
      if (debug_str)
        *debug_str = debug_stream.str();
      cudaGetLastError();  // Make sure the error state doesn't get returned in
                           // the next cudaGetLastError().
      return true;
    }
    debug_stream << "Device " << device << ": " << cudaGetErrorString(e) << ".  ";
  }
  if (debug_str)
    *debug_str = debug_stream.str();
  return false;
}


void CuDevice::InitializeLocal() {
  // This function may be called in the following two situations:
  //
  // (1) in the main thread, only when a GPU is not currently being used, either
  // within a call like CuDevice()::Instantiate().SelectGpuId(..)
  // (where the Instantiate() call will call Initialize() before SelectGpuId()
  // is called, just because of how Instantiate() works), or in a call
  // to 'CuDevice::Instantiate().Enabled()'.  In this case it will just
  // set initialized_ to true and notice that device_id_ == 1, and do nothing.
  //
  // (2) in threads created by the user, as soon as someone calls something that
  //   might potentially use the GPU, via CuDevice()::Instantiate().
  //   If device_id_ is >= 0, this will create the cuBLAS and cuSparse handles.
  KALDI_ASSERT(!initialized_);
  initialized_ = true;
  if (device_id_ == -1) {
    // There is nothing to do; we are not using a GPU.
    return;
  } else {
    if (!multi_threaded_) {
      multi_threaded_ = true;
      KALDI_WARN << "For multi-threaded code that might use GPU, you should call "
          "CuDevice::Instantiate().AllowMultithreading() at the start of "
          "the program.";
    }
    device_id_copy_ = device_id_;
    cudaSetDevice(device_id_);
    // Initialize CUBLAS.
    CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_));
    CUBLAS_SAFE_CALL(cublasSetStream(cublas_handle_, cudaStreamPerThread));
    // Initialize the cuSPARSE library
    CUSPARSE_SAFE_CALL(cusparseCreate(&cusparse_handle_));
    CUSPARSE_SAFE_CALL(cusparseSetStream(cusparse_handle_, cudaStreamPerThread));
  }
}

void CuDevice::SelectGpuId(std::string use_gpu) {
  if (device_id_ != -1) {
    KALDI_ERR << "You cannot call SelectGpuId twice if, on the first time, "
        "you requested a GPU.";
  }
  if (use_gpu != "yes" && use_gpu != "no" && use_gpu != "optional" && use_gpu != "wait") {
    KALDI_ERR << "Please choose : --use-gpu=yes|no|optional|wait, passed '" << use_gpu << "'";
  }
  if (use_gpu == "no") {
    KALDI_LOG << "Manually selected to compute on CPU.";
    return;
  }
  // Check that we have a gpu available
  int32 num_gpus = 0;

  cudaError_t e = cudaGetDeviceCount(&num_gpus);

  // Make sure the global allocator object has the up-to-date options.
  g_cuda_allocator.SetOptions(g_allocator_options);

  if (num_gpus == 0) {
    if (use_gpu == "yes" || use_gpu == "wait") {
      KALDI_CUDA_ERR(e, "No CUDA GPU detected!");
    }
    if (use_gpu == "optional") {
      KALDI_WARN << "No CUDA GPU detected; running on CPU since --use-gpu=optional specified.";
      return;
    }
  }

  // Create a CUDA context.
  std::string debug_str;
  bool got_context = GetCudaContext(num_gpus, &debug_str);

  if (use_gpu != "wait") {
    if (!got_context) {
      // So far no we don't have context, sleep a bit and retry.
      int32 sec_sleep = (use_gpu == "yes" ? 20 : 2);
      KALDI_WARN << "Will try again to get a GPU after " << sec_sleep
                 << " seconds.";
      Sleep(sec_sleep);
      if (!GetCudaContext(num_gpus, &debug_str)) {
        if (use_gpu == "yes") {
          {
            Input input;
            input.Open("nvidia-smi 1>&2 |");
          }
          KALDI_LOG << debug_str;
          KALDI_ERR << "Failed to create CUDA context, no more unused GPUs? ";
        }
        if (use_gpu == "optional") {
          KALDI_WARN << "Running on CPU!!! No more unused CUDA GPUs?";
          return;
        }
      }
    }
  } else {
    int32 num_times = 0;
    BaseFloat wait_time = 0.0;
    while (!got_context) {
      int32 sec_sleep = 5;
      if (num_times == 0)
        KALDI_WARN << "Will try again indefinitely every " << sec_sleep
                   << " seconds to get a GPU.";
      num_times++;
      wait_time += sec_sleep;
      Sleep(sec_sleep);
      got_context = GetCudaContext(num_gpus, NULL);
    }

    KALDI_WARN << "Waited " << wait_time
               << " seconds before creating CUDA context";
  }

  // Double check that we have the context
  KALDI_ASSERT(cudaSuccess == cudaDeviceSynchronize());

  // Check if the machine use compute exclusive mode
  if (IsComputeExclusive()) {
    KALDI_LOG << "CUDA setup operating under Compute Exclusive Mode.";
    FinalizeActiveGpu();
    return;
  } else {
    // Suggest to use compute exclusive mode
    KALDI_WARN << "Not in compute-exclusive mode.  Suggestion: use "
        "'nvidia-smi -c 3' to set compute exclusive mode";
    // We want to choose the device more carefully, so release the CUDA context.
    e = cudaDeviceReset();
    if (e != cudaSuccess) {
      KALDI_CUDA_ERR(e, "Failed to release CUDA context on a GPU");
    }

    // And select the GPU according to proportion of free memory
    if (SelectGpuIdAuto()) {
      FinalizeActiveGpu();
      return;
    } else {
      // We could not get a GPU the second time, after prevously having the CUDA
      // context.  Strange but not impossible.
      if (use_gpu == "yes") {
        KALDI_ERR << "Error acquiring GPU.";
      }
      if (use_gpu == "optional") {
        KALDI_WARN << "Running on CPU!!! Error acquiring GPU.";
        return;
      }
    }
  }
}

void CuDevice::SetGpuId(int n) {
  cudaError_t e = cudaSetDevice(n);
  if (e == cudaSuccess) {
      char name[128];
      DeviceGetName(name,128,n);
      int64 free, total;
      std::string mem_stats;
      mem_stats = GetFreeMemory(&free, &total);
      KALDI_LOG << "cudaSetDevice(" << n << "): "
                        << name << "\t" << mem_stats;
      FinalizeActiveGpu();
  }
  else{
      KALDI_WARN << "Cannot select this device: return code " << e
                  << ", Error message: \"" << cudaGetErrorString(e) << "\"";
  }
}

void CuDevice::FinalizeActiveGpu() {
  // The device at this point should have an active GPU, so we can query its
  // name and memory stats and notify user which GPU is being used.

  // Get the device-id of the active device.
  {
    int device_id;
    cudaError_t e = cudaGetDevice(&device_id);
    if (e != cudaSuccess) {
      KALDI_CUDA_ERR(e, "Failed to get device-id of active device.");
    }
    device_id_ = device_id;
    device_id_copy_ = device_id;
    initialized_ = true;  // Prevent Initialize() from being called on this,
                          // the main thread.
    // Initialize CUBLAS.
    CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_));
    CUBLAS_SAFE_CALL(cublasSetStream(cublas_handle_, cudaStreamPerThread));
    // Initialize the cuSPARSE library
    CUSPARSE_SAFE_CALL(cusparseCreate(&cusparse_handle_));
    CUSPARSE_SAFE_CALL(cusparseSetStream(cusparse_handle_, cudaStreamPerThread));

    // Notify the user which GPU is being userd.
    char name[128];
    DeviceGetName(name,128, device_id);

    CU_SAFE_CALL(cudaGetDeviceProperties(&properties_, device_id));

    KALDI_LOG << "The active GPU is [" << device_id << "]: " << name << "\t"
              << GetFreeGpuMemory(&free_memory_at_startup_, NULL) << " version "
              << properties_.major << "." << properties_.minor;
  }
  return;
}

bool CuDevice::DoublePrecisionSupported() {
  if (!Enabled()) return true;
  return properties_.major > 1 || (properties_.major == 1 && properties_.minor >= 3);
  // Double precision is supported from version 1.3
}


bool CuDevice::IsComputeExclusive() {
  // assume we already have an CUDA context created
  KALDI_ASSERT(cudaSuccess == cudaDeviceSynchronize());

  // get the device-id and its device-properties
  int gpu_id = -1;
  cudaError_t e = cudaGetDevice(&gpu_id);
  if (e != cudaSuccess) {
    KALDI_CUDA_ERR(e, "Failed to get current device");
  }
  struct cudaDeviceProp gpu_prop;
  e = cudaGetDeviceProperties(&gpu_prop, gpu_id);
  if (e != cudaSuccess) {
    KALDI_CUDA_ERR(e,  "Failed to get device properties");
  }
  // find out whether compute exclusive mode is used
  switch (gpu_prop.computeMode) {
    case cudaComputeModeExclusive :
      return true;
      break;
    case cudaComputeModeExclusiveProcess :
      return true;
      break;
    default :
      // in this case we release the GPU context...
      return false;
  }
}

template<typename TA, typename TB>
bool greater_pair(const std::pair<TA, TB> &left, const std::pair<TA, TB>& right) {
  return left.second > right.second;
}

bool CuDevice::SelectGpuIdAuto() {
  // Check that we have at least one gpu
  int32 num_gpus = 0;
  cudaError_t e = cudaGetDeviceCount(&num_gpus);
  if (num_gpus == 0) {
    KALDI_WARN << "No CUDA devices found";
    if (e != cudaSuccess) {
      KALDI_WARN << "cudaGetDeviceCount() returned " << e
                 <<", meaning: \"" << cudaGetErrorString(e)  << "\"";
    }
    return false;
  }

  // The GPU is selected according to maximal free memory ratio
  std::vector< std::pair<int, float> > free_mem_ratio(num_gpus);

  // Get ratios of memory use, if possible
  KALDI_LOG << "Selecting from " << num_gpus << " GPUs";
  for(int32 n = 0; n < num_gpus; n++) {
    int32 ret = cudaSetDevice(n);
    switch(ret) {
      case cudaSuccess : {
        // create the CUDA context for the thread
        cudaDeviceSynchronize();
        // get GPU name
        char name[128];
        DeviceGetName(name,128,n);
        // get GPU memory stats
        int64 free, total;
        std::string mem_stats;
        mem_stats = GetFreeGpuMemory(&free, &total);
        // log
        KALDI_LOG << "cudaSetDevice(" << n << "): "
                  << name << "\t" << mem_stats;

        // We have seen that in some cases GetFreeGpuMemory returns zero
        // That will produce nan after division, which might confuse
        // the sorting routine. Or maybe not, but let's keep it clean
        if (total <= 0) {
          KALDI_LOG << "Total memory reported for device " << n
                    << " is zero (or less).";
        }
        float mem_ratio = total > 0 ? free/(float)total : 0;
        free_mem_ratio[n] = std::make_pair(n, mem_ratio);

        // destroy the CUDA context for the thread
        cudaDeviceReset();
      } break;
      case cudaErrorDeviceAlreadyInUse :
        KALDI_LOG << "cudaSetDevice(" << n << "): "
                  << "Device cannot be accessed, used EXCLUSIVE-THREAD mode...";
        break;
      case cudaErrorInvalidDevice :
        KALDI_LOG << "cudaSetDevice(" << n << "): "
                  << "Device cannot be accessed, not a VALID CUDA device!";
        break;
      default :
        KALDI_LOG << "cudaSetDevice(" << n << "): "
                  << "returned " << ret << ", "
                  << cudaGetErrorString((cudaError_t)ret);
    }
  }
  // find GPU with max free memory
  int32 max_id=0;
  std::sort(free_mem_ratio.begin(), free_mem_ratio.end(),
            greater_pair<int, float>);
  // the free_mem_ratio should be bigger than zero
  KALDI_ASSERT(free_mem_ratio[max_id].second > 0.0);

  int dev_id;
  float mem_ratio;
  do {
    // try to select the GPU in the best to worst order
    // Note we have to check the return codes manually, as the CU_SAFE_CALL
    // contains call to KALDI_ERR (which will cause the program to abort)

    dev_id = free_mem_ratio[max_id].first;
    mem_ratio = free_mem_ratio[max_id].second;

    KALDI_LOG << "Trying to select device: " << dev_id << " (automatically), mem_ratio: " << mem_ratio;
    e = cudaSetDevice(dev_id);
    if (e != cudaSuccess) {
      KALDI_WARN << "Cannot select this device: return code " << e
                 << ", Error message: \"" << cudaGetErrorString(e) << "\"";
    } else {
      e = cudaDeviceSynchronize();
      if (e != cudaSuccess) {
        KALDI_WARN << "Cannot select this device: return code " << e
                   << ", Error message: \"" << cudaGetErrorString(e) << "\"";
      }
    }
    max_id++;
  } while ((e != cudaSuccess) && (max_id < free_mem_ratio.size()));

  if (e != cudaSuccess) {
    KALDI_WARN << "Failed to (automatically) select any device";
    return false;
  }
  KALDI_LOG << "Success selecting device " << dev_id << " free mem ratio: " << mem_ratio;
  return true;
}


void CuDevice::AccuProfile(const char *function_name,
                           const CuTimer &timer) {
  if (GetVerboseLevel() >= 2) {
    std::unique_lock<std::mutex> lock(profile_mutex_, std::defer_lock_t());
    if (multi_threaded_)
      lock.lock();
    std::string key(function_name);
    // by passing 0 as the stream to cudaStreamSynchronize, we are using the
    // per-thread default stream.  Since we compile with
    // -DCUDA_API_PER_THREAD_DEFAULT_STREAM, this equates to a per-thread
    // stream.
    cudaStreamSynchronize(0);
    double elapsed = timer.Elapsed();
    if (profile_map_.find(key) == profile_map_.end())
      profile_map_[key] = elapsed;
    else
      profile_map_[key] += elapsed;
  }
}

void CuDevice::PrintMemoryUsage() const {
  if (Enabled())
    g_cuda_allocator.PrintMemoryUsage();
}

void CuDevice::PrintProfile() {
  if (GetVerboseLevel() >= 1) {
    std::ostringstream os;
    os << "-----\n[cudevice profile]\n";
    unordered_map<std::string, double, StringHasher>::iterator it;
    std::vector<std::pair<double, std::string> > pairs;
    double total_time = 0.0;
    for(it = profile_map_.begin(); it != profile_map_.end(); ++it) {
      std::string function_name = it->first;
      double elapsed_time = it->second;
      total_time += elapsed_time;
      pairs.push_back(std::make_pair(elapsed_time, function_name));
    }
    // display from shortest to longest time, so tail will show the longest
    // times at the end.
    if (profile_map_.find("compute_ctc_loss") != profile_map_.end())
        os << "compute_ctc_loss" << "\t" << profile_map_["compute_ctc_loss"] << "s\n";
    std::sort(pairs.begin(), pairs.end());
    size_t max_print = 15, start_pos = (pairs.size() <= max_print ?
                                        0 : pairs.size() - max_print);
    for (size_t i = start_pos; i < pairs.size(); i++)
      os << pairs[i].second << "\t" << pairs[i].first << "s\n";
    os << "Total GPU time:\t" << total_time << "s (may involve some double-counting)\n";
    os << "-----";
    KALDI_LOG << os.str();
    PrintMemoryUsage();
  }
}


std::string CuDevice::GetFreeMemory(int64* free, int64* total) const {
  // WARNING! the CUDA API is inconsistent accross versions!
#ifdef _MSC_VER
  size_t mem_free, mem_total;
  cuMemGetInfo_v2(&mem_free, &mem_total);
#else
#if (CUDA_VERSION >= 3020)
  // define the function signature type
  size_t mem_free, mem_total;
#else
  unsigned int mem_free, mem_total;
#endif
  {
    // we will load cuMemGetInfo_v2 dynamically from libcuda.so
    // pre-fill ``safe'' values that will not cause problems
    mem_free = 1; mem_total = 1;
    // open libcuda.so
    void* libcuda = dlopen("libcuda.so",RTLD_LAZY);
    if (NULL == libcuda) {
      KALDI_WARN << "cannot open libcuda.so";
    } else {
      // define the function signature type
      // and get the symbol
#if (CUDA_VERSION >= 3020)
      typedef CUresult (*cu_fun_ptr)(size_t*, size_t*);
      cu_fun_ptr dl_cuMemGetInfo = (cu_fun_ptr)dlsym(libcuda,"cuMemGetInfo_v2");
#else
      typedef CUresult (*cu_fun_ptr)(int*, int*);
      cu_fun_ptr dl_cuMemGetInfo = (cu_fun_ptr)dlsym(libcuda,"cuMemGetInfo");
#endif
      if (NULL == dl_cuMemGetInfo) {
        KALDI_WARN << "cannot load cuMemGetInfo from libcuda.so";
      } else {
        // call the function
        dl_cuMemGetInfo(&mem_free, &mem_total);
      }
      // close the library
      dlclose(libcuda);
    }
  }
#endif
  // copy the output values outside
  if (NULL != free) *free = mem_free;
  if (NULL != total) *total = mem_total;
  // prepare the text output
  std::ostringstream os;
  os << "free:" << mem_free/(1024*1024) << "M, "
     << "used:" << (mem_total-mem_free)/(1024*1024) << "M, "
     << "total:" << mem_total/(1024*1024) << "M, "
     << "free/total:" << mem_free/(float)mem_total;
  return os.str();
}


void CuDevice::DeviceGetName(char* name, int32 len, int32 dev) {
  // prefill with something reasonable
  strncpy(name,"Unknown GPU",len);
#ifdef _MSC_VER
  cuDeviceGetName(name, len, dev);
#else
  // open libcuda.so
  void* libcuda = dlopen("libcuda.so",RTLD_LAZY);
  if (NULL == libcuda) {
    KALDI_WARN << "cannot open libcuda.so";
  } else {
    // define the function signature type
    typedef CUresult (*cu_fun_ptr)(char*,int,CUdevice);
    // get the symbol
    cu_fun_ptr cuDeviceGetName_ptr = (cu_fun_ptr)dlsym(libcuda,"cuDeviceGetName");
    if (NULL == cuDeviceGetName_ptr) {
      KALDI_WARN << "cannot load cuDeviceGetName from libcuda.so";
    } else {
      // call the function
      cuDeviceGetName_ptr(name, len, dev);
    }
    // close the library
    dlclose(libcuda);
  }
#endif
}


void CuDevice::CheckGpuHealth() {
  if (!Enabled()) return;
  CuTimer t;
  // prepare small matrices for a quick test
  Matrix<BaseFloat> a(50, 100);
  Matrix<BaseFloat> b(100 ,50);
  a.SetRandn();
  b.SetRandUniform();
  // multiply 2 small matrices in CPU:
  Matrix<BaseFloat> c(50, 50);
  c.AddMatMat(1.0, a, kNoTrans, b, kNoTrans, 0.0);
  // multiply same matrices in GPU:
  CuMatrix<BaseFloat> c1(50, 50);
  c1.AddMatMat(1.0, CuMatrix<BaseFloat>(a), kNoTrans, CuMatrix<BaseFloat>(b), kNoTrans, 0.0);
  // check that relative differnence is <1%
  AssertEqual(c, Matrix<BaseFloat>(c1), 0.01);
  // measure time spent in this check
  AccuProfile(__func__, t);
}


/*
  void CuDevice::Free(void *ptr) {
  CU_SAFE_CALL(cudaFree(ptr));
  }

  void* CuDevice::MallocPitch(size_t row_bytes, size_t num_rows, size_t *pitch) {
  void *ans = NULL;
  cudaError_t e = cudaMallocPitch(&ans, pitch, row_bytes, num_rows);
  if (e != cudaSuccess) {
  PrintMemoryUsage();
  KALDI_ERR << "CuDevice::MallocPitch: cannot allocate the requested memory ("
  << row_bytes << " x " << num_rows << " = "
  << row_bytes * num_rows << " bytes )";
  }
  return ans;
  }

  void* CuDevice::Malloc(size_t size) {
  void *ans = NULL;
  cudaError_t e = cudaMalloc(&ans, size);
  if (e != cudaSuccess) {
  PrintMemoryUsage();
  KALDI_ERR << "CuDevice::Malloc: cannot allocate the requested memory"
  << " (" << size << " bytes )";
  }
  return ans;
  }
*/

//wd007
void
CuDevice::GetBandwidth(int32 gpu_idx, float &d2h, float &h2d) {
    int memSize = 64*1024*1024;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    unsigned char *h_idata = NULL;
    unsigned char *h_odata = NULL;
    cudaEvent_t start, stop;
    bool PINNED = true;
    int MEMCOPY_ITERATIONS = 5;

    CU_SAFE_CALL(cudaEventCreate(&start));
    CU_SAFE_CALL(cudaEventCreate(&stop));

    //allocate host memory
    if (PINNED) {
	  	#if CUDART_VERSION >= 2020
        CU_SAFE_CALL(cudaHostAlloc((void **)&h_idata, memSize, cudaHostAllocPortable));
        CU_SAFE_CALL(cudaHostAlloc((void **)&h_odata, memSize, cudaHostAllocPortable));
	  	#else
        CU_SAFE_CALL(cudaMallocHost((void **)&h_idata, memSize)));
        CU_SAFE_CALL(cudaMallocHost((void **)&h_odata, memSize)));
	  	#endif 
    } else {
	         //pageable memory mode - use malloc
	         h_odata = (unsigned char *)malloc(memSize);

	         if (h_odata == 0) {
	         	KALDI_ERR << "Not enough memory available on host to run test!\n";
	             exit(EXIT_FAILURE);
	         }
    }

	//initialize the memory
	for (unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
	    h_idata[i] = (unsigned char)(i & 0xff);

	// allocate device memory
	unsigned char *d_idata;
	CU_SAFE_CALL(cudaMalloc((void **) &d_idata, memSize));

	//initialize the device memory
	CU_SAFE_CALL(cudaMemcpy(d_idata, h_idata, memSize,cudaMemcpyHostToDevice));

	//copy data from GPU to Host

	CU_SAFE_CALL(cudaEventRecord(start, 0));

	if (PINNED) {
	    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
	    	CU_SAFE_CALL(cudaMemcpyAsync(h_odata, d_idata, memSize,cudaMemcpyDeviceToHost, 0));
	} else {
	    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
	    	CU_SAFE_CALL(cudaMemcpy(h_odata, d_idata, memSize,cudaMemcpyDeviceToHost));
	}

    CU_SAFE_CALL(cudaEventRecord(stop, 0));

	    // make sure GPU has finished copying
	CU_SAFE_CALL(cudaDeviceSynchronize());
	    //get the the total elapsed time in ms

	CU_SAFE_CALL(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));


	//calculate bandwidth in MB/s
	bandwidthInMBs = (1e3f * memSize * (float)MEMCOPY_ITERATIONS) / (elapsedTimeInMs * (float)(1 << 20));
	d2h = bandwidthInMBs;

	/////////////////////////////////////////////////////
	//copy data from Host to GPU
	CU_SAFE_CALL(cudaEventRecord(start, 0));

	if (PINNED) {
	    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
	    	CU_SAFE_CALL(cudaMemcpyAsync(d_idata, h_odata, memSize,cudaMemcpyHostToDevice, 0));
	} else {
	    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
	    	CU_SAFE_CALL(cudaMemcpy(d_idata, h_odata, memSize,cudaMemcpyHostToDevice));
	}

	CU_SAFE_CALL(cudaEventRecord(stop, 0));

	// make sure GPU has finished copying
	CU_SAFE_CALL(cudaDeviceSynchronize());
	//get the the total elapsed time in ms

	CU_SAFE_CALL(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));


	//calculate bandwidth in MB/s
	bandwidthInMBs = (1e3f * memSize * (float)MEMCOPY_ITERATIONS) / (elapsedTimeInMs * (float)(1 << 20));
	h2d = bandwidthInMBs;

	//clean up memory
	CU_SAFE_CALL(cudaEventDestroy(stop));
	CU_SAFE_CALL(cudaEventDestroy(start));

	if (PINNED) {
		CU_SAFE_CALL(cudaFreeHost(h_idata));
		CU_SAFE_CALL(cudaFreeHost(h_odata));
	}
	else {
	    free(h_idata);
	    free(h_odata);
	}

	CU_SAFE_CALL(cudaFree(d_idata));
}

bool CuDevice::Initialize() {
	 // Check that we have at least one gpu
	  int32 n_gpu = 0;
	  cudaError_t e;
	  cudaGetDeviceCount(&n_gpu);
	  if(n_gpu == 0) {
	    KALDI_WARN << "No CUDA devices found";
	    return false;
	  }

	  gpuinfo_.resize(n_gpu);

	  // Get ratios of memory use, if possible
	  KALDI_LOG << "Selecting from " << n_gpu << " GPUs";
	  for(int32 n = 0; n < n_gpu; n++) {
	    int32 ret = cudaSetDevice(n);
	    switch(ret) {
	      case cudaSuccess : {
	        //create the CUDA context for the thread
	    	  e = cudaThreadSynchronize(); // deprecated, but for legacy not cudaDeviceSynchronize
      		if (e != cudaSuccess) {
        		KALDI_WARN << "Cannot select this device: return code " << e 
          		<< ", Error message: \"" << cudaGetErrorString(e) << "\"";
        		gpuinfo_[n].used = true;
        		continue;
      		}    

	        //get GPU name
	        char name[128];
	        DeviceGetName(name,128,n);
	        //get GPU memory stats
	        int64 free, total;
		float d2h, h2d;
	        std::string mem_stats;
	        mem_stats = GetFreeMemory(&free, &total);
	        //log
	        KALDI_LOG << "cudaSetDevice(" << n << "): "
	                  << name << "\t" << mem_stats;

	        GetBandwidth(n, d2h, h2d);
	        gpuinfo_[n].gpuid = n;
	        gpuinfo_[n].d2h_bandwidth = d2h;
	        gpuinfo_[n].h2d_bandwidth = h2d;
	        gpuinfo_[n].mem_free = free;
	        gpuinfo_[n].mem_total = total;
	        gpuinfo_[n].mem_ratio = free/(float)total;
	        gpuinfo_[n].used = false;

	        KALDI_LOG  << "gpu: "  << n << " ==> "
					<< "free: " << free/1024/1024 << "M, "
					<< "total: "<< total/1024/1024 << "M, "
	                << "ratio: "<< free/(float)total << "   "
	        	    << "d2h bandwidth: " << d2h << "MB/s, "
	                << "h2d bandwidth: "<< h2d << "MB/s";

	        //destroy the CUDA context for the thread
	        cudaThreadExit(); //deprecated, but for legacy reason not cudaDeviceReset
	      } break;

	#if (CUDA_VERSION > 3020)
	      case cudaErrorDeviceAlreadyInUse :
	        KALDI_LOG << "cudaSetDevice(" << n << "): "
	                  << "Device cannot be accessed, used EXCLUSIVE-THREAD mode...";
	        break;
	#endif
	      case cudaErrorInvalidDevice :
	        KALDI_LOG << "cudaSetDevice(" << n << "): "
	                  << "Device cannot be accessed, not a VALID CUDA device!";
	        break;
	      default :
	        KALDI_LOG << "cudaSetDevice(" << n << "): "
	                  << "returned " << ret << ", "
	                  << cudaGetErrorString((cudaError_t)ret);
	    }
	  }
      multi_gpu_ = true;
      //device_id_ = 0;
      return true;
}

int CuDevice::SelectGpu() {
	int max_id = 0;
	// select device if more than one
	int32 n_gpu = 0;
	cudaGetDeviceCount(&n_gpu);

	if(n_gpu == 0) {
		 KALDI_WARN << "No CUDA devices found";
		 return -1;
	}

	// reset gpuinfo if all GPU are used.
	int i = 0;
	for (i = 0; i < gpuinfo_.size(); i++) {
		if (!gpuinfo_[i].used) {
			break;
		}
	}

	if (i == gpuinfo_.size()) {
		for (i = 0; i < gpuinfo_.size(); i++)
			gpuinfo_[i].used = false;
	}

	// select best GPU
	if (n_gpu > 0) {
		for (int i = 0; i < gpuinfo_.size(); i++) {
			if (!gpuinfo_[i].used) {
				max_id = i;
				break;
			}
		}
		//find GPU with max free memory
		for (int n = 1; n < gpuinfo_.size(); n++) {
			if (!gpuinfo_[n].used && gpuinfo_[n].mem_ratio > gpuinfo_[max_id].mem_ratio)
				max_id = n;
		}

		CU_SAFE_CALL(cudaSetDevice(max_id));
        FinalizeActiveGpu();

        /*
        KALDI_LOG << "Selected device: " << max_id << " (automatically)";

        KALDI_LOG << "free: " << gpuinfo_[max_id].mem_free/1024/1024 << "M, "
                  << "total: "<< gpuinfo_[max_id].mem_total/1024/1024 << "M, "
                  << "ratio: "<< gpuinfo_[max_id].mem_ratio << "   "
        	  << "d2h bandwidth: " << gpuinfo_[max_id].d2h_bandwidth << "MB/s, "
                  << "h2d bandwidth: "<< gpuinfo_[max_id].h2d_bandwidth << "MB/s";

		CU_SAFE_CALL(cudaSetDevice(max_id));
        device_id_ = max_id;
        // Initialize CUBLAS.
        CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle_));
        CUBLAS_SAFE_CALL(cublasSetStream(cublas_handle_, cudaStreamPerThread));
        // Initialize the cuSPARSE library
        CUSPARSE_SAFE_CALL(cusparseCreate(&cusparse_handle_));
        CUSPARSE_SAFE_CALL(cusparseSetStream(cusparse_handle_, cudaStreamPerThread));
        */
	}

	gpuinfo_[max_id].used = true;
	return max_id;
}


void CuDevice::SelectGpu(int gpu_id) {

	int32 n_gpu = 0;
	cudaGetDeviceCount(&n_gpu);
	if(gpu_id >= n_gpu || gpu_id < 0) {
		KALDI_ERR << "Cannot select GPU " << gpu_id
              << ", detected " << n_gpu << " CUDA capable cards!";
	}

	CU_SAFE_CALL(cudaSetDevice(gpu_id));
    FinalizeActiveGpu();

	gpuinfo_[gpu_id].used = true;
}

void
CuDevice::SelectPreferGpu(int gpu_id) {

	int32 n_gpu = 0, select_id = gpu_id;
	cudaGetDeviceCount(&n_gpu);
	if(gpu_id >= n_gpu || gpu_id < 0) {
		KALDI_ERR << "Cannot select GPU " << gpu_id
              << ", detected " << n_gpu << " CUDA capable cards!";
	}

	if (gpuinfo_[gpu_id].used) {
		KALDI_WARN << "Cannot select specified recommend device "<<  gpu_id <<
				", maybe something wrong happened.";
		select_id = -1;
		for (int i = 0; i < gpuinfo_.size(); i++) {
			if (!gpuinfo_[i].used) {
				select_id = i;
				break;
			}
		}
	}

	if (select_id < 0) {
		KALDI_ERR << "Cannot select valid CUDA capable GPU cards, all CUDA-capable devices are busy or unavailable!";
	}

	gpu_id = select_id;

	CU_SAFE_CALL(cudaSetDevice(gpu_id));
    FinalizeActiveGpu();

	//gpuinfo_[gpu_id].used = true;
}

int CuDevice::GetDeviceId() {
	int32 n_gpu;
	cudaGetDevice(&n_gpu);

	return n_gpu;
}

int CuDevice::MPISelectGpu(MPIGpuInfo *gpuinfo, MPI_Win &win, int thread_idx, int num_threads) {
	int myid, numprocs, len, i, j, max = 0, size, id;
	char name[256];
	MPI_Status status;
	MPI_Request handle;

    int32 n_gpu = 0;
    cudaGetDeviceCount(&n_gpu);
    if(n_gpu == 0) {
	    KALDI_WARN << "No CUDA devices found";
	    return -1;
    }


	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Get_processor_name(name,&len);
	bool selected;
	int gpuid = -1;
	size = num_threads*numprocs*sizeof(MPIGpuInfo);
	id = myid+thread_idx*numprocs;

	if (myid == 0) {
		MPI_Win_create(gpuinfo, size, sizeof(MPIGpuInfo), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
		//MPI_Win_fence(0, win);

		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);


		for (i = 0; i < gpuinfo_.size(); i++) {
			selected = false;
			for (j = 0; j < num_threads*numprocs; j++) {
				if (strcmp(name, gpuinfo[j].hostname) == 0 && gpuinfo[j].gpuid == i) {
					selected = true;
					break;
				}
			}

			if ((gpuid == -1 || gpuinfo_[i].mem_ratio > gpuinfo_[gpuid].mem_ratio) && !selected)
				gpuid = i;

			if (gpuinfo_[i].mem_ratio > gpuinfo_[max].mem_ratio)
				max = i;
		}
		if (gpuid == -1) gpuid = max;
		strcpy(gpuinfo[id].hostname, name);
		gpuinfo[id].gpuid = gpuid;
		gpuinfo[id].myid = myid;

		MPI_Win_unlock(0, win);

		//MPI_Win_fence(0, win);

	} else {
		MPI_Win_create(NULL, 0, sizeof(MPIGpuInfo), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
		//MPI_Win_fence(0, win);

		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
		MPI_Rget(gpuinfo, size, MPI_BYTE, 0, 0, size, MPI_BYTE, win, &handle);
		MPI_Wait(&handle, &status);

		for (i = 0; i < gpuinfo_.size(); i++) {
			selected = false;
			for (j = 0; j < num_threads*numprocs; j++) {
				if (strcmp(name, gpuinfo[j].hostname) == 0 && gpuinfo[j].gpuid == i) {
					selected = true;
					break;
				}
			}

			if ((gpuid == -1 || gpuinfo_[i].mem_ratio > gpuinfo_[gpuid].mem_ratio) && !selected)
				gpuid = i;

			if (gpuinfo_[i].mem_ratio > gpuinfo_[max].mem_ratio)
				max = i;
		}
		if (gpuid == -1)
			gpuid = max;
		strcpy(gpuinfo[id].hostname, name);
		gpuinfo[id].gpuid = gpuid;
		gpuinfo[id].myid = myid;
		MPI_Put(&gpuinfo[id],sizeof(MPIGpuInfo),MPI_BYTE,0,id,sizeof(MPIGpuInfo),MPI_BYTE,win);

		MPI_Win_unlock(0, win);

		//MPI_Win_fence(0, win);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast((void*)gpuinfo, size, MPI_BYTE, 0, MPI_COMM_WORLD);

    /*
    KALDI_LOG << "Selected device: " << gpuid << " (automatically)";

    KALDI_LOG << "free: " << gpuinfo_[gpuid].mem_free/1024/1024 << "M, "
              << "total: "<< gpuinfo_[gpuid].mem_total/1024/1024 << "M, "
              << "ratio: "<< gpuinfo_[gpuid].mem_ratio << "   "
    	      << "d2h bandwidth: " << gpuinfo_[gpuid].d2h_bandwidth << "MB/s, "
              << "h2d bandwidth: "<< gpuinfo_[gpuid].h2d_bandwidth << "MB/s";
    */

	CU_SAFE_CALL(cudaSetDevice(gpuid));
    FinalizeActiveGpu();

	gpuinfo_[gpuid].used = true;
    device_id_ = gpuinfo[id].gpuid;

	return gpuinfo[id].gpuid;
}

void CuDevice::CreateHandle(cublasHandle_t *handle){ CU_SAFE_CALL(cublasCreate(handle));}
void CuDevice::DestroyHandle(cublasHandle_t handle){ CU_SAFE_CALL(cublasDestroy(handle));}

CuDevice::CuDevice():
    device_id_(-1),
    initialized_(false),
    device_id_copy_(-1),
    cublas_handle_(NULL),
    cusparse_handle_(NULL) {
}

CuDevice::~CuDevice() {
  if (cublas_handle_)
    CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle_));
  if (cusparse_handle_)
    CUSPARSE_SAFE_CALL(cusparseDestroy(cusparse_handle_));
}


// Each thread has its own copy of the CuDevice object.
// Note: this was declared "static".
thread_local CuDevice CuDevice::this_thread_device_;
// CuDevice CuDevice::this_thread_device_;

// define and initialize the static members of the CuDevice object.
// int32 CuDevice::device_id_ = -1;
bool CuDevice::multi_threaded_ = false;
bool CuDevice::multi_gpu_ = false;
unordered_map<std::string, double, StringHasher> CuDevice::profile_map_;
std::mutex CuDevice::profile_mutex_;
int64 CuDevice::free_memory_at_startup_;
cudaDeviceProp CuDevice::properties_;
bool CuDevice::debug_stride_mode_ = false;
std::vector<GpuInfo> CuDevice::gpuinfo_;


void SynchronizeGpu() {
  cuda_legacy_noop();
  CU_SAFE_CALL(cudaGetLastError());
}

}  // namespace kaldi

#else  // #if HAVE_CUDA == 1

namespace kaldi {
// SynchronizeGpu() does nothing if we didn't compile for GPU.
void SynchronizeGpu() { }
}

#endif  // #if HAVE_CUDA == 1
