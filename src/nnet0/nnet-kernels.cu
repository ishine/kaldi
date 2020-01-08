// nnet0/nnet-kernels.cu

// Copyright  2015  Johns Hopkins University (author: Daniel Povey)


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


#include <cfloat>
#include "nnet0/nnet-kernels-ansi.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 200
#error - Kaldi no longer supports CC1.x devices. Please use a newer GPU or \
         configure with --use-cuda=no (this will disable the use of GPU).
#endif

#define ATOMIC_CONST 32
#define CU_BLOCK_DIM 1024

template <typename Real>
__host__ __device__ 
inline float log_plus(Real a, Real b) {
    if (a == -float(INFINITY)) return b;
    if (b == -float(INFINITY)) return a;
    float m = a > b ? a : b;
    return log1pf(expf(-fabs(a - b))) + m;
}

template <typename Real>
__device__ 
float atomic_log_plus(Real *addr_f, Real value) {
    int *addr = (int*)addr_f;
    float expected = *addr_f;
    float sum = log_plus(expected, value);
    int old_value = atomicCAS(addr, __float_as_int(expected), __float_as_int(sum));

    while (old_value != __float_as_int(expected)) {
        expected = __int_as_float(old_value);
        sum = log_plus(expected, value);
        old_value = atomicCAS(addr, __float_as_int(expected), __float_as_int(sum));
    }
    return __int_as_float(old_value);
}

// <<<batch_size, CU_BLOCK_CONST>>>
template <typename Real>
__global__ 
static void alpha_first_kernel(Real *alpha,
                                   const int alpha_size,
                                   const int batch_size,
                                   const int T,
                                   const Real * const start_weight) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    for (int idx = tid; idx < alpha_size; idx += blockDim.x) {
        alpha[mini_batch_idx * alpha_size * (T+1) + idx] = start_weight[idx];
    }
}

__global__ 
static void alpha_kernel(float *alpha,
                             const float* const logits,                   
                             const int batch_size,
                             const int T,
                             const int t,
                             const int * const input_lengths,
                             const int alpha_size,
                             const int logits_size,
                             const IntPair * const alpha_transition_index,
                             const Transition * const alpha_transition,
                             bool batch_first) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (t > input_lengths[mini_batch_idx]) return;

    int idx1 = mini_batch_idx * alpha_size * (T+1) + alpha_size * t;
    int idx2 = mini_batch_idx * alpha_size * (T+1) + alpha_size * (t-1);
    int idx3 = 0;
    if (batch_first)
    	idx3 = mini_batch_idx * logits_size * T + logits_size * (t-1);
    else
    	idx3 = batch_size * logits_size * (t-1) + mini_batch_idx * logits_size;

    for (int idx = tid; idx < alpha_size; idx += blockDim.x) {
        int start = alpha_transition_index[idx].first;
        int end = alpha_transition_index[idx].second;
        float result = -float(INFINITY);
        for (int k = start; k <= end; k++) {
            result = log_plus(alpha[idx2+alpha_transition[k].state] + 
                alpha_transition[k].weight + logits[idx3+alpha_transition[k].label], result);
        }
        alpha[idx1+idx] = result;
    }
}

__global__ 
static void alpha_last_kernel(float *alpha,
                                  const int alpha_size,
                                  const int batch_size,
                                  const int T,
                                  const int * const input_lengths,
                                  const float * const end_weight) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int alpha_start = mini_batch_idx * alpha_size * (T+1);
    int cT = input_lengths[mini_batch_idx];

    for (int idx = tid; idx < alpha_size; idx += blockDim.x) {
        alpha[alpha_start+cT*alpha_size+idx] += end_weight[idx];
    }
}

// <<< minibatch, N = 32,64,128...>>>
__global__ 
static void alpha_lld_kernal(const float * const alpha,
                                 const int alpha_size,
                                 const int T,
                                 const int * const input_lengths,
                                 float * loglikelihood) {
    int mini_batch_idx = blockIdx.x;
    int idx = threadIdx.x;
    int block_dim = blockDim.x;
    int cT = input_lengths[mini_batch_idx];
    int last_idx = alpha_size * (T+1) * mini_batch_idx + cT*alpha_size;
    // printf("enter alpha_lld_kernal, block.x: %d, thread.x: %d\n", blockIdx.x, threadIdx.x);

    extern __shared__ float sdata[];
    float temp = -float(INFINITY);

    for (int i = idx; i < alpha_size; i += block_dim) {
        temp = log_plus(temp, alpha[last_idx+i]);
    }
    sdata[idx] = temp;
    __syncthreads();

    for (int shift = block_dim / 2; shift > warpSize; shift >>= 1) {
        if (idx < shift) {
            sdata[idx] = log_plus(sdata[idx], sdata[idx+shift]);
        }
        __syncthreads();
    }

    if (idx < warpSize) {
        for (int shift = warpSize; shift > 0; shift >>= 1) {
            sdata[idx] = log_plus(sdata[idx], sdata[idx+shift]);
        }
    }
    __syncthreads();

    if (idx == 0) {
        loglikelihood[mini_batch_idx] = sdata[0];
        // printf("alpha loglikelihod: %f mini_batch %d\n", loglikelihood[mini_batch_idx], mini_batch_idx);
    }
}

template <typename Real>
__global__ 
static void beta_last_kernel(Real *beta,
                                 const int beta_size,
                                 const int batch_size,
                                 const int * const input_lengths,
                                 const Real * const end_weight) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int cT = input_lengths[mini_batch_idx];

    for (int idx = tid; idx < beta_size; idx += blockDim.x) {
        beta[mini_batch_idx * 2 * beta_size + (cT % 2) * beta_size + idx] = end_weight[idx];
    }
}

template <typename Real>
__global__ void beta_first_kernel(Real *beta, 
                                  const int beta_size,
                                  const int batch_size,
                                  const Real * const start_weight) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    for (int idx = tid; idx < beta_size; idx += blockDim.x) {
        beta[mini_batch_idx * 2 * beta_size + idx] += start_weight[idx];
    }
}

template <typename Real>
__global__ 
static void beta_kernel(Real *beta,
                            const Real* const alpha,
                            const Real* const logits, 
                            Real *grad_storage,                            
                            const int batch_size,
                            const int T,
                            const int t,
                            const int *input_lengths,
                            const int beta_size,
                            const int logits_size,
                            const IntPair * const beta_transition_index,
                            const Transition * const beta_transition,
                            const bool batch_first) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= input_lengths[mini_batch_idx]) return;
    int idx1 = mini_batch_idx * beta_size * (T+1) + beta_size * t;
    int idx2 = mini_batch_idx * beta_size * 2 + beta_size * ((t+1) % 2);
    int idx3 = mini_batch_idx * beta_size * 2 + beta_size * (t % 2);
    int idx4 = 0;
    if (batch_first)
    	idx4 = mini_batch_idx * logits_size * T + logits_size * t;
    else
    	idx4 = batch_size * logits_size * t + mini_batch_idx * logits_size;
    int idx5 = mini_batch_idx * logits_size * ATOMIC_CONST;

    for (int idx = tid; idx < beta_size; idx += blockDim.x) {
        int start = beta_transition_index[idx].first;
        int end = beta_transition_index[idx].second;

        float beta_result = -float(INFINITY);
        float temp_value = -float(INFINITY);

        for (int k = start; k <= end; k++) {
            temp_value = beta[idx2+beta_transition[k].state] + beta_transition[k].weight +
                logits[idx4+beta_transition[k].label];
            beta_result = log_plus(temp_value, beta_result);
            float partial_grad = alpha[idx1+idx] + temp_value; 
            float *grad_position = grad_storage + idx5 + beta_transition[k].label * ATOMIC_CONST + threadIdx.x % ATOMIC_CONST;
            atomic_log_plus(grad_position, partial_grad);
        }
        beta[idx3+idx] = beta_result;
    }
}

template <typename Real>
__global__ 
static void copy_grad(Real *grad_storage,
                      Real *grad_net,
                      const Real * const alpha_lld,
                      const int * const input_lengths,                     
                      const int batch_size,
                      const int logits_size,
                      const int T,
                      const int t,
                      const bool batch_first) {
    int mini_batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= input_lengths[mini_batch_idx]) return;
	
	int idx1 = 0;
	if (batch_first)
    	idx1 = mini_batch_idx * logits_size * T + logits_size * t;
    else
    	idx1 = batch_size * logits_size * t + mini_batch_idx * logits_size;
    	
    float lld = alpha_lld[mini_batch_idx];
    for (int idx = tid; idx < logits_size; idx += blockDim.x) {
        float *grad_position = grad_net + idx1 + idx;
        int idx_storage = mini_batch_idx*logits_size*ATOMIC_CONST+idx*ATOMIC_CONST;

        float grad = -float(INFINITY);
        for (int i = 0; i < ATOMIC_CONST; i++) {
            grad = log_plus(grad_storage[idx_storage+i], grad);
            grad_storage[idx_storage+i] = -float(INFINITY);
        }
        *grad_position = expf(grad - lld);
    }
}

template <typename Real>
__global__ 
static void beta_lld_kernal(const Real * const beta,
                            const int beta_size,
                            Real * loglikelihood) {
    int idx = threadIdx.x;
    int first_idx = beta_size * 2 * idx;
    loglikelihood[idx] = beta[first_idx];
}


void cuda_compute_alpha(dim3 Gr, dim3 Bl,
				   BaseFloat *alpha,
                   const BaseFloat *logits,                   
                   const int batch_size,
                   int T,
                   const int alpha_size,
                   int logits_size,
                   int *input_lengths,
                   BaseFloat *loglikelihood,
                   const BaseFloat *start_weight,
                   const BaseFloat *end_weight,
                   const IntPair *transition_index_alpha,
                   const Transition *transition_alpha,
                   cudaStream_t stream,
                   const bool batch_first) {
    int alpha_lld_dim = 128;
    alpha_first_kernel<<<Gr, Bl, 0, stream>>>(alpha, alpha_size, batch_size, T, start_weight);

    for (int t = 1; t <= T; t++) {
        alpha_kernel<<<Gr, Bl, 0, stream>>>(alpha, logits, batch_size, T, t, input_lengths, 
            alpha_size, logits_size, transition_index_alpha, transition_alpha, batch_first);
    }

    alpha_last_kernel<<<Gr, Bl, 0, stream>>>(alpha, alpha_size, batch_size, T, input_lengths, end_weight);
    alpha_lld_kernal<<<Gr, alpha_lld_dim, sizeof(float)*alpha_lld_dim, stream>>>(alpha, alpha_size, T, input_lengths, loglikelihood);
    // cudaDeviceSynchronize();
}

void cuda_compute_beta_and_grad(dim3 Gr, dim3 Bl,
					   BaseFloat *beta,
					   const BaseFloat * alpha,
					   const BaseFloat * logits,
					   const BaseFloat * alpha_lld,
					   BaseFloat *grad_storage,
					   BaseFloat *grad_net,
					   const int batch_size,
					   const int T,
					   const int beta_size,
					   const int logits_size,
					   const int * input_lengths,
					   BaseFloat * loglikelihood,
					   const BaseFloat *start_weight,
					   const BaseFloat *end_weight,
					   const IntPair *transition_index_beta,
					   const Transition *transition_beta,
                       cudaStream_t stream,
					   const bool batch_first) {
    // set grad_storage
    copy_grad<<<Gr, Bl, 0, stream>>>(grad_storage, grad_net, alpha_lld, input_lengths, batch_size, logits_size, T, 0, batch_first);

    beta_last_kernel<<<Gr, Bl, 0, stream>>>(beta, beta_size, batch_size, input_lengths, end_weight);
    for (int t = T-1; t >= 0; t--) {
        beta_kernel<<<Gr, Bl, 0, stream>>>(beta, alpha, logits, grad_storage, batch_size, T, t, input_lengths, beta_size, logits_size,
            transition_index_beta, transition_beta, batch_first);
        copy_grad<<<Gr, Bl, 0, stream>>>(grad_storage, grad_net, alpha_lld, input_lengths, batch_size, logits_size, T, t, batch_first);
    }

    beta_first_kernel<<<Gr, Bl, 0, stream>>>(beta, beta_size, batch_size, start_weight);
    beta_lld_kernal<<<1, Gr>>>(beta, beta_size, loglikelihood);
}
