// lm/lm-model-sync.cc

// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

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

#include "nnet0/nnet-affine-transform.h"
#include "nnet0/nnet-lstm-projected-streams-fast.h"
#include "nnet0/nnet-lstm-projected-streams.h"
#include "nnet0/nnet-lstm-streams.h"
#include "nnet0/nnet-class-affine-transform.h"
#include "nnet0/nnet-word-vector-transform.h"

#include "lm/lm-model-sync.h"

namespace kaldi {
namespace lm {

void LmModelSync::Init(Nnet *nnet)
{
	if (NULL != this->free_data_)
		return;

	size_t size = 0;
	void *data = NULL;
	void *free_data = NULL;
	int32 dim = 0;

	dim = this->GetDim(nnet);
	this->dim_ = dim;

	size = dim * sizeof(BaseFloat)+16;

	thread_data_.resize(num_threads_+2);
	thread_free_data_.resize(num_threads_+2);

	for (int i = 0; i < thread_data_.size(); i++)
	{
		CU_SAFE_CALL(cudaHostAlloc((void**) &free_data, size, cudaHostAllocPortable)); // cudaHostAllocDefault
		data = (free_data ? (void *)( (((unsigned long)*(&free_data)) + 15) & ~0xFUL ) : NULL) ;
		if (NULL != data)
		{
			this->thread_data_[i] = static_cast<BaseFloat*> (data);
			this->thread_free_data_[i] = static_cast<BaseFloat*> (free_data);
		}
		else
		{
		    throw std::bad_alloc();
		}
	}

	this->data_ = thread_data_[num_threads_];
	this->gradient_data_ = thread_data_[num_threads_+1];

	CU_SAFE_CALL(cudaMemset(this->gradient_data_, 0, dim_*sizeof(BaseFloat)));

}

void LmModelSync::MultiMachineInit()
{
    if (opts_->num_procs > 1)
    {
        //p_merge_func_ = ModelMergeFunction::Factory(opts_, this);
#if HAVE_CUDA == 1
        gpuinfo_ = (MPIGpuInfo*)malloc(opts_->num_procs * opts_->num_threads * sizeof(MPIGpuInfo));
        std::memset(gpuinfo_, 0, opts_->num_procs * opts_->num_threads * sizeof(MPIGpuInfo));
#endif
    }
}

void LmModelSync::Destory()
{
	if (NULL != this->data_)
	{
		for (int i = 0; i < thread_free_data_.size(); i++)
		{
			CU_SAFE_CALL(cudaFreeHost(thread_free_data_[i]));
			this->thread_free_data_[i] = NULL;
			this->thread_data_[i] = NULL;
		}
		this->data_ = NULL;
		this->gradient_data_ = NULL;
		this->dim_ = 0;
	}
}

int LmModelSync::GetDim(Nnet *nnet)
{
	return nnet->GetDim();
}

void LmModelSync::GetWeight(Nnet *nnet, int32 thread_idx, int32 buffer_idx)
{
	if (NULL == this->data_)
		this->Init(nnet);

	KALDI_ASSERT(thread_idx <= num_threads_ - 1);

	void *host_data_ = buffer_idx < 0 ? (void*)this->data_ : this->thread_data_[thread_idx];

	// device to host
	nnet->WeightCopy(host_data_, LmModelSync::kDstAddress, LmModelSync::kCudaMemcpyDeviceToHost);
}

void LmModelSync::SetWeight(Nnet *nnet, int32 thread_idx, int32 buffer_idx)
{
	KALDI_ASSERT(this->data_ != NULL);

	KALDI_ASSERT(thread_idx <= num_threads_ - 1);

	void *host_data_ = buffer_idx < 0 ? (void *)this->data_ : this->thread_data_[thread_idx];

	// host to device
	nnet->WeightCopy(host_data_, LmModelSync::kSrcAddress, LmModelSync::kCudaMemcpyHostToDevice);
}

void LmModelSync::InnerMachineSyncStatus(int32 status)
{
    if (status == 0) {
	    inner_mutex_.Lock();
	    num_finished_++;
	    inner_mutex_.Unlock();
    }
}

void LmModelSync::CrossMachineSyncStatus(int status)
{
	int total_status = 0;
	if ((left_merge_ <= 1 && !is_lastmerge_) || status == 0)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&status, (void*)(&total_status), 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		if (total_status < opts_->num_procs)
			is_lastmerge_ = true;
	}
}

void LmModelSync::CrossMachineSync()
{
	// cross machine reduce
	//void *srcaddr = (void *) (opts_->myid==0 ? MPI_IN_PLACE : this->thread_data_[0]);
	void *srcaddr = (void *) MPI_IN_PLACE;
	void *dstaddr = (void *) this->thread_data_[0];
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(srcaddr, dstaddr, this->dim_, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void LmModelSync::ThreadSync(int32 thread_idx, int status)
{
	Timer synctm, tm;
	// internal machine thread sync status
	if (status == 0 ) 
        InnerMachineSyncStatus(status);
	while (true)
	{
		this->barrier_.Wait();
		if (num_finished_ == 0 || num_finished_ == num_threads_)
			break;
        else if (status == 1) {
            is_lastmerge_ = true;
            return;
        }
	}

	// cross machine process sync status
	if (opts_->num_procs > 1)
	{
        if (thread_idx == 0)
		    CrossMachineSyncStatus(status);

	    this->barrier_.Wait();
		if (this->is_lastmerge_ && status == 1)
			return;
	}

	tm.Reset();
	this->barrier_.Wait();

	BaseFloat *cur_model, *thread_model, *last_model, *gradient;
	int offset = this->dim_/num_threads_ * thread_idx;
	int len = (thread_idx == num_threads_-1 ? dim_-offset : dim_/num_threads_);
    int num_jobs = num_threads_;

	cur_model = this->thread_data_[0] + offset;
	last_model = this->data_ + offset;
	gradient = this->gradient_data_ + offset;

	for (int i = 1; i < num_threads_; i++)
	{
		thread_model = this->thread_data_[i] + offset;
		cblas_Xaxpy(len, 1.0, thread_model, 1, cur_model, 1);
	}

	//KALDI_VLOG(2) << "THREAD_Reduce: " << tm.Elapsed();

	// cross machine reduce
	if (opts_->num_procs > 1)
	{
	    this->barrier_.Wait();

	    tm.Reset();
        if (thread_idx == 0)
		    CrossMachineSync();
	    KALDI_VLOG(2) << "CrossMachineSync MPI_AllReduce: " << tm.Elapsed();
        num_jobs *= opts_->num_procs;

	    this->barrier_.Wait();
	}

	tm.Reset();
	// model merge ...
	// average W(t)
	cblas_Xscal(len, 1.0/num_jobs, cur_model, 1);
	// global gradient G(t) = average W(t) - W(t-1)
	cblas_Xaxpy(len, -1, last_model, 1, cur_model, 1);
	// delta(t) = mmt * delta_(t-1) + lr * G(t)
	if (mmt_ < 0.0) mmt_ = 1.0 - 1.0/num_jobs;
	cblas_Xscal(len, mmt_, gradient, 1);
	cblas_Xaxpy(len, learnrate_, cur_model, 1, gradient, 1);

	// CBM: W(t) = W(t-1) + delta(t)
	cblas_Xaxpy(len, 1.0, gradient, 1, last_model, 1);
	// NBM: W(t) = W(t-1) + delta(t) + mmt*delta(t)
	//cblas_Xaxpy(len, 1.0+mmt_, gradient, 1, last_model, 1);

	this->barrier_.Wait();
	//KALDI_VLOG(2) << "THREAD_Merge: " << tm.Elapsed();

    if (thread_idx == 0) left_merge_--;
	KALDI_VLOG(2) << "ThreadSync total : " << synctm.Elapsed();
}

/*
 * 'ark,o:copy-feats scp:exp/tri_dnn_mmi/scplist/train.scp ark:- |'
 */

std::string LmParallelUtil::AddSuffix(std::string filename, int idx)
{
  char buf[1024];
  char suf[1024], ext[1024], fn[1024];
  int  len;

  const char *pfn = filename.c_str();
  len = strlen(pfn);
  const char *p1, *p2;
  p1 = strstr(pfn,"scp:");
  if (NULL == p1) return "";
  p2 = strchr(p1, ' ');
  if (NULL == p2) p2 = pfn+len;

  strncpy(fn, pfn, p2-pfn); fn[p2-pfn] = '\0';
  int l1 = strlen(fn);
  char *p3 = strrchr(fn, '.');
  *p3='\0';

  strncpy(suf,p3+1, fn+l1-p3); suf[fn+l1-p3]='\0';

  strncpy(ext, p2, pfn+len-p2); ext[pfn+len-p2]='\0';

  sprintf(buf,"%s.%d.%s%s",fn,idx,suf, ext);

  return buf;
}

std::string LmParallelUtil::FAddSuffix(std::string filename, int idx)
{
  char buf[1024];
  char ext[128], fn[128];
  int  len;

  const char *pfn = filename.c_str();
  len = strlen(pfn);
  const char *p2;

  p2 = strchr(pfn, '.');

  strncpy(fn,pfn, p2-pfn); fn[p2-pfn]='\0';
  strncpy(ext, p2+1, pfn+len-p2); ext[pfn+len-p2]='\0';

  sprintf(buf,"%s.%d.%s",fn,idx,ext);

  return buf;
}

std::string LmParallelUtil::GetFilename(std::string filename)
{
  char fn[128];

  const char *pfn = filename.c_str();
  const char *p1, *p2;
  p1 = strstr(pfn,"scp:");
  p2 = strchr(p1, ' ');


  strncpy(fn,p1+4, p2-p1-4); fn[p2-p1-4]='\0';

  return fn;
}

int LmParallelUtil::NumofMerge(std::string fn, int merge_size)
{
	std::string sfn = fn+".len";
	std::ifstream in(sfn.c_str());
	std::string str, featname;
	int len, piece = 0;
	size_t frames = 0;
	while(std::getline(in, str))
	{
		std::istringstream ss(str);
		ss>>featname>>len;

		if (frames + len > merge_size)
		{
			piece++;
			frames = 0;
		}
		frames += len;
	}

	if (frames > merge_size/3)
		piece++;

	return piece;
}

int LmParallelUtil::NumofCEMerge(std::string fn, int merge_size)
{
	std::string sfn = fn+".len";
	std::ifstream in(sfn.c_str());
	std::string str, featname;
	int len, piece = 0;
	size_t frames = 0;
	while(std::getline(in, str))
	{
		std::istringstream ss(str);
		ss>>featname>>len;

		frames += len;
	}

	piece = frames/merge_size + 1;

	return piece;
}

} // namespace nnet
} // namespace kaldi
