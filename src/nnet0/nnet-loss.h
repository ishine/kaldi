// nnet0/nnet-loss.h

// Copyright 2011-2015  Brno University of Technology (author: Karel Vesely)

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

#ifndef KALDI_NNET_NNET_LOSS_H_
#define KALDI_NNET_NNET_LOSS_H_

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"
#include "hmm/posterior.h"

#include "warp-ctc/include/ctc.h"
//#include "warp-transducer/include/rnnt.h"
#include "add_network/include/rnnt.h"
#include "nnet0/nnet-kernels-ansi.h"

namespace kaldi {
namespace nnet0 {

struct LossOptions {
  int32 loss_report_frames; ///< Report loss value every 'report_interval' frames,
  bool loss_report_class;

  LossOptions():
    loss_report_frames(1*3600*100), loss_report_class(false) // 5h,
  { }

  void Register(OptionsItf *opts) {
    opts->Register("loss-report-frames", &loss_report_frames,
        "Report loss per blocks of N frames (0 = no reports)");
    opts->Register("loss-report-class", &loss_report_class,
            "Generate string with per-class error report");
  }
};

class LossItf {
 public:
  LossItf(LossOptions& opts) {
	opts_ = opts;
  }
  virtual ~LossItf() { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  virtual void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat> &net_out, 
            const CuMatrixBase<BaseFloat> &target,
			CuMatrixBase<BaseFloat> *diff) = 0;

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  virtual void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat> &net_out, 
            const Posterior &target,
            CuMatrix<BaseFloat> *diff) = 0;
  
  /// Generate string with error report,
  virtual std::string Report() = 0;

  /// Get loss value (frame average),
  virtual BaseFloat AvgLoss() = 0;

  /// Merge statistic data
  virtual void Add(LossItf *loss){};
  virtual void Merge(int myid, int root){};

 protected:
  LossOptions opts_;
  Timer timer_;
};


class Xent : public LossItf {
 public:
  Xent(LossOptions &opts) : LossItf(opts), frames_(0.0), correct_(0.0), loss_(0.0), entropy_(0.0),
           frames_progress_(0.0), xentropy_progress_(0.0), entropy_progress_(0.0),
		   correct_progress_(0.0), elapsed_seconds_(0) { }
  ~Xent() { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat> &net_out, 
            const CuMatrixBase<BaseFloat> &target,
			CuMatrixBase<BaseFloat> *diff);

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat> &net_out, 
            const Posterior &target,
            CuMatrix<BaseFloat> *diff);
  
  /// Evaluate cross entropy using target-posteriors (supports soft labels),
   void Eval(const VectorBase<BaseFloat> &frame_weights,
             const CuMatrixBase<BaseFloat> &net_out,
             const std::vector<int32> &target,
			 CuMatrixBase<BaseFloat> *diff);

   void GetTargetWordPosterior(Vector<BaseFloat> &tgt);

  /// Generate string with error report,
  std::string Report();

  /// Generate string with per-class error report,
  std::string ReportPerClass();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
    return (loss_ - entropy_) / frames_;
  }

  /// Merge statistic data
  void Add(LossItf *loss);
  void Merge(int myid, int root);

 private: 
  // main stats collected per target-class,
  CuVector<double> frames_vec_;
  CuVector<double> xentropy_vec_;
  CuVector<double> entropy_vec_;

  Vector<double> correct_vec_;
  Vector<double> frames_h_vec_;
  Vector<double> xentropy_h_vec_;
  Vector<double> entropy_h_vec_;

  double frames_;
  double correct_;
  double loss_;
  double entropy_;

  // partial results during training
  double frames_progress_;
  double xentropy_progress_;
  double entropy_progress_;
  double correct_progress_;
  std::vector<float> loss_vec_;
  double elapsed_seconds_;

  // weigting buffer,
  CuVector<BaseFloat> frame_weights_;
  CuVector<BaseFloat> target_sum_;

  // loss computation buffers
  CuMatrix<BaseFloat> tgt_mat_;
  CuMatrix<BaseFloat> frames_aux_;
  CuMatrix<BaseFloat> xentropy_aux_;
  CuMatrix<BaseFloat> entropy_aux_;

  // frame classification buffers, 
  CuArray<int32> max_id_out_;
  CuArray<int32> max_id_tgt_;
};


class CBXent {
 public:
	CBXent() : frames_(0.0), correct_(0.0), loss_(0.0), entropy_(0.0), ppl_(0.0), logzt_(0.0), logzt_variance_(0.0),
           frames_progress_(0.0), loss_progress_(0.0), entropy_progress_(0.0),
		   correct_progress_(0.0), ppl_progress_(0.0),
		   logzt_progress_(0.0), logzt_variance_progress_(0.0), var_penalty_(0.0), frame_zt_ptr_(NULL), class_frame_zt_ptr_(NULL) { }
  ~CBXent() { }

  void SetClassBoundary(const std::vector<int32>& class_boundary);

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  void Eval();

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
			const std::vector<int32> &target,
			CuMatrixBase<BaseFloat> *diff);

  void GetTargetWordPosterior(Vector<BaseFloat> &tgt);

  /// Generate string with error report,
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
    return (loss_ - entropy_) / frames_;
  }

  /// Merge statistic data
  void Add(CBXent *xent);
  void Merge(int myid, int root);

  void SetVarPenalty(float var_penalty) {
	  var_penalty_ = var_penalty;
  }

  void SetZt(CuVector<BaseFloat> *frame_zt_ptr, std::vector<CuSubVector<BaseFloat>* > *class_frame_zt_ptr) {
	  frame_zt_ptr_ = frame_zt_ptr;
	  class_frame_zt_ptr_ = class_frame_zt_ptr;
  }

  void SetConstClassZt(const Vector<BaseFloat> &classzt) {
	  int size = classzt.Dim();
	  const_class_zt_.resize(size);
	  for (int i = 0; i < size; i++)
		  const_class_zt_[i] = classzt(i);
  }

  void GetConstZtMean(Vector<BaseFloat> &const_class_zt) {
	  int size = class_zt_mean_.size();
	  const_class_zt.Resize(size);
	  for (int i = 0; i < size; i++)
		  const_class_zt(i) = (class_frames_[i] == 0 ? 0 : class_zt_mean_[i]/class_frames_[i]);
  }

 private:
  double frames_;
  double correct_;
  double loss_;
  double entropy_;
  double ppl_;

  double logzt_;
  double logzt_variance_;

  // partial results during training
  double frames_progress_;
  double loss_progress_;
  double entropy_progress_;
  double correct_progress_;
  double ppl_progress_;
  double logzt_progress_;
  double logzt_variance_progress_;
  std::vector<float> loss_vec_;

  // weigting buffer,
  CuVector<BaseFloat> frame_weights_;
  CuVector<BaseFloat> target_sum_;

  std::vector<int32> class_boundary_;
  std::vector<int32> word2class_;

  // loss computation buffers
  Matrix<BaseFloat> hos_tgt_mat_;
  CuMatrix<BaseFloat> tgt_mat_;
  CuMatrix<BaseFloat> xentropy_aux_;
  CuMatrix<BaseFloat> entropy_aux_;
  CuArray<int32> tgt_id_;

  std::vector<CuSubVector<BaseFloat>* > class_frame_weights_;
  std::vector<CuSubVector<BaseFloat>* > class_target_sum_;

  std::vector<SubMatrix<BaseFloat>* > class_hos_target_;
  std::vector<CuSubMatrix<BaseFloat>* > class_target_;
  std::vector<CuSubMatrix<BaseFloat>* > class_netout_;
  std::vector<CuSubMatrix<BaseFloat>* > class_xentropy_aux_;
  std::vector<CuSubMatrix<BaseFloat>* > class_entropy_aux_;
  std::vector<CuSubMatrix<BaseFloat>* > class_diff_;

  // frame classification buffers,
  CuArray<int32> max_id_out_;
  CuArray<int32> max_id_tgt_;

  // beta penalty of the variance regularization approximation item
  float var_penalty_;

  // constant normalizing
  CuVector<BaseFloat> *frame_zt_ptr_;
  std::vector<CuSubVector<BaseFloat>* > *class_frame_zt_ptr_;
  std::vector<BaseFloat> class_zt_mean_;
  std::vector<BaseFloat> class_zt_variance_;
  std::vector<double> class_frames_;
  // minbatch buffer
  std::vector<int> class_counts_;
  std::vector<int> class_id_;
  // constant class zt
  std::vector<float> const_class_zt_;


#if HAVE_CUDA == 1
  std::vector<cudaStream_t > streamlist_;
#endif
};


class Mse : public LossItf {
 public:
  Mse(LossOptions &opts) : LossItf(opts), frames_(0.0), loss_(0.0),
          frames_progress_(0.0), loss_progress_(0.0) { }
  ~Mse() { }

  /// Evaluate mean square error using target-matrix,
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const CuMatrixBase<BaseFloat>& target,
            CuMatrixBase<BaseFloat>* diff);

  /// Evaluate mean square error using target-posteior,
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const Posterior& target,
            CuMatrix<BaseFloat>* diff);
  
  /// Generate string with error report
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
    return loss_ / frames_;
  }

  /// Merge statistic data
  void Add(LossItf *loss);
  void Merge(int myid, int root);

 private:
  double frames_;
  double loss_;
  
  double frames_progress_;
  double loss_progress_;
  std::vector<float> loss_vec_;

  CuVector<BaseFloat> frame_weights_;
  CuMatrix<BaseFloat> tgt_mat_;
  CuMatrix<BaseFloat> diff_pow_2_;
};


class MultiTaskLoss : public LossItf {
 public:
  MultiTaskLoss(LossOptions &opts):
	  LossItf(opts) { }

  ~MultiTaskLoss() {
    while (loss_vec_.size() > 0) {
      delete loss_vec_.back();
      loss_vec_.pop_back();
    }
  }

  /// Initialize from string, the format for string 's' is :
  /// 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
  ///
  /// Practically it can look like this :
  /// 'multitask,xent,2456,1.0,mse,440,0.001'
  void InitFromString(const std::string& s);

  /// Evaluate mean square error using target-matrix,
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const CuMatrixBase<BaseFloat>& target,
            CuMatrixBase<BaseFloat>* diff) {
    KALDI_ERR << "This is not supposed to be called!";
  }

  /// Evaluate mean square error using target-posteior,
  void Eval(const VectorBase<BaseFloat> &frame_weights, 
            const CuMatrixBase<BaseFloat>& net_out, 
            const Posterior& target,
            CuMatrix<BaseFloat>* diff);
  
  /// Generate string with error report
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss();

  /// Merge statistic data
  void Add(LossItf *loss);
  void Merge(int myid, int root);

 private:
  std::vector<LossItf*>  loss_vec_;
  std::vector<int32>     loss_dim_;
  std::vector<BaseFloat> loss_weights_;
  
  std::vector<int32>     loss_dim_offset_;

  CuMatrix<BaseFloat>    tgt_mat_;
};

class CtcItf {
public:
	CtcItf() : frames_(0), sequences_num_(0), ref_num_(0), error_num_(0), obj_total_(0),
		  frames_progress_(0), ref_num_progress_(0), error_num_progress_(0),
		  sequences_progress_(0), obj_progress_(0.0), report_step_(1000), num_dropped_(0) { }
	virtual ~CtcItf() {};

	/// CTC training over a single sequence from the labels. The errors are returned to [diff]
	virtual void Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, CuMatrix<BaseFloat> *diff) {};

	/// CTC training over multiple sequences. The errors are returned to [diff]
	virtual void EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
					std::vector< std::vector<int32> > &label, CuMatrix<BaseFloat> *diff, Vector<BaseFloat> *ppzx = NULL) {};

	/// Compute token error rate from the softmax-layer activations and the given labels. From the softmax activations,
	/// we get the frame-level labels, by selecting the label with the largest probability at each frame. Then, the frame
	/// -level labels are shrunk by removing the blanks and collasping the repetitions. This gives us the utterance-level
	/// labels, from which we can compute the error rate. The error rate is the Levenshtein distance between the hyp labels
	/// and the given reference label sequence.
	virtual void ErrorRate(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, float* err, std::vector<int32> *hyp);

	/// Compute token error rate over multiple sequences.
	virtual void ErrorRateMSeq(const std::vector<int> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out, std::vector< std::vector<int> > &label);

	/// Set the step of reporting
	void SetReportStep(int32 report_step) { report_step_ = report_step;  }

	/// Generate string with report
	virtual std::string Report();

	float NumErrorTokens() const { return error_num_;}
	int32 NumRefTokens() const { return ref_num_;}

	/// Merge statistic data
	virtual void Add(CtcItf *loss);
	virtual void Merge(int myid, int root);

protected:
	double frames_;                    // total frame number
	int32 sequences_num_;
	double ref_num_;                   // total number of tokens in label sequences
	double error_num_;                 // total number of errors (edit distance between hyp and ref)
	double obj_total_;                       // total optimization objective

	int32 frames_progress_;
	int32 ref_num_progress_;
	float error_num_progress_;

	int32 sequences_progress_;         // registry for the number of sequences
	double obj_progress_;              // registry for the optimization objective

	int32 report_step_;                // report obj and accuracy every so many sequences/utterances
    int32 num_dropped_;
};


class Ctc : public CtcItf {
 public:

  /// CTC training over a single sequence from the labels. The errors are returned to [diff]
  void Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, CuMatrix<BaseFloat> *diff);

  /// CTC training over multiple sequences. The errors are returned to [diff]
  void EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
                    std::vector< std::vector<int32> > &label, CuMatrix<BaseFloat> *diff, Vector<BaseFloat> *ppzx = NULL);

 private:
  std::vector<int32> label_expand_;  // expanded version of the label sequence
  CuMatrix<BaseFloat> alpha_;        // alpha values
  CuMatrix<BaseFloat> beta_;         // beta values
  CuMatrix<BaseFloat> ctc_err_;      // ctc errors
};


/// Baidu warp ctc
class WarpCtc : public CtcItf {
public:
    WarpCtc(int blank_label = 0);

	/// CTC training over a single sequence from the labels. The errors are returned to [diff]
	void Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, CuMatrix<BaseFloat> *diff);

	/// CTC training over multiple sequences. The errors are returned to [diff]
	void EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
					std::vector< std::vector<int32> > &label, CuMatrix<BaseFloat> *diff, Vector<BaseFloat> *ppzx = NULL);

	inline void throw_on_error(ctcStatus_t status, const char* message) {
	    if (status != CTC_STATUS_SUCCESS) {
	        throw std::runtime_error(message + (", stat = " + std::string(ctcGetStatusString(status))));
	    }
	}

private:
	int blank_label_;
	ctcOptions options_;
#if HAVE_CUDA == 1
    cudaStream_t stream_;
#endif
	CuVector<BaseFloat> ctc_workspace_;
    CuMatrix<BaseFloat> net_out_act_;
};

/*
/// Alex Graves 2013 RNNT join network
class WarpRNNT : public CtcItf {
public:
	WarpRNNT();
	WarpRNNT(int maxT, int maxU, int blank_label = 0);

	/// RNNT training over a single sequence from the labels. The errors are returned to [diff]
	void Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, CuMatrix<BaseFloat> *diff);

	/// RNNT training over multiple sequences. The errors are returned to [diff]
	void EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
					std::vector< std::vector<int32> > &label, CuMatrix<BaseFloat> *diff);

	inline void throw_on_error(rnntStatus_t status, const char* message) {
	    if (status != RNNT_STATUS_SUCCESS) {
	        throw std::runtime_error(message + (", stat = " + std::string(rnntGetStatusString(status))));
	    }
	}

	rnntOptions &GetOption() {
		return options_;
	}

private:
	rnntOptions options_;
#if HAVE_CUDA == 1
    cudaStream_t stream_;
#endif
	CuVector<BaseFloat> rnnt_workspace_;
    CuMatrix<BaseFloat> net_out_act_;
};
*/


/// Alex Graves 2012 RNNT add network
class WarpRNNT : public CtcItf {
public:
	WarpRNNT();
	WarpRNNT(int maxT, int maxU, int blank_label = 0);

	/// RNNT training over a single sequence from the labels. The errors are returned to [diff]
	void Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, CuMatrix<BaseFloat> *diff);

	/// RNNT training over multiple sequences. The errors are returned to [diff]
	void EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
					std::vector< std::vector<int32> > &label, CuMatrix<BaseFloat> *diff, Vector<BaseFloat> *ppzx = NULL);

	inline void throw_on_error(rnntStatus_t status, const char* message) {
	    if (status != RNNT_STATUS_SUCCESS) {
	        throw std::runtime_error(message + (", stat = " + std::string(rnntGetStatusString(status))));
	    }
	}

	rnntOptions &GetOption() {
		return options_;
	}

private:
	rnntOptions options_;
#if HAVE_CUDA == 1
    cudaStream_t stream_;
#endif
	CuVector<BaseFloat> rnnt_workspace_;
    CuMatrix<BaseFloat> trans_act_;
    CuMatrix<BaseFloat> pred_act_;
};


/// Fst based calculating denominator gradients in log domain
class Denominator {
public:
	Denominator():
		frames_(0), sequences_num_(0), ref_num_(0), error_num_(0), obj_total_(0),
		frames_progress_(0), ref_num_progress_(0), error_num_progress_(0),
		sequences_progress_(0), obj_progress_(0.0), report_step_(1000), num_dropped_(0),
		den_fst_(NULL), batch_first_(false) {
#if HAVE_CUDA == 1
        stream_ = NULL;
#endif
        }

	Denominator(fst::StdVectorFst *den_fst, bool batch_first = false):
		frames_(0), sequences_num_(0), ref_num_(0), error_num_(0), obj_total_(0),
		frames_progress_(0), ref_num_progress_(0), error_num_progress_(0),
		sequences_progress_(0), obj_progress_(0.0), report_step_(1000), num_dropped_(0),
		den_fst_(den_fst), batch_first_(batch_first) {
#if HAVE_CUDA == 1
        stream_ = NULL;
#endif
		InitalizeFst();
		LoadFstToGPU();
	}

	virtual ~ Denominator() { ReleaseFstFromGPU();}

	/// CRF denominator training over multiple sequences. The errors are returned to [diff]
	void EvalParallel(const std::vector<int> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
			CuMatrix<BaseFloat> *diff, Vector<BaseFloat> *alpha_llk = NULL);

	/// Set the step of reporting
	void SetReportStep(int32 report_step) { report_step_ = report_step;  }

	/// Generate string with report
	virtual std::string Report();

	/// Merge statistic data
	virtual void Add(Denominator *loss);
	virtual void Merge(int myid, int root);

protected:
#if HAVE_CUDA == 1
    cudaStream_t stream_;
#endif
	void InitalizeFst();
	void LoadFstToGPU();
	void ReleaseFstFromGPU();

	double frames_;                    // total frame number
	int32 sequences_num_;
	double ref_num_;                   // total number of tokens in label sequences
	double error_num_;                 // total number of errors (edit distance between hyp and ref)
	double obj_total_;                       // total optimization objective

	int32 frames_progress_;
	int32 ref_num_progress_;
	float error_num_progress_;

	int32 sequences_progress_;         // registry for the number of sequences
	double obj_progress_;              // registry for the optimization objective

	int32 report_step_;                // report obj and accuracy every so many sequences/utterances
    int32 num_dropped_;


    // den fst
	fst::StdVectorFst *den_fst_;
	bool batch_first_;

	// fst
	int num_states_;
	int num_arcs_;
	std::vector<std::vector<int> > alpha_next_;
	std::vector<std::vector<int> > beta_next_;
	std::vector<std::vector<int> > alpha_ilabel_;
	std::vector<std::vector<int> > beta_ilabel_;
	std::vector<std::vector<BaseFloat> > alpha_weight_;
	std::vector<std::vector<BaseFloat> > beta_weight_;
	std::vector<BaseFloat> start_weight_;
	std::vector<BaseFloat> end_weight_;

    std::vector<Transition> transition_alpha_;
    std::vector<Transition> transition_beta_;
    std::vector<IntPair> transition_index_alpha_;
    std::vector<IntPair> transition_index_beta_;

    Transition* cu_transition_alpha_;
    Transition* cu_transition_beta_;
    IntPair* cu_transition_index_alpha_;
    IntPair* cu_transition_index_beta_;
    CuVector<BaseFloat> cu_start_weight_;
    CuVector<BaseFloat> cu_end_weight_;

    CuVector<BaseFloat> costs_alpha_;
    CuVector<BaseFloat> costs_beta_;

	CuVector<BaseFloat> alpha_;
	CuVector<BaseFloat> beta_;
	CuVector<BaseFloat> grad_storage_;
};

class CrfCtc : public CtcItf {
public:
    CrfCtc() {
        ctc_ = new WarpCtc;
        den_ = new Denominator;
    }

	CrfCtc(fst::StdVectorFst *den_fst, BaseFloat lambda, int blank_label, bool batch_first = false);

	virtual ~ CrfCtc() { Destroy(); }

	/// CRFCTC training over multiple sequences. The errors are returned to [diff]
	void EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
					std::vector< std::vector<int32> > &label, Vector<BaseFloat> &path_weight,
					CuMatrix<BaseFloat> *diff, Vector<BaseFloat> *objs = NULL);

	/// Generate string with report
	virtual std::string Report();

	/// Merge statistic data
	virtual void Add(CtcItf *loss);
	virtual void Merge(int myid, int root);

private:
	void Destroy();

	BaseFloat lambda_;
	BaseFloat real_obj_progress_;
	BaseFloat real_obj_total_;

	Vector<BaseFloat> ctc_objs_;
	Vector<BaseFloat> den_objs_;
	Vector<BaseFloat> objs_;

	WarpCtc *ctc_;
	Denominator *den_;
	CuMatrix<BaseFloat> dendiff_;
};

} // namespace nnet0
} // namespace kaldi

#endif

