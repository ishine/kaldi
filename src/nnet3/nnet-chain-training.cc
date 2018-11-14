// nnet3/nnet-chain-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)
//                2016    Xiaohui Zhang

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

#include "nnet3/nnet-chain-training.h"
#include "nnet3/nnet-utils.h"
#include "matrix/kaldi-matrix.h"


namespace kaldi {
namespace nnet3 {

NnetChainTrainer::NnetChainTrainer(const NnetChainTrainingOptions &opts,
                                   const fst::StdVectorFst &den_fst,
                                   Nnet *nnet,
								   Nnet *t_nnet):
    opts_(opts),
    den_graph_(den_fst, nnet->OutputDim("output")),
    nnet_(nnet),
	t_nnet_(t_nnet),                                           // T-S
    compiler_(*nnet, opts_.nnet_config.optimize_config,
              opts_.nnet_config.compiler_config),
	t_compiler_(*t_nnet, opts_.nnet_config.optimize_config,     // T-S 
              opts_.nnet_config.compiler_config),
    num_minibatches_processed_(0),
    srand_seed_(RandInt(0, 100000)) {
  if (opts.nnet_config.zero_component_stats)
    ZeroComponentStats(nnet);
  KALDI_ASSERT(opts.nnet_config.momentum >= 0.0 &&
               opts.nnet_config.max_param_change >= 0.0 &&
               opts.nnet_config.backstitch_training_interval > 0);
  delta_nnet_ = nnet_->Copy();
  ScaleNnet(0.0, delta_nnet_);
  t_delta_nnet_ = NULL;
  const int32 num_updatable = NumUpdatableComponents(*delta_nnet_);
  num_max_change_per_component_applied_.resize(num_updatable, 0);
  num_max_change_global_applied_ = 0;

  if (opts.nnet_config.read_cache != "") {
    bool binary;
    try {
      Input ki(opts.nnet_config.read_cache, &binary);
      compiler_.ReadCache(ki.Stream(), binary);
      KALDI_LOG << "Read computation cache from " << opts.nnet_config.read_cache;
    } catch (...) {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  }
}


void NnetChainTrainer::Train(const NnetChainExample &chain_eg) {
  bool need_model_derivative = true;
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  bool use_xent_regularization = (opts_.chain_config.xent_regularize != 0.0);
  ComputationRequest request;
  
  GetChainComputationRequest(*nnet_, chain_eg, need_model_derivative,
                             nnet_config.store_component_stats,
                             use_xent_regularization, need_model_derivative,
                             &request);

  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(request);

  if (nnet_config.backstitch_training_scale > 0.0 && num_minibatches_processed_
      % nnet_config.backstitch_training_interval ==
      srand_seed_ % nnet_config.backstitch_training_interval) {
    // backstitch training is incompatible with momentum > 0
    KALDI_ASSERT(nnet_config.momentum == 0.0);
    FreezeNaturalGradient(true, delta_nnet_);
    bool is_backstitch_step1 = true;
    srand(srand_seed_ + num_minibatches_processed_);
    ResetGenerators(nnet_);
    TrainInternalBackstitch(chain_eg, *computation, is_backstitch_step1);
    FreezeNaturalGradient(false, delta_nnet_); // un-freeze natural gradient
    is_backstitch_step1 = false;
    srand(srand_seed_ + num_minibatches_processed_);
    ResetGenerators(nnet_);
    TrainInternalBackstitch(chain_eg, *computation, is_backstitch_step1);
  } else { // conventional training
    TrainInternal(chain_eg, *computation);
  }
  if (num_minibatches_processed_ == 0) {
    ConsolidateMemory(nnet_);
    ConsolidateMemory(delta_nnet_);
  }
  num_minibatches_processed_++;
}

void NnetChainTrainer::TrainTS(const NnetChainExample &chain_eg) {
  bool need_model_derivative = true;
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  bool use_xent_regularization = (opts_.chain_config.xent_regularize != 0.0);
  ComputationRequest request, t_request;
  
  GetTSChainComputationRequest(*nnet_, *t_nnet_, chain_eg, need_model_derivative,
                             nnet_config.store_component_stats,
                             use_xent_regularization, need_model_derivative,
							 &request, &t_request);  

  std::shared_ptr<const NnetComputation> s_computation = compiler_.Compile(request);
  std::shared_ptr<const NnetComputation> t_computation = t_compiler_.Compile(t_request);

  // now T-S training do not support backstitch, add it manually if needed
  TrainInternalTS(chain_eg, *s_computation, *t_computation);
  if (num_minibatches_processed_ == 0) {
    ConsolidateMemory(nnet_);
    ConsolidateMemory(delta_nnet_);
  }
  num_minibatches_processed_++;
}

void NnetChainTrainer::TrainInternal(const NnetChainExample &eg,
                                     const NnetComputation &computation) {
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);

  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.inputs);
  computer.Run();

  this->ProcessOutputs(false, eg, &computer);
  computer.Run();

  // If relevant, add in the part of the gradient that comes from L2
  // regularization.
  ApplyL2Regularization(*nnet_,
                        GetNumNvalues(eg.inputs, false) *
                        nnet_config.l2_regularize_factor,
                        delta_nnet_);

  // Updates the parameters of nnet
  bool success = UpdateNnetWithMaxChange(*delta_nnet_,
      nnet_config.max_param_change, 1.0, 1.0 - nnet_config.momentum, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);

  // The following will only do something if we have a LinearComponent
  // or AffineComponent with orthonormal-constraint set to a nonzero value.
  ConstrainOrthonormal(nnet_);

  // Scale delta_nnet
  if (success)
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);
}

void NnetChainTrainer::TrainInternalTS(const NnetChainExample &eg,
                                       const NnetComputation &s_computation, const NnetComputation &t_computation) {
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer s_computer(nnet_config.compute_config, s_computation,
                          nnet_, delta_nnet_);
  NnetComputer t_computer(nnet_config.compute_config, t_computation,
						  t_nnet_, t_delta_nnet_);

  // give the inputs to the computer object.
  s_computer.AcceptTSInputs(*nnet_, eg.inputs, false);
  s_computer.Run();

  t_computer.AcceptTSInputs(*t_nnet_, eg.inputs, true);
  t_computer.Run();

  this->ProcessTSOutputs(false, eg, &s_computer, &t_computer);
  s_computer.Run();

  // If relevant, add in the part of the gradient that comes from L2
  // regularization.
  ApplyL2Regularization(*nnet_,
                        GetNumNvalues(eg.inputs, false) *
                        nnet_config.l2_regularize_factor,
                        delta_nnet_);

  // Updates the parameters of nnet
  bool success = UpdateNnetWithMaxChange(*delta_nnet_,
      nnet_config.max_param_change, 1.0, 1.0 - nnet_config.momentum, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  // Scale down the batchnorm stats (keeps them fresh... this affects what
  // happens when we use the model with batchnorm test-mode set).
  ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);

  // The following will only do something if we have a LinearComponent
  // or AffineComponent with orthonormal-constraint set to a nonzero value.
  ConstrainOrthonormal(nnet_);

  // Scale delta_nnet
  if (success)
    ScaleNnet(nnet_config.momentum, delta_nnet_);
  else
    ScaleNnet(0.0, delta_nnet_);
}

void NnetChainTrainer::TrainInternalBackstitch(const NnetChainExample &eg,
                                               const NnetComputation &computation,
                                               bool is_backstitch_step1) {
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  // note: because we give the 1st arg (nnet_) as a pointer to the
  // constructor of 'computer', it will use that copy of the nnet to
  // store stats.
  NnetComputer computer(nnet_config.compute_config, computation,
                        nnet_, delta_nnet_);
  // give the inputs to the computer object.
  computer.AcceptInputs(*nnet_, eg.inputs);
  computer.Run();

  bool is_backstitch_step2 = !is_backstitch_step1;
  this->ProcessOutputs(is_backstitch_step2, eg, &computer);
  computer.Run();

  BaseFloat max_change_scale, scale_adding;
  if (is_backstitch_step1) {
    // max-change is scaled by backstitch_training_scale;
    // delta_nnet is scaled by -backstitch_training_scale when added to nnet;
    max_change_scale = nnet_config.backstitch_training_scale;
    scale_adding = -nnet_config.backstitch_training_scale;
  } else {
    // max-change is scaled by 1 + backstitch_training_scale;
    // delta_nnet is scaled by 1 + backstitch_training_scale when added to nnet;
    max_change_scale = 1.0 + nnet_config.backstitch_training_scale;
    scale_adding = 1.0 + nnet_config.backstitch_training_scale;
    // If relevant, add in the part of the gradient that comes from L2
    // regularization.  It may not be optimally inefficient to do it on both
    // passes of the backstitch, like we do here, but it probably minimizes
    // any harmful interactions with the max-change.
    ApplyL2Regularization(*nnet_,
        1.0 / scale_adding * GetNumNvalues(eg.inputs, false) *
        nnet_config.l2_regularize_factor, delta_nnet_);
  }

  // Updates the parameters of nnet
  UpdateNnetWithMaxChange(*delta_nnet_,
      nnet_config.max_param_change, max_change_scale, scale_adding, nnet_,
      &num_max_change_per_component_applied_, &num_max_change_global_applied_);

  if (is_backstitch_step1) {
    // The following will only do something if we have a LinearComponent or
    // AffineComponent with orthonormal-constraint set to a nonzero value. We
    // choose to do this only on the 1st backstitch step, for efficiency.
    ConstrainOrthonormal(nnet_);
  }

  if (!is_backstitch_step1) {
    // Scale down the batchnorm stats (keeps them fresh... this affects what
    // happens when we use the model with batchnorm test-mode set).  Do this
    // after backstitch step 2 so that the stats are scaled down before we start
    // the next minibatch.
    ScaleBatchnormStats(nnet_config.batchnorm_stats_scale, nnet_);
  }

  ScaleNnet(0.0, delta_nnet_);
}

void NnetChainTrainer::ProcessOutputs(bool is_backstitch_step2,
                                      const NnetChainExample &eg,
                                      NnetComputer *computer) {
  // normally the eg will have just one output named 'output', but
  // we don't assume this.
  // In backstitch training, the output-name with the "_backstitch" suffix is
  // the one computed after the first, backward step of backstitch.
  const std::string suffix = (is_backstitch_step2 ? "_backstitch" : "");
  std::vector<NnetChainSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetChainSupervision &sup = *iter;
    int32 node_index = nnet_->GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_->IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);
    CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                          nnet_output.NumCols(),
                                          kUndefined);

    bool use_xent = (opts_.chain_config.xent_regularize != 0.0);
    std::string xent_name = sup.name + "-xent";  // typically "output-xent".
    CuMatrix<BaseFloat> xent_deriv;

    BaseFloat tot_objf, tot_l2_term, tot_weight;

    ComputeChainObjfAndDeriv(opts_.chain_config, den_graph_,
                             sup.supervision, nnet_output,
                             &tot_objf, &tot_l2_term, &tot_weight,
                             &nnet_output_deriv,
                             (use_xent ? &xent_deriv : NULL));

    if (use_xent) {
      // this block computes the cross-entropy objective.
      const CuMatrixBase<BaseFloat> &xent_output = computer->GetOutput(
          xent_name);
      // at this point, xent_deriv is posteriors derived from the numerator
      // computation.  note, xent_objf has a factor of '.supervision.weight'
      BaseFloat xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);
      objf_info_[xent_name + suffix].UpdateStats(xent_name + suffix,
                                        opts_.nnet_config.print_interval,
                                        num_minibatches_processed_,
                                        tot_weight, xent_objf);
    }

    if (opts_.apply_deriv_weights && sup.deriv_weights.Dim() != 0) {
      CuVector<BaseFloat> cu_deriv_weights(sup.deriv_weights);
      nnet_output_deriv.MulRowsVec(cu_deriv_weights);
      if (use_xent)
        xent_deriv.MulRowsVec(cu_deriv_weights);
    }

    computer->AcceptInput(sup.name, &nnet_output_deriv);

    objf_info_[sup.name + suffix].UpdateStats(sup.name + suffix,
                                     opts_.nnet_config.print_interval,
                                     num_minibatches_processed_,
                                     tot_weight, tot_objf, tot_l2_term);

    if (use_xent) {
      xent_deriv.Scale(opts_.chain_config.xent_regularize);
      computer->AcceptInput(xent_name, &xent_deriv);
    }
  }
}

void NnetChainTrainer::ProcessTSOutputs(bool is_backstitch_step2,
                                        const NnetChainExample &eg,
                                        NnetComputer *s_computer,
									    NnetComputer *t_computer) {
  // normally the eg will have just one output named 'output', but
  // we don't assume this.
  // In backstitch training, the output-name with the "_backstitch" suffix is
  // the one computed after the first, backward step of backstitch.

  const std::string suffix = ("_student");
  std::vector<NnetChainSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetChainSupervision &sup = *iter;
    int32 node_index = nnet_->GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_->IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &s_nnet_output = s_computer->GetOutput(sup.name);
	const CuMatrixBase<BaseFloat> &t_nnet_output = t_computer->GetOutput(sup.name);
	if (s_nnet_output.NumRows() != t_nnet_output.NumRows() || s_nnet_output.NumCols() != t_nnet_output.NumCols())
	  KALDI_ERR << "Outputs of s_net and t_net are not equal"
	            << "NumRows of s_net is " << s_nnet_output.NumRows() << "NumRows of t_net is " << t_nnet_output.NumRows()
				<< "NumCols of s_net is " << s_nnet_output.NumCols() << "NumCols of t_net is " << t_nnet_output.NumCols();

    CuMatrix<BaseFloat> s_nnet_output_deriv(s_nnet_output.NumRows(),
                                            s_nnet_output.NumCols(),
                                            kUndefined);

	CuMatrix<BaseFloat> t_nnet_output_deriv(t_nnet_output.NumRows(),
                                            t_nnet_output.NumCols(),
                                            kUndefined);

    bool use_xent = (opts_.chain_config.xent_regularize != 0.0);
    std::string xent_name = sup.name + "-xent";  // typically "output-xent".
    CuMatrix<BaseFloat> xent_deriv(s_nnet_output.NumRows(),
                                   s_nnet_output.NumCols(),
                                   kUndefined);
	xent_deriv.SetZero();

    BaseFloat teacher_objf, tot_l2_term, tot_weight;
	
	NnetIo io_post = eg.inputs.back();

    ComputeTSObjfAndDeriv(opts_.chain_config, den_graph_,
	                      sup.supervision, io_post.features, 
						  s_nnet_output, t_nnet_output,
                          &teacher_objf, &tot_l2_term, &tot_weight,
                          &s_nnet_output_deriv,
						  &t_nnet_output_deriv,
                          (use_xent ? &xent_deriv : NULL));


/*   // The following code is to test the frame-level alignments
	//To be deleted
	s_nnet_output_deriv.SetZero();
	GeneralMatrix &posterior = io_post.features;
	const SparseMatrix<BaseFloat> &post = posterior.GetSparseMatrix();
    CuSparseMatrix<BaseFloat> cu_post(post);
	cu_post.CopyToMat(&s_nnet_output_deriv);
	//End 
*/

	// The following code using the softmax of teacher-net to teach the student-net (remember that the teacher-net should have softmax layer)
	s_nnet_output_deriv.SetZero();
	s_nnet_output_deriv.AddMat(1.0, t_nnet_output);
	s_nnet_output_deriv.ApplyExp();
//    Matrix<BaseFloat> out_deriv_write(s_nnet_output_deriv);
//    if (num_minibatches_processed_ == 1) {
//	  std::string out_deriv_wspecifier = "ark:out_deriv.ark";
//      BaseFloatMatrixWriter out_deriv_writer(out_deriv_wspecifier);
//	  out_deriv_writer.Write("key", out_deriv_write); 
//	  out_deriv_writer.Flush();
//   }
	//End

    if (use_xent) {
      // this block computes the cross-entropy objective.
      const CuMatrixBase<BaseFloat> &xent_output = s_computer->GetOutput(
          xent_name);
	  BaseFloat xent_objf;
      // at this point, xent_deriv is posteriors derived from the numerator
      // computation.  note, xent_objf has a factor of '.supervision.weight'
	  GeneralMatrix &posterior = io_post.features;
	  switch (io_post.features.Type()) {
        case kSparseMatrix: {
		  const SparseMatrix<BaseFloat> &post = posterior.GetSparseMatrix();
          CuSparseMatrix<BaseFloat> cu_post(post);
          // The cross-entropy objective is computed by a simple dot product,
          // because after the LogSoftmaxLayer, the output is already in the form
          // of log-likelihoods that are normalized to sum to one.
          xent_objf = TraceMatSmat(xent_output, cu_post, kTrans);
          cu_post.CopyToMat(&xent_deriv);

		  Matrix<BaseFloat> xent_out_write(xent_output), xent_deriv_write(xent_deriv),
			out_write(s_nnet_output), out_deriv_write(s_nnet_output_deriv);

		  printf("num_minibatches_processed_ = %d\n", num_minibatches_processed_);
		  if (num_minibatches_processed_ == 1) {
			std::string xent_out_wspecifier = "ark:xent_out.ark", xent_deriv_wspecifier = "ark:xent_deriv.ark",
	           out_wspecifier = "ark:out.ark", out_deriv_wspecifier = "ark:out_deriv.ark";
            BaseFloatMatrixWriter xent_out_writer(xent_out_wspecifier), xent_deriv_writer(xent_deriv_wspecifier),
	           out_writer(out_wspecifier), out_deriv_writer(out_deriv_wspecifier);
            xent_out_write.ApplyExp();
		    out_write.ApplyExp();
		    xent_out_writer.Write("key", xent_out_write);
		    xent_deriv_writer.Write("key", xent_deriv_write);
		    out_writer.Write("key", out_write);
		    out_deriv_writer.Write("key", out_deriv_write); 
			xent_out_writer.Flush();
			xent_deriv_writer.Flush();
			out_writer.Flush();
			out_deriv_writer.Flush();
		  }
//          s_computer->AcceptInput(xent_name, &xent_deriv);
		  break;
        }

        case kFullMatrix: {
          // there is a redundant matrix copy in here if we're not using a GPU
          // but we don't anticipate this code branch being used in many cases.
		  CuMatrix<BaseFloat> cu_post(posterior.GetFullMatrix());
          xent_objf = TraceMatMat(xent_output, cu_post, kTrans);
          xent_deriv.AddMat(1.0, cu_post);
//		  s_computer->AcceptInput(xent_name, &xent_deriv);
          break;
        }
        case kCompressedMatrix: {
          Matrix<BaseFloat> post;
		  posterior.GetMatrix(&post);
          xent_deriv.Swap(&post);
		  xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);
//		  s_computer->AcceptInput(xent_name, &xent_deriv);
          break;
        }
      }
      //BaseFloat xent_objf = TraceMatMat(xent_output, xent_deriv, kTrans);
      objf_info_[xent_name + suffix].UpdateStats(xent_name + suffix,
                                        opts_.nnet_config.print_interval,
                                        num_minibatches_processed_,
                                        tot_weight, xent_objf);
    }

    if (opts_.apply_deriv_weights && sup.deriv_weights.Dim() != 0) {
      CuVector<BaseFloat> cu_deriv_weights(sup.deriv_weights);
      s_nnet_output_deriv.MulRowsVec(cu_deriv_weights);
      if (use_xent)
        xent_deriv.MulRowsVec(cu_deriv_weights);
    }


	BaseFloat student_objf = TraceMatMat(s_nnet_output, s_nnet_output_deriv, kTrans);
	objf_info_[sup.name + "_student"].UpdateStats(sup.name + "_student",
                                        opts_.nnet_config.print_interval,
                                        num_minibatches_processed_,
										tot_weight, student_objf);
/*
	const CuMatrixBase<BaseFloat> &t_xent_output = t_computer->GetOutput(
          xent_name);	      
	BaseFloat xent_objf = TraceMatMat(t_xent_output, s_nnet_output_deriv, kTrans);
    objf_info_[xent_name + "_teacher"].UpdateStats(xent_name + "_teacher",
                                        opts_.nnet_config.print_interval,
                                        num_minibatches_processed_,
                                        tot_weight, xent_objf);
*/

    objf_info_[sup.name + "_teacher"].UpdateStats(sup.name + "_teacher",
                                     opts_.nnet_config.print_interval,
                                     num_minibatches_processed_,
                                     tot_weight, teacher_objf, tot_l2_term);
	s_computer->AcceptInput(sup.name, &s_nnet_output_deriv);

    if (use_xent) {
      xent_deriv.Scale(opts_.chain_config.xent_regularize);
      s_computer->AcceptInput(xent_name, &xent_deriv);
    }
  }
}

bool NnetChainTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo, StringHasher>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const ObjectiveFunctionInfo &info = iter->second;
    ans = info.PrintTotalStats(name) || ans;
  }
  PrintMaxChangeStats();
  return ans;
}

void NnetChainTrainer::PrintMaxChangeStats() const {
  KALDI_ASSERT(delta_nnet_ != NULL);
  const NnetTrainerOptions &nnet_config = opts_.nnet_config;
  int32 i = 0;
  for (int32 c = 0; c < delta_nnet_->NumComponents(); c++) {
    Component *comp = delta_nnet_->GetComponent(c);
    if (comp->Properties() & kUpdatableComponent) {
      UpdatableComponent *uc = dynamic_cast<UpdatableComponent*>(comp);
      if (uc == NULL)
        KALDI_ERR << "Updatable component does not inherit from class "
                  << "UpdatableComponent; change this code.";
      if (num_max_change_per_component_applied_[i] > 0)
        KALDI_LOG << "For " << delta_nnet_->GetComponentName(c)
                  << ", per-component max-change was enforced "
                  << (100.0 * num_max_change_per_component_applied_[i]) /
                     (num_minibatches_processed_ *
                     (nnet_config.backstitch_training_scale == 0.0 ? 1.0 :
                     1.0 + 1.0 / nnet_config.backstitch_training_interval))
                  << " \% of the time.";
      i++;
    }
  }
  if (num_max_change_global_applied_ > 0)
    KALDI_LOG << "The global max-change was enforced "
              << (100.0 * num_max_change_global_applied_) /
                 (num_minibatches_processed_ *
                 (nnet_config.backstitch_training_scale == 0.0 ? 1.0 :
                 1.0 + 1.0 / nnet_config.backstitch_training_interval))
              << " \% of the time.";
}

NnetChainTrainer::~NnetChainTrainer() {
  if (opts_.nnet_config.write_cache != "") {
    Output ko(opts_.nnet_config.write_cache, opts_.nnet_config.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), opts_.nnet_config.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << opts_.nnet_config.write_cache;
  }
  delete delta_nnet_;
}


} // namespace nnet3
} // namespace kaldi
