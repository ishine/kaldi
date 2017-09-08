// nnetbin/nnet-init-sparse-dnn.cc

// Copyright 2017  Sogou (author: Kaituo Xu)

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-affine-transform.h"
#include "nnet/nnet-sparse-affine-transform.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;
    typedef kaldi::float32 float32;

    const char *usage = 
      "Initialize Sparse Neural Network parameters according to dense nnet1.\n"
      "Usage: nnet-init-sparse-dnn [options] <dense-nnet> <sparse-nnet-out>\n"
      "e.g.: nnet-init-sparse-dnn --prune-ratio=\"0.8,0.8,0.8\" "
          "exp/dnn/final.nnet exp/sparse_dnn/sparse_nnet.init\n";

    ParseOptions po(usage);

    bool binary_write = true;
    po.Register("binary", &binary_write, "Write output in binary mode");

    std::string prune_ratio;
    po.Register("prune-ratio", &prune_ratio, "Prune ratio of each layer");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_dense_filename = po.GetArg(1),
        nnet_sparse_filename = po.GetArg(2);

    Nnet nnet_dense;
    nnet_dense.Read(nnet_dense_filename);

    Nnet nnet_sparse;

    std::vector<float32> prune_ratioes;
    kaldi::SplitStringToFloats(prune_ratio, ",", false, &prune_ratioes);
    for (size_t i = 0; i < prune_ratioes.size(); i++) {
      KALDI_LOG << prune_ratioes[i];
    }

    /// Convert (Sparse)AffineTransform to SparseAffineTransform
    for (int32 i = 0, j = 0; i < nnet_dense.NumComponents(); i++) {
      if (nnet_dense.GetComponent(i).GetType() == Component::kAffineTransform ||
          nnet_dense.GetComponent(i).GetType() ==
              Component::kSparseAffineTransform) {
        // Copy dense params
        AffineTransform *dense_comp = 
          dynamic_cast<AffineTransform*>(nnet_dense.GetComponent(i).Copy());
        int32 input_dim = dense_comp->InputDim();
        int32 output_dim = dense_comp->OutputDim();
        SparseAffineTransform *sparse_comp =
            new SparseAffineTransform(input_dim, output_dim);
        Vector<BaseFloat> params(dense_comp->NumParams());
        dense_comp->GetParams(&params);
        sparse_comp->SetParams(params);
        sparse_comp->SetLearnRateCoef(dense_comp->GetLearnRateCoef());
        sparse_comp->SetBiasLearnRateCoef(dense_comp->GetBiasLearnRateCoef());
        // Set prune params
        KALDI_ASSERT(j < prune_ratioes.size());
        sparse_comp->SetPruneRatio(prune_ratioes[j++]);
        sparse_comp->ComputePruneMask();
        // append to nnet
        KALDI_LOG << Component::TypeToMarker(sparse_comp->GetType());
        nnet_sparse.AppendComponentPointer(sparse_comp);
      } else {
        nnet_sparse.AppendComponent(nnet_dense.GetComponent(i));
      }
    }

    nnet_sparse.Write(nnet_sparse_filename, binary_write);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
