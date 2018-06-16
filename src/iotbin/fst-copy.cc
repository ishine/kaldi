#include "util/kaldi-io.h"
#include "util/parse-options.h"

#include "iot/wfst.h"

int main(int argc, char * argv[]) {
  using namespace kaldi;
  using namespace kaldi::iot;

  const char *usage =
    "Copy OpenFst graph into iot format\n"
    "Usage:  fst-copy <model-in> <model-out>\n"
     "e.g.:\n"
     " fst-copy openfst.graph iot.graph\n";
  
  ParseOptions po(usage);

  bool binary = true;
  po.Register("binary", &binary, "Write output in binary mode");

  bool openfst = true;
  po.Register("openfst", &openfst, "input openfst graph");

  std::string itable="";
  po.Register("itable", &itable, "input symbol table");

  std::string otable="";
  po.Register("otable", &otable, "output symbol table");

  po.Read(argc, argv);
  if (po.NumArgs() != 2) {
    po.PrintUsage();
    exit(1);
  }

  std::string in_filename = po.GetArg(1),
  out_filename = po.GetArg(2);

  Wfst *g = new Wfst;

  if (openfst) {
    FILE *fp = fopen(in_filename.c_str(), "r");
    KALDI_ASSERT(fp != NULL);

    FILE *itable_fp = NULL;
    if (itable != "") {
      itable_fp = fopen(itable.c_str(), "r");
    }

    FILE *otable_fp = NULL;
    if (otable != "") {
      otable_fp = fopen(otable.c_str(), "r");
    }

    g->ReadOpenFst(fp, itable_fp, otable_fp);

    if (itable_fp != NULL) {
      fclose(itable_fp);
    }

    if (otable_fp != NULL) {
      fclose(otable_fp);
    }

    fclose(fp);
    
  } else {
    bool binary_read;
    Input ki(in_filename.c_str(), &binary_read);
    g->Read(ki.Stream(), binary_read);
  }

  {
    Output ko(out_filename, binary);
    g->Write(ko.Stream(), binary);
  }

  KALDI_LOG << "num_nodes: " << g->NumStates() << "\n";
  KALDI_LOG << "num_arcs:  " << g->NumArcs() << "\n";
  delete g;

  return 0;
}
