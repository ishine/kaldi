#include <iostream>
#include <fstream>
#include <string>
#include "cnpy.h"

using namespace std;

void WriteData(ofstream &nnet_file, vector<size_t> shape, float *data) {
	// write vector
	if (shape.size() == 2) {
		int row, col;
		row = shape[0];
		col = shape[1];
		nnet_file << " [\n";
		for (int r = 0; r < row; r++) {
			nnet_file << "  ";
			for (int c = 0; c < col; c++) {
				nnet_file << data[r*col + c] << " ";
			}
			if (r != row - 1) {
				nnet_file << "\n";
			} else {
				nnet_file << "]\n";
			}
		}
	} else if (shape.size() == 1) {
        nnet_file << " [ ";
        for (int i = 0; i < shape[0]; ++i) {
            nnet_file << data[i] << " ";
        }
        nnet_file << "]\n";
    }

}


int main(int argc, char *argv[]) {
	cnpy::npz_t npz_file = cnpy::npz_load(argv[1]);
	ofstream nnet_file(argv[2]);

	cnpy::NpyArray arr;
	vector<size_t> shape;

    nnet_file << std::fixed;
	nnet_file << "<Nnet> \n";

	// Write embedding
	arr = npz_file["embedding"];
	shape = arr.shape;
	nnet_file << "<Embedding> " << shape[1] << " 1 \n";
	nnet_file << "<VocabSize> " << shape[0] << " <LearnRateCoef> 1 <BiasLearnRateCoef> 1 \n";
	WriteData(nnet_file, shape, arr.data<float>());
	nnet_file << "<!EndOfComponent> \n";

	// Write LSTM
	int num_layer = npz_file["num_layer"].data<float>()[0];
	cout << "num_layer = " << num_layer << endl;
	for (int l = 0; l < num_layer; ++l) {
		int D, P, H;
		string name;

		name = "layer" + to_string(l) + "/weights_i";
		D = npz_file[name].shape[1];
		H = npz_file[name].shape[0] / 4;
		name = "layer" + to_string(l) + "/weights_r";
		P = npz_file[name].shape[1];

		nnet_file << "<LstmProjected> " << P << " " << D << " \n";
		nnet_file << "<CellDim> " << H << " <LearnRateCoef> 1 <BiasLearnRateCoef> 1 \n";

		name = "layer" + to_string(l) + "/weights_i";
		arr = npz_file[name];
		WriteData(nnet_file, arr.shape, arr.data<float>()); 

		name = "layer" + to_string(l) + "/weights_r";
		arr = npz_file[name];
		WriteData(nnet_file, arr.shape, arr.data<float>()); 


		name = "layer" + to_string(l) + "/biases";
		arr = npz_file[name];
		WriteData(nnet_file, arr.shape, arr.data<float>()); 

		name = "layer" + to_string(l) + "/w_i_diag";
		arr = npz_file[name];
		WriteData(nnet_file, arr.shape, arr.data<float>()); 
		name = "layer" + to_string(l) + "/w_f_diag";
		arr = npz_file[name];
		WriteData(nnet_file, arr.shape, arr.data<float>()); 
		name = "layer" + to_string(l) + "/w_o_diag";
		arr = npz_file[name];
		WriteData(nnet_file, arr.shape, arr.data<float>()); 

		name = "layer" + to_string(l) + "/projection/weights";
		arr = npz_file[name];
		WriteData(nnet_file, arr.shape, arr.data<float>()); 

        nnet_file << "<!EndOfComponent> \n";
	}

    // Write output layer
	arr = npz_file["softmax_w"];
	shape = arr.shape;
    int C = shape[0];
    int D = shape[1];
    nnet_file << "<AffineTransform> " << C << " " << D << " \n";
    nnet_file << "<LearnRateCoef> 1 <BiasLearnRateCoef> 1 \n";
    WriteData(nnet_file, arr.shape, arr.data<float>()); 

	arr = npz_file["softmax_b"];
	shape = arr.shape;
    WriteData(nnet_file, arr.shape, arr.data<float>()); 
    nnet_file << "<!EndOfComponent> \n";

    nnet_file << "<Softmax> " << C << " " << C << " \n";
    nnet_file << "<!EndOfComponent> \n";
	nnet_file << "</Nnet> ";
    return 0;
}
