#include  "data_loader.h"
#include <vector>

using namespace std;
using namespace arma;

typedef pair<mat, mat> mpair;

void dataLoader(vector <mpair> & training_data, vector <pair <mat,int> > & test_data) {
        
        mat tr_d_x(training_size, input_layer_size, fill::zeros);
        mat tr_d_y(training_size, 1, fill::zeros);
        tr_d_x.load("./data/mnist_train_x.csv", csv_ascii);
        tr_d_y.load("./data/mnist_train_y.csv", csv_ascii);

        vector <mat> training_inputs;
        for(int i=0; i < training_size; i++) {
                training_inputs.push_back(tr_d_x.row(i).t());
        }

        vector <mat> training_results;
        for(int i=0; i < training_size; i++) {
                training_results.push_back(vectorizedResults(tr_d_y(i,0)));
        }

        mat te_d_x(test_size, input_layer_size, fill::zeros);
        mat te_d_y(test_size, 1, fill::zeros);
        te_d_x.load("./data/mnist_test_x.csv", csv_ascii);
        te_d_y.load("./data/mnist_test_y.csv", csv_ascii);

        vector <mat> test_inputs;
        for(int i=0; i < test_size; i++) {
        test_inputs.push_back(te_d_x.row(i).t());
        }
        
        for(int i=0; i < training_size; i++) {
                training_data.push_back(mpair(training_inputs[i], training_results[i]));
        }
        
        for(int i=0; i < test_size; i++) {
                test_data.push_back(pair<mat,int> (test_inputs[i], te_d_y(i,0)));
        }
}


mat vectorizedResults(int y) {
        mat e(10, 1, fill::zeros);
        e(y,0) = 1;
        return e;
}


