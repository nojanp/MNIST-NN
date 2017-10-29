#include <iostream>
#include <cstdio>
#include <armadillo>
#include <vector>
#include "data_loader.h"
#include "neural_net.h"

using namespace std;
using namespace arma;

typedef pair<mat, mat> mpair;


int main() {
        vector <mpair> training_data;
        vector < pair <mat, int> > test_data;
        dataLoader(training_data, test_data);

        int input_size = 784;
        int output_size = 10;

        NeuralNet net({input_size, 30, output_size});


        // ---- TRAIN ----

        int num_epochs = 30;
        int mini_batch_size = 10;
        double learning_rate = 0.5;
        double reg_param = 5.0;
        
        net.train(training_data, num_epochs, mini_batch_size, learning_rate, reg_param);

        // ---- TEST ----

        int num_correct_results = net.accuracy(test_data);
        cout << "Test results: " <<  ((double) num_correct_results) / (double) test_data.size() << endl;



        return 0;
}
