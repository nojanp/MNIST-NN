#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <iostream>
#include <random>
#include <vector>
#include <armadillo>

using namespace std;
using namespace arma;

typedef pair <mat,mat> mpair;


class NeuralNet {
        public:
                NeuralNet(vector <int> sizes);
                
                void initWeights();

                void feedForward(mat & a);
                
                void train(vector <mpair> & training_data, int num_epochs, int mini_batch_size, double learning_rate, double reg_param);

                void updateMiniBatchIterative(vector <mpair>& mini_batch, double learning_rate, double reg_param, int n);
                
                void updateMiniBatch(vector <mpair>& mini_batch, double learning_rate, double reg_param, int n);

                void backProp(mat & x, mat & y, vector <mat> & nabla_w, vector <mat> & nabla_b, bool is_vectorized);

                int accuracy(vector < pair <mat, int> > & test_data);

        private:
                int num_layers_;
                vector <int> sizes_;
                vector <mat> weights_;
                vector <mat> biases_;
};


void costDerivative(const mat & a, const mat & y, mat & cost_derivative);

void sigmoid(const mat & input, mat & output);
void sigmoidPrime(const mat & input, mat & output);

#endif

