#include "neural_net.h"

using namespace std;
using namespace arma;

typedef pair<mat, mat> mpair;

NeuralNet::NeuralNet(vector <int> sizes) {
        sizes_ = sizes;
        num_layers_ = sizes_.size();
        initWeights();
}

void NeuralNet::initWeights() {
        srand(time(0));
        arma_rng::set_seed_random();  // set the seed to a random value
        for(int i=0; i < num_layers_-1; i++) {
               weights_.push_back(randn(sizes_[i+1], sizes_[i])); 
        }
        for(int i=1; i < num_layers_; i++) {
                biases_.push_back(randn(sizes_[i], 1));
        }
}

void NeuralNet::feedForward(mat & a) {
        for(int i=0; i < num_layers_-1; i++) {
                sigmoid(weights_[i]*a+biases_[i], a);
        //        cout << size(a) << endl;
        }
}


void NeuralNet::train(vector <mpair> & training_data, int num_epochs, int mini_batch_size, double learning_rate, double reg_param) {
//        int stop_thr = 10; // early stopping threshhold
        double decay = 0.9; // learning rate decay

        int n = training_data.size();

        for(int i=0; i < num_epochs; i++) {
                random_shuffle(training_data.begin(), training_data.end());
                vector <mpair> mini_batch;
                mini_batch.clear();
                for(int k=0; k < training_data.size(); k++) {
                        mini_batch.push_back(training_data[k]);
                        if(mini_batch.size() == mini_batch_size || 
                                        (k == training_data.size()-1 && !mini_batch.empty()) ) {
                                        //updateMiniBatchIterative(mini_batch, learning_rate, reg_param, n);
                                        updateMiniBatch(mini_batch, learning_rate, reg_param, n);
                                        mini_batch.clear();
                                        //cout << "minibatch" << endl;
                        }
                }
                learning_rate*=decay;
                cout << "Epoch {" << i << "} complete" << endl;
        }
}

void NeuralNet::updateMiniBatchIterative(vector <mpair> & mini_batch, double learning_rate, double reg_param, int n) {
        vector <mat> nabla_w, nabla_b;
        for(int i=0; i < num_layers_-1; i++) {
                nabla_w.push_back(zeros(arma::size(weights_[i])));
                nabla_b.push_back(zeros(arma::size(biases_[i])));
        }
        for(int i=0; i < mini_batch.size(); i++) {
                vector <mat> delta_nabla_w, delta_nabla_b;
                backProp(mini_batch[i].first, mini_batch[i].second, delta_nabla_w, delta_nabla_b, false);
                for(int i=0; i < nabla_w.size(); i++) {
                        nabla_w[i] = nabla_w[i] + delta_nabla_w[i];
                        nabla_b[i] = nabla_b[i] + delta_nabla_b[i];
                }
        }
        for(int i=0; i < num_layers_-1; i++) {
                weights_[i] = (1-learning_rate * (reg_param/n)) * weights_[i] - (learning_rate / mini_batch.size()) * nabla_w[i];
                biases_[i] = biases_[i] - (learning_rate / mini_batch.size()) * nabla_b[i];
        }
}

void NeuralNet::updateMiniBatch(vector <mpair> & mini_batch, double learning_rate, double reg_param, int n) {
        vector <mat> nabla_w, nabla_b;
        for(int i=0; i < num_layers_-1; i++) {
                nabla_w.push_back(zeros(arma::size(weights_[i])));
                nabla_b.push_back(zeros(arma::size(biases_[i])));
        }

        mat x_batch;
        mat y_batch;

        for(int i=0; i < mini_batch.size(); i++) {
                x_batch.insert_cols(i, mini_batch[i].first);
                y_batch.insert_cols(i, mini_batch[i].second);
        }

        backProp(x_batch, y_batch, nabla_w, nabla_b, true);

        for(int i=0; i < num_layers_-1; i++) {
                weights_[i] = (1-learning_rate * (reg_param/n)) * weights_[i] - (learning_rate / mini_batch.size()) * nabla_w[i];
                biases_[i] = biases_[i] - (learning_rate / mini_batch.size()) * nabla_b[i];
        }
}

void NeuralNet::backProp(mat & x, mat & y, vector <mat> & nabla_w, vector <mat> & nabla_b, bool is_vectorized) {
        for(int i=0; i < num_layers_-1; i++) {
                nabla_w.push_back(zeros(arma::size(weights_[i])));
                nabla_b.push_back(zeros(arma::size(biases_[i])));
        }

        // forward pass
        mat a = x;
        vector <mat> activations;
        activations.push_back(a);
        vector <mat> weighted_inputs;

        // feedforward layer by layer
        for(int i=0; i < num_layers_-1; i++) {
                mat z;
                if(is_vectorized) {
                        z = weights_[i]*a + repmat(biases_[i],1,a.n_cols);
                }
                else {
                        z = weights_[i]*a + biases_[i];
                }
                sigmoid(z, a);
                weighted_inputs.push_back(z);
                activations.push_back(a);
        }

        // backward pass
        //compute the output error
        mat delta;
        costDerivative(activations[num_layers_-1], y, delta);

        nabla_w[num_layers_-2] = delta * activations[num_layers_-2].t();
        nabla_b[num_layers_-2] = sum(delta, 1);

        for(int i = num_layers_-3; i >= 0; i--) {
                mat z = weighted_inputs[i];
                mat sigmoid_prime_mult_term = zeros(arma::size(a));
                sigmoidPrime(z, sigmoid_prime_mult_term);
                delta = (weights_[i+1].t() * delta) % sigmoid_prime_mult_term;
                
                nabla_w[i] = delta * activations[i].t();
                nabla_b[i] = sum(delta, 1);
        }
}

int NeuralNet::accuracy(vector < pair <mat, int> > & test_data) {
        int correct_results_count = 0;
        for(int i=0; i < test_data.size(); i++) {
               feedForward(test_data[i].first);
                if(test_data[i].first.index_max() == test_data[i].second) {
                        correct_results_count++;
                }
        }
        return correct_results_count;
}




/*
costFn(const mat & a, const mat & y, mat & cost_fn) {
        //
}
*/

void costDerivative(const mat & a, const mat & y, mat & cost_derivative) {
        cost_derivative = a - y;
}

void sigmoid(const mat& input, mat& output) {
        output = exp(-input);
        output.transform([](double val){return (1.0/(1.0 + val));});
        //cout << "asdf" << endl;
        //return output;
}

void sigmoidPrime(const mat& input, mat & output) {
        output = exp(-input);
        output.transform([](double val){return (1.0/(1.0 + val));});
        output.transform([](double val){return (val * (1.0 - val));});
//        return output;
}


