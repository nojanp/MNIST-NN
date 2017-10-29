#ifndef DATA_LOADER_H
#define DATA_LOADE_H

#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

typedef pair <mat, mat> mpair;

constexpr int training_size = 50000;
constexpr int test_size = 10000;
constexpr int input_layer_size = 784;

void tester();
void dataLoader(vector <mpair> &, vector < pair <mat, int> > &);
mat vectorizedResults(int);

#endif
