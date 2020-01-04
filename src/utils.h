/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_UTILS_H
#define EDDL_UTILS_H

#include <cstdint> // uint64_t
#include <vector>


using namespace std;

void msg(const string& text, const string& title="");

float *get_fmem(long int size, char *str);

char *humanSize(uint64_t bytes);

unsigned long get_free_mem();

string get_extension(string filename);

vector<vector<int>> parse_indices(vector<string> str_indices, const vector<int>& shape);

vector<int> indices2shape(vector<vector<int>> ranges);

int shape2size(vector<int> shape);

vector<int> shape2stride(const vector<int>& shape);

vector<int> permute_shape(const vector<int>& ishape, const vector<int>& dims);

int* permute_indices(const vector<int>& ishape, const vector<int>& dims);

int* ranges2indices(vector<int> ishape, vector<vector<int>> ranges);

bool is_number(const std::string& s);

bool pathExists(const std::string &s);

template<typename T>
string printVector(vector<T> myvector){
    string temp = "";
    for(int i = 0; i<myvector.size()-1; i++){
        temp += to_string(myvector[i]) + ", ";
    }
    temp += to_string(myvector[myvector.size()-1]);
    return temp;
}

#endif //EDDL_UTILS_H
