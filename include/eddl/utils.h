/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_UTILS_H
#define EDDL_UTILS_H

#include <cstdint> // uint64_t
#include <vector>


using namespace std;

void msg(const string& text, const string& title="");

float *get_fmem(unsigned long int size, const string &str);

string bytes2human(unsigned long long int bytes, int decimals=2);

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

string get_parent_dir(const string& fname);

template<typename T>
string printVector(vector<T> myvector){
    string temp = "";
    for(int i = 0; i<myvector.size()-1; i++){
        temp += to_string(myvector[i]) + ", ";
    }
    temp += to_string(myvector[myvector.size()-1]);
    return temp;
}

enum WrappingMode {Constant=0, Reflect=1, Nearest=2, Mirror=3, Wrap=4, Original=5};
WrappingMode getWrappingMode(string mode);

void __show_profile();

void show_deprecated_warning(const string& deprecated_name, const string& new_name="", const string& type="function", const string& version="future");

#endif //EDDL_UTILS_H
