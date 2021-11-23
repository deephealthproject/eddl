/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_UTILS_H
#define EDDL_UTILS_H

#include <cstdint> // uint64_t
#include <vector>

class AsymmetricPaddingException : public std::exception {
    std::string error_msg; // Error message to show
    std::vector<int> asymmetric_pads; // To store the padding values that have produced the error

public:
    AsymmetricPaddingException(std::string msg, const std::vector<int> pads) : error_msg(msg), asymmetric_pads(pads) {}

    std::vector<int> get_asymmetric_pads() { return asymmetric_pads; }

    const char* what() const noexcept override { return error_msg.c_str(); }
};

using namespace std;

void msg(const string& text, const string& title="");

void set_text_green();
void set_text_red();
void set_text_default();

void * eddl_malloc(size_t size, const string & str_info = "");

void eddl_free(void * ptr);

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

vector<int> expand_shape(const vector<int>& ishape, int size);
int* expand_indices(const vector<int>& ishape, int size);

bool is_number(const std::string& s);

bool pathExists(const std::string &s);

string get_parent_dir(const string& fname);

vector<int> compute_squeeze(vector<int> shape, int axis, bool ignore_batch=false);
vector<int> compute_unsqueeze(vector<int> shape, int axis, bool ignore_batch=false);

void fast_address2indices(unsigned int address, unsigned int* indices, const unsigned int* shape, const unsigned int* strides, unsigned int ndim);
vector<int> address2indices(unsigned int address, const vector<int>& shape, const vector<int>& strides);

// https://isocpp.org/wiki/faq/inline-functions#inline-member-fns
inline unsigned int fast_indices2address(const unsigned int* indices, const unsigned int* strides, unsigned int ndim){
    unsigned int address = 0;
    for (int i=0; i< ndim; i++){
        address += indices[i] * strides[i];
    }
    return address;
}


inline void fast_address2indices(unsigned int address, unsigned int* indices, const unsigned int* shape, const unsigned int* strides, unsigned int ndim){
    for(int i=0; i<ndim; i++) {
        indices[i] = address / strides[i] % shape[i];
    }
}


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

enum TransformationMode {HalfPixel=0, PytorchHalfPixel=1, AlignCorners=2, Asymmetric=3, TFCropAndResize=4};
TransformationMode getTransformationMode(string mode);
string getTransformationModeName(TransformationMode mode);

void __show_profile();

void show_deprecated_warning(const string& deprecated_name, const string& new_name="", const string& type="function", const string& version="future");


#endif //EDDL_UTILS_H
