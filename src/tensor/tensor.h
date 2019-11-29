/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_TENSOR_H
#define EDDL_TENSOR_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <string>
#include <mutex>

#include <Eigen/Dense>


#define DEV_CPU 0

#define DEV_GPU 1000
#define DEV_GPU_0 1000
#define DEV_GPU_1 1001
#define DEV_GPU_2 1002
#define DEV_GPU_3 1003
#define DEV_GPU_4 1004
#define DEV_GPU_5 1005
#define DEV_GPU_6 1006
#define DEV_GPU_7 1007
#define DEV_GPU_8 1008

#define DEV_FPGA 2000
#define DEV_FPGA_0 2000
#define DEV_FPGA_1 2001
#define DEV_FPGA_2 2002
#define DEV_FPGA_3 2003
#define DEV_FPGA_4 2004
#define DEV_FPGA_5 2005
#define DEV_FPGA_6 2006
#define DEV_FPGA_7 2007
#define DEV_FPGA_8 2008

#define MAX_GPUS 8

using namespace std;

// TODO: Remove this. Don't like here
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> MatrixXRMf;
typedef vector<int> tshape;
void msg(string s);
void msg(string s, string s2);


class Tensor {
private:
    // Load methods
    static Tensor* load_from_bin(std::ifstream &ifs);
    static Tensor* load_from_onnx(std::ifstream &ifs);
    static Tensor* load_from_img(const string &filename, string format);

    // Save methods
    void save2bin(std::ofstream &ofs);
    void save2onnx(std::ofstream &ofs);
    void save2img(const string &filename, string format);

public:
    int device;
    int ndim;
    int size;
    vector<int> shape;
    vector<int> stride;

    // Data pointers
    float *ptr;
    Eigen::MatrixXf *ptr2;  // TODO: I don't like. float or eigen, not both

    // Aux variables
    int gpu_device;
    mutex *tsem;  // Multithreading. Tensor semaphore


    // Constructors
    Tensor();
    explicit Tensor(const vector<int> &shape, int dev=DEV_CPU);
    Tensor(const vector<int> &shape, float *fptr, int dev=DEV_CPU);
    Tensor(const vector<int> &shape, Tensor *T);

    // Destructors
    ~Tensor();

    // Copy data
    void toCPU(int dev=DEV_CPU);
    void toGPU(int dev=DEV_GPU);
    Tensor* clone();

    // Resize
    void resize(int b, float *fptr);
    void resize(int b);
    void resize(int b, Tensor *T);

    // Check device
    int isCPU();
    int isGPU();
    int isFPGA();

    // View methods
    void info();
    void print();

    // Core
    vector<int> getShape();
    static int get_mode(string mode);

    // Serialization
    static Tensor* loadfs(std::ifstream &ifs, string format="");
    static Tensor* load(const string& filename, string format="");
    void savefs(std::ofstream &ofs, string format="");
    void save(const string& filename, string format="");

    // ***** Core (in-place) *****************************
    void fill_(float v);
    bool valid_indices(vector<int> indices);
    int get_address_rowmajor(vector<int> indices);
    vector<int> get_indices_rowmajor(int address);
    void reshape_(vector<int> shape);
    float get_(vector<int> indices);
    void set_(vector<int> indices, float value);

    // ***** Core (static) *****************************
    Tensor* permute(vector<int> axis);

    // ************************************************
    // ****** Tensor operations ***********************
    // ************************************************
    // Creation ops ***********************************
    static Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);
    static Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);
    static Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);
    static Tensor* arange(float start, float end, float step=1.0f, int dev=DEV_CPU);
    static Tensor* range(float start, float end, float step=1.0f, int dev=DEV_CPU);
    static Tensor* linspace(float start, float end, int steps=100, int dev=DEV_CPU);
    static Tensor* logspace(float start, float end, int steps=100, float base=10.0f, int dev=DEV_CPU);
    static Tensor* eye(int size, int dev=DEV_CPU);
    static Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);


    // ***** Transformations *****************************
    static void shift(Tensor *A,Tensor *B, vector<int> shift, string mode="constant", float constant=0.0f);
    static void rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center={0,0}, string mode="constant", float constant=0.0f);
    static void scale(Tensor *A, Tensor *B, vector<int> new_shape, string mode="nearest", float constant=0.0f);
    static void flip(Tensor *A, Tensor *B, int axis=0);
    static void crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant=0.0f);
    static void crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, string mode="nearest", float constant=0.0f);
    static void cutout(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant=0.0f);

    // ***** Data augmentation *****************************
    static void shift_random(Tensor *A,Tensor *B, vector<float> factor_x, vector<float> factor_y, string mode="constant", float constant=0.0f);
    static void rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center={0,0}, string mode="constant", float constant=0.0f);
    static void scale_random(Tensor *A, Tensor *B, vector<float> factor, string mode="nearest", float constant=0.0f);
    static void flip_random(Tensor *A, Tensor *B, int axis);
    static void crop_random(Tensor *A, Tensor *B);
    static void crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, string mode="nearest", float constant=0.0f);
    static void cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant=0.0f);

    // Math operations ********************************
    static Tensor* interpolate(float factor1, Tensor *A, float factor2, Tensor *B);

    // Math operations: Pointwise ops (in-place)
    void abs_();
    static Tensor* abs(Tensor *A);

    void acos_();
    static Tensor* acos(Tensor *A);

    void add_(float v);
    void add_(Tensor *A);
    static Tensor* add(Tensor *A, Tensor *B);
    static void add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
    static void add(Tensor *A, Tensor *B, Tensor *C);
    static void inc(Tensor *A, Tensor *B);

    void asin_();
    static Tensor* asin(Tensor *A);

    void atan_();
    static Tensor* atan(Tensor *A);

    void ceil_();
    static Tensor* ceil(Tensor *A);

    void clamp_(float min, float max);
    static Tensor* clamp(Tensor *A, float min, float max);

    void clampmax_(float max);
    static Tensor* clampmax(Tensor *A, float max);

    void clampmin_(float min);
    static Tensor* clampmin(Tensor *A, float min);

    void cos_();
    static Tensor* cos(Tensor *A);

    void cosh_();
    static Tensor* cosh(Tensor *A);

    void inv_();

    void div_(float v);
    static Tensor* div(Tensor *A, float v);
    static void el_div(Tensor *A, Tensor *B, Tensor *C, int incC);

    void exp_();
    static Tensor* exp(Tensor *A);

    void floor_();
    static Tensor* floor(Tensor *A);

    void log_();
    static Tensor* log(Tensor *A);

    void log2_();
    static Tensor* log2(Tensor *A);

    void log10_();
    static Tensor* log10(Tensor *A);

    void logn_(float n);
    static Tensor* logn(Tensor *A, float n);

    float max();
    float min();

    void mod_(float v);
    static Tensor* mod(Tensor *A, float v);

    void mult_(float v);
    static Tensor* mult(Tensor *A, float v);
    static void mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
    static void el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);

    void neg_();
    static Tensor* neg(Tensor *A);

    void normalize_(float min=0.0f, float max=1.0f);
    static Tensor* normalize(Tensor *A, float min=0.0f, float max=1.0f);

    void pow_(float exp);
    static Tensor* pow(Tensor *A, float exp);

    void powb_(float base);
    static Tensor* powb(Tensor *A, float base);

    void reciprocal_();
    static Tensor* reciprocal(Tensor *A);

    void remainder_(float v);
    static Tensor* remainder(Tensor *A, float v);

    void round_();
    static Tensor* round(Tensor *A);

    void rsqrt_();
    static Tensor* rsqrt(Tensor *A);

    void sigmoid_();
    static Tensor* sigmoid(Tensor *A);

    void sign_();
    static Tensor* sign(Tensor *A);
    static void sign(Tensor *A, Tensor *B);

    void sin_();
    static Tensor* sin(Tensor *A);

    void sinh_();
    static Tensor* sinh(Tensor *A);

    void sqr_();
    static Tensor* sqr(Tensor *A);

    void sqrt_();
    static Tensor* sqrt(Tensor *A);

    void sub_(float v);
    static Tensor* sub(Tensor *A, Tensor *B);

    float sum();
//    static Tensor* sum(Tensor *A);
    static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
    static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);

    float sum_abs();
    static Tensor* sum_abs(Tensor *A);

    void tan_();
    static Tensor* tan(Tensor *A);

    void tanh_();
    static Tensor* tanh(Tensor *A);

    void trunc_();
    static Tensor* trunc(Tensor *A);

    // Math operations: Reduction ops
    static void reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
    static void reduceTosum(Tensor *A, Tensor *B, int axis);
    static void reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims, Tensor *C, int incB);
    static void delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims, Tensor *C,int incB);
    static void reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC);
    static void delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C, int incC);

    // Math operations: Comparison ops
    static int eqsize(Tensor *A, Tensor *B);
    static int equal(Tensor *A, Tensor *B, float epsilon=1e-3);

    // Math operations: Other ops
    static int cross(Tensor *A, Tensor *B); // TODO
    static int diag(Tensor *A); // TODO
    static int einsum(string subscripts, Tensor *A); // TODO
    static int flatten(Tensor *A); // TODO
    static int flip(Tensor *A);  // TODO
    static int trace(Tensor *A);
    static int dot(Tensor *A);  // TODO

    // Indexing, Slicing, Joining, Mutating Ops *******
    static void transpose(Tensor *A, Tensor *B, vector<int> dims);
    static void copy(Tensor *A, Tensor *B);
    static void fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);
    Tensor* select(vector<vector<int>> indices);
    static void select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);
    static void deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);
    static void tile(Tensor *A, Tensor *B);

    // Generators (In-place) *************************************
    // Rethink names
    void rand_bernoulli(); // Todo
    void rand_multinomial(); // Todo
    void rand_uniform(float v);
    void rand_signed_uniform(float v);
    void rand_normal(float m, float s, bool fast_math=true);
    void rand_binary(float v);
};



#endif //EDDL_TENSOR_H
