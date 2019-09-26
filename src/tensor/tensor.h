
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


#ifndef EDDL_TENSOR_H
#define EDDL_TENSOR_H

#include <stdio.h>
#include <vector>
#include <string>
#include <mutex>

#include <Eigen/Dense>
#include "../descriptors/descriptors.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

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

public:
    int device;
    int ndim;
    int size;
    vector<int> shape;
    vector<int> stride;
    float *ptr;

    // Aux stuff
    Eigen::MatrixXf *ptr2;
    int gpu_device;
    mutex *tsem;  // Multithreading. Tensor semaphore

    // Constructors
    Tensor();
    explicit Tensor(const vector<int> &shape, int dev=DEV_CPU);
    Tensor(const vector<int> &shape, float *fptr, int dev=DEV_CPU);
    Tensor(const vector<int> &shape, Tensor *T);

    // Destructors
    ~Tensor();

    // Check device
    int isCPU();
    int isGPU();
    int isFPGA();

    // View methods
    void info();
    void print();

    // Core
    int numel();
    vector<int> getShape();
    void point2data(const vector<int>& shape, float *ptr);
    void copydata(const vector<int>& s, float *newptr);
    void set(float v);

    // Serialization
    void save(string s);
    void save(FILE *fe);
    void load(FILE *fe);

    // ************************************************
    // ****** Tensor operations ***********************
    // ************************************************

    // Creation ops ***********************************
    static Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);
    static Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);
    static Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);
    static Tensor* arange(float min, float max, float step=1.0, int dev=DEV_CPU);
    static Tensor* range(float min, float max, float step=1.0, int dev=DEV_CPU);
    static Tensor* linspace(float start, float end, int steps=100, int dev=DEV_CPU);
    static Tensor* logspace(float start, float end, int steps=100, float base=10.0, int dev=DEV_CPU);
    static Tensor* eye(int size, int dev=DEV_CPU);
    static Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);

    // Math operations ********************************
    // Math operations: Pointwise ops (in-place)
    void abs_();
    static Tensor* abs(Tensor *A);

    void acos_(); // Todo
    static Tensor* acos(Tensor *A);

    void add_(float v);
    static Tensor* add(Tensor *A, Tensor *B);
    static void add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
    static void add(Tensor *A, Tensor *B, Tensor *C);
    static void inc(Tensor *A, Tensor *B);

    void asin_(); // Todo
    static Tensor* asin(Tensor *A);

    void atan_(); // Todo
    static Tensor* atan(Tensor *A);

    void ceil_(); // Todo
    static Tensor* ceil(Tensor *A);

    void clamp_(); // Todo
    static Tensor* clamp(Tensor *A);

    void cos_(); // Todo
    static Tensor* cos(Tensor *A);

    void cosh_(); // Todo
    static Tensor* cosh(Tensor *A);

    void div_(float v);
    static Tensor* div(Tensor *A);
    static void el_div(Tensor *A, Tensor *B, Tensor *C, int incC);

    void exp_();
    static Tensor* exp(Tensor *A);

    void floor_(); // Todo
    static Tensor* floor(Tensor *A);

    void log_();
    static Tensor* log(Tensor *A);

    void log2_();
    static Tensor* log2(Tensor *A);

    void log10_();
    static Tensor* log10(Tensor *A);

    void logn_(float n);
    static Tensor* logn(Tensor *A);

    void mod_(); // Todo
    static Tensor* mod(Tensor *A);

    void mult_(float v);
    static Tensor* mult(Tensor *A);
    static void mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
    static void el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);

    void neg_(); // Todo
    static Tensor* neg(Tensor *A);

    void pow_(float exp);
    static Tensor* pow(Tensor *A);

    void reciprocal_(); // Todo
    static Tensor* reciprocal(Tensor *A);

    void remainder_(); // Todo
    static Tensor* remainder(Tensor *A);

    void round_(); // Todo
    static Tensor* round(Tensor *A);

    void rsqrt_(); // Todo
    static Tensor* rsqrt(Tensor *A);

    void sigmoid_(); // Todo
    static Tensor* sigmoid(Tensor *A);

    void sign_(); // Todo
    static Tensor* sign(Tensor *A);
    static void sign(Tensor *A, Tensor *B);

    void sin_(); // Todo
    static Tensor* sin(Tensor *A);

    void sinh_(); // Todo
    static Tensor* sinh(Tensor *A);

    void sqr_();
    static Tensor* sqr(Tensor *A);

    void sqrt_();
    static Tensor* sqrt(Tensor *A);

    void sub_(float v);
    static Tensor* sub(Tensor *A, Tensor *B);

    float sum_();
    static Tensor* sum(Tensor *A);
    static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
    static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);

    float sum_abs_();
    static Tensor* sum_abs(Tensor *A);

    void tan_(); // Todo
    static Tensor* tan(Tensor *A);

    void tanh_(); // Todo
    static Tensor* tanh(Tensor *A);

    // Math operations: Reduction ops
    static void reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
    static void reduceTosum(Tensor *A, Tensor *B, int axis);
    static void reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims, Tensor *C, int incB);
    static void delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims, Tensor *C,int incB);
    static void reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC);
    static void delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C,int incC);

    // Math operations: Comparison ops
    static int eqsize(Tensor *A, Tensor *B);
    static int equal(Tensor *A, Tensor *B);

    // Math operations: Other ops
    static int cross(Tensor *A, Tensor *B); // TODO
    static int diag(Tensor *A); // TODO
    static int einsum(string subscripts, Tensor *A); // TODO
    static int flatten(Tensor *A); // TODO
    static int flip(Tensor *A);  // TODO
    static int trace(Tensor *A);  // TODO
    static int dot(Tensor *A);  // TODO

    // Indexing, Slicing, Joining, Mutating Ops *******
    static void transpose(Tensor *A, Tensor *B, vector<int> dims);
    static void copy(Tensor *A, Tensor *B);
    static void fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);
    static void select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);

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

