/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_TENSOR_H
#define EDDL_TENSOR_H

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <string>
#include <mutex>

#include "Eigen/Dense"

#include "eddl/utils.h"
#include "eddl/descriptors/tensor_descriptors.h"

// Read/Write Numpy
#include "eddl/tensor/cnpy/cnpy.h"

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

class Tensor {
private:
    // Load methods
    static Tensor* load_from_bin(std::ifstream &ifs);
    static Tensor* load_from_onnx(std::ifstream &ifs);
    static Tensor* load_from_img(const string &filename, const string &format);
    template<typename T> static Tensor* load_from_numpy(const string &filename, const string &format);
    static Tensor* load_from_txt(std::ifstream &ifs, char delimiter, int headerRows);

    // Save methods
    void save2bin(std::ofstream &ofs);
    void save2onnx(std::ofstream &ofs);
    void save2img(const string &filename, string format);
    void save2numpy(const string &filename, string format);
    void save2txt(std::ofstream &ofs, const char delimiter, const vector<string> &header);

public:
    int device;
    unsigned int ndim;
    unsigned long int size;
    vector<int> shape;
    vector<int> stride;

    // Data pointers
    float *ptr;
    Eigen::MatrixXf *ptr2;  // TODO: I don't like it. float or eigen, not both

    // Aux variables
    int gpu_device;
    mutex *tsem;  // Multithreading. Tensor semaphore

    // Constructors
    Tensor();
    explicit Tensor(const vector<int> &shape, int dev=DEV_CPU);
    Tensor(const vector<int> &shape, float *fptr, int dev);
    Tensor(const vector<int> &shape, Tensor *T);

    // Destructors
    ~Tensor();

    // Internal methods
    void updateDevice(int dev);
    void updateShape(const vector<int> &new_shape);
    void updateSize();
    void updateStrides();
    void updateData(float* ptr);
    void deleteData();

    /**
      *  @brief Clone a tensor to the CPU.
    */
    void toCPU(int dev=DEV_CPU);

    /**
      *  @brief Clone a tensor to the GPU.
    */
    void toGPU(int dev=DEV_GPU);

    /**
      *  @brief Check if the tensor is in CPU.
      *
      *  @return int
    */
    int isCPU();

    /**
      *  @brief Check if the tensor is in GPU.
      *
      *  @return int
    */
    int isGPU();

    /**
      *  @brief Check if the tensor is in FPGA.
      *
      *  @return int
    */
    int isFPGA();


    /**
      *  @brief Print shape, device and size information.
      *
      *  @return    void
    */
    void info();

    /**
      *  @brief Print the tensor values.
      *
      *  @return    void
    */
    void print(int precision=6, bool raw=false);

    /**
      *  @brief Returns the device name where the tensor is allocated ("CPU", "GPU" or "FPGA")
      *
      *  @return    string
    */
    string getDeviceName();

    // Core
    vector<int> getShape();

    /**
      *  @brief Check if all dimensions in the tensor are the same.
      *
      *  @param A   Tensor
      *  @return    bool
    */
    static bool isSquared(Tensor *A);


    // Serialization *****************************
    /**
      *  @brief Load tensor from filestream.
      *
      *  @param ifs  Filestream
      *  @param format    File format. Accepted formats are: bin, onnx, csv, tsv, txt.
      *  @return    Tensor
    */
    static Tensor* loadfs(std::ifstream &ifs, string format="");

    /**
      *  @brief Load tensor from file.
      *
      *  @param filename  Name of the file to load the tensor from.
      *  @param format    Filetype. The accepted filetypes are the following:
      *                     - Images: jpg, jpeg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm.
      *                     - Numpy: npy, npz
      *                     - Other: bin, onnx
      *  @return    Tensor
    */
    static Tensor* load(const string& filename, string format="");
    template<typename T> static Tensor* load(const string& filename, string format="");

    /**
      *  @brief Load data from a text file
      *
      *  @param filename  Name of the file to load the tensor from.
      *  @param delimiter    Character used to separate the columns of the file.
      *  @param headerRows   Number of top rows to avoid, generally because they correspond to the header.
      *  @return    Tensor
    */
    static Tensor* load_from_txt(const string& filename, const char delimiter=',', int headerRows=1);

    /**
      *  @brief Save tensor to a filestream.
      *
      *  @param ofs     Filestream.
      *  @param format    Format to use. The accepted formats are the following:
      *                     - Text: csv, tsv, txt
      *                     - Other: bin, onnx
      *  @return    void
    */
    void savefs(std::ofstream &ofs, string format="");

    /**
      *  @brief Save tensor to a file.
      *
      *  @param filename    Name of the file to save the tensor to.
      *  @param format    Filetype. The accepted filetypes are the following:
      *                     - Images: png, bmp, tga, jpg, jpeg, hdr.
      *                     - Numpy: npy, npz
      *                     - Text: csv, tsv, txt
      *                     - Other: bin, onnx
      *  @return    void
    */
    void save(const string& filename, string format="");

    /**
      *  @brief Save tensor to a text file.
      *
      *  @param filename    Name of the file to save the tensor to.
      *  @param delimiter   Character to use to separate the columns of the file.
      *  @param header      Header rows.
      *  @return    void
    */
    void save2txt(const string& filename, const char delimiter=',', const vector<string> &header={});

    // ************************************************
    // ****** Tensor operations ***********************
    // ************************************************

    // Creation ops ***********************************

    /**
      *  @brief Create a tensor of the specified shape filled with uninitialized data
      *
      *  @param shape  Shape of the tensor to create.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with zeros
     */
    static Tensor* empty(const vector<int> &shape, int dev=DEV_CPU);


    /**
      *  @brief Create a tensor of the specified shape and filled with zeros.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with zeros
    */
    static Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);

    /**
      *  @brief Create a tensor of the specified shape and filled with ones.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with ones
    */
    static Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);

    /**
      *  @brief Create a tensor of the specified shape and filled with a specific value.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param value  Value to use to fill the tensor.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with the value
    */
    static Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);
    static Tensor* arange(float start, float end, float step=1.0f, int dev=DEV_CPU);
    static Tensor* range(float start, float end, float step=1.0f, int dev=DEV_CPU);
    static Tensor* linspace(float start, float end, int steps=100, int dev=DEV_CPU);
    static Tensor* logspace(float start, float end, int steps=100, float base=10.0f, int dev=DEV_CPU);
    static Tensor* geomspace(float start, float end, int steps=100, int dev=DEV_CPU);

    /**
      *  @brief
      *
      *  @param rows  Number of rows of the tensor.
      *  @param offset
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with the value
    */
    static Tensor* eye(int rows, int offset=0, int dev=DEV_CPU);

    /**
      *  @brief Create a tensor representing the identity matrix. Equivalent to calling function ``eye`` with ``offset = 0``.
      *
      *  @param shape  Shape of the tensor to create.
      *  @param value  Value to use to fill the tensor.
      *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
      *  @return     Tensor of the specified shape filled with the value
    */
    static Tensor* identity(int rows, int dev=DEV_CPU);
    static Tensor* diag(Tensor* A, int k=0, int dev=DEV_CPU);
    static Tensor* randu(const vector<int> &shape, int dev=DEV_CPU);
    static Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);


    // Math operations (zero) ************************
    // TODO: Deprecated? They should be reductions (unless for speed)
    float max();
    static float max(Tensor* A);

    float min();
    static float min(Tensor* A);

    float sum();
    static float sum(Tensor* A);

    float sum_abs();
    static float sum_abs(Tensor* A);


    // Math operations (unary) ************************
    void abs_();
    static void abs(Tensor *A, Tensor *B);

    void acos_();
    static void acos(Tensor *A, Tensor *B);

    void add_(float v);
    void add_(Tensor* A);  // this = this .+ A
    static void add(Tensor *A, Tensor *B, float v); // B = A + v

    void asin_();
    static void asin(Tensor *A, Tensor *B);

    void atan_();
    static void atan(Tensor *A, Tensor *B);

    void ceil_();
    static void ceil(Tensor *A, Tensor *B);

    void clamp_(float min, float max);
    static void clamp(Tensor *A, Tensor *B, float min, float max);

    void clampmax_(float max);
    static void clampmax(Tensor *A, Tensor *B, float max);

    void clampmin_(float min);
    static void clampmin(Tensor *A, Tensor *B, float min);

    void cos_();
    static void cos(Tensor *A, Tensor *B);

    void cosh_();
    static void cosh(Tensor *A, Tensor *B);

    void div_(float v);
    void div_(Tensor* A); // this = this ./ A
    static void div(Tensor *A, Tensor *B, float v); // B = A / v

    void exp_();
    static void exp(Tensor *A, Tensor *B);

    void floor_();
    static void floor(Tensor *A, Tensor *B);

    void inv_(float v=1.0f);
    static void inv(Tensor *A, Tensor *B, float v=1.0f);

    void log_();
    static void log(Tensor *A, Tensor *B);

    void log2_();
    static void log2(Tensor *A, Tensor *B);

    void log10_();
    static void log10(Tensor *A, Tensor *B);

    void logn_(float n);
    static void logn(Tensor *A, Tensor *B, float n);

    void mod_(float v);
    static void mod(Tensor *A, Tensor *B, float v);

    void mult_(float v);
    void mult_(Tensor* A); // this = this .* A
    static void mult(Tensor *A, Tensor *B, float v); // B = A * v

    void neg_();
    static void neg(Tensor *A, Tensor *B);

    void normalize_(float min=0.0f, float max=1.0f);
    static void normalize(Tensor *A, Tensor *B, float min=0.0f, float max=1.0f);

    void pow_(float exp);
    static void pow(Tensor *A, Tensor *B, float exp);

    void powb_(float base);
    static void powb(Tensor *A, Tensor *B, float base);

    void reciprocal_();
    static void reciprocal(Tensor *A, Tensor *B);

    void remainder_(float v);
    static void remainder(Tensor *A, Tensor *B, float v);

    void round_();
    static void round(Tensor *A, Tensor *B);

    void rsqrt_();
    static void rsqrt(Tensor *A, Tensor *B);

    void sigmoid_();
    static void sigmoid(Tensor *A, Tensor *B);

    void sign_(float zero_sign=0.0f);
    static void sign(Tensor *A, Tensor *B, float zero_sign=0.0f);

    void sin_();
    static void sin(Tensor *A, Tensor *B);

    void sinh_();
    static void sinh(Tensor *A, Tensor *B);

    void sqr_();
    static void sqr(Tensor *A, Tensor *B);

    void sqrt_();
    static void sqrt(Tensor *A, Tensor *B);

    void sub_(float v);
    void sub_(Tensor* A); // this = this .- A
    static void sub(Tensor *A, Tensor *B, float v);

    void tan_();
    static void tan(Tensor *A, Tensor *B);

    void tanh_();
    static void tanh(Tensor *A, Tensor *B);

    void trunc_();
    static void trunc(Tensor *A, Tensor *B);


    // Math operations (binary) ************************
    static Tensor* add(Tensor *A, Tensor *B); // (new)C = A + B
    static void add(Tensor *A, Tensor *B, Tensor *C); // C = A + B

    static Tensor* div(Tensor *A, Tensor *B); // (new)C = A / B
    static void div(Tensor *A, Tensor *B, Tensor *C); // C = A / B

    static Tensor* mult(Tensor *A, Tensor *B); // (new)C = A * B
    static void mult(Tensor *A, Tensor *B, Tensor *C); // C = A * B

    static Tensor* interpolate(float factor1, Tensor *A, float factor2, Tensor *B); // (new)C = f1*A + f2*B
    static void interpolate(float factor1, Tensor *A, float factor2, Tensor *B, Tensor *C);  // C = f1*A + f2*B

    static Tensor* sub(Tensor *A, Tensor *B); // (new)C = A - B
    static void sub(Tensor *A, Tensor *B, Tensor *C); // C = A - B



    // ***** Core *****************************
    void fill_(float v);
    static void fill(Tensor* A, float v);

    void permute_(const vector<int>& dims);
    static Tensor* permute(Tensor* A, const vector<int>& dims);

    void moveaxis_(int source, int destination);
    static Tensor* moveaxis(Tensor* A, int source, int destination);

    void swapaxis_(int axis1, int axis2);
    static Tensor* swapaxis(Tensor* A, int axis1, int axis2);

    void reshape_(const vector<int> &new_shape);
    static Tensor* reshape(Tensor *A, const vector<int> &shape);

    void flatten_();
    static Tensor* flatten(Tensor *A);

    void squeeze_();
    static Tensor* squeeze(Tensor *A);

    void unsqueeze_();
    static Tensor* unsqueeze(Tensor *A);


    // ***** Transformations *****************************
    static void shift(Tensor *A,Tensor *B, vector<int> shift, WrappingMode mode=WrappingMode::Constant, float cval=0.0f);
    static void rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center={0,0}, WrappingMode mode=WrappingMode::Constant, float cval=0.0f);
    static void scale(Tensor *A, Tensor *B, vector<int> new_shape, WrappingMode mode=WrappingMode::Nearest, float cval=0.0f);
    static void flip(Tensor *A, Tensor *B, int axis=0);
    static void crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float cval=0.0f);
    static void crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, WrappingMode mode=WrappingMode::Nearest, float cval=0.0f);
    static void cutout(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float cval=0.0f);

    // ***** Data augmentation *****************************
    static void shift_random(Tensor *A,Tensor *B, vector<float> factor_x, vector<float> factor_y, WrappingMode mode=WrappingMode::Constant, float cval=0.0f);
    static void rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center={0,0}, WrappingMode mode=WrappingMode::Constant, float cval=0.0f);
    static void scale_random(Tensor *A, Tensor *B, vector<float> factor, WrappingMode mode=WrappingMode::Nearest, float cval=0.0f);
    static void flip_random(Tensor *A, Tensor *B, int axis);

    static void crop_random(Tensor *A, Tensor *B);
    static void crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, WrappingMode mode=WrappingMode::Nearest, float cval=0.0f);
    static void cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float cval=0.0f);



    // Logic funcions: Truth value testing *****************************

    /**
      *  @brief Test whether all elements evaluate to True.
      *
      *  @param A   Tensor to evaluate
      *  @return    bool
    */
    static bool all(Tensor *A);

    /**
      *  @brief Test whether any element evaluates to True.
      *
      *  @param A   Tensor to evaluate
      *  @return    bool
    */
    static bool any(Tensor *A);

    // Logic funcions: Logical ops

    /**
      *  @brief Test element-wise for finiteness (not infinity or not Not a Number).
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isfinite(Tensor *A, Tensor* B);

    /**
      *  @brief Test element-wise for positive or negative infinity.
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isinf(Tensor *A, Tensor* B);

    /**
      *  @brief Test element-wise for Nan.
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isnan(Tensor *A, Tensor* B);

    /**
      *  @brief Test element-wise for negative infinity.
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isneginf(Tensor *A, Tensor* B);

    /**
      *  @brief Test element-wise for positive infinity.
      *
      *  @param A   Tensor to evaluate
      *  @param B   Tensor to store the results of the test as booleans
      *  @return    void
    */
    static void isposinf(Tensor *A, Tensor* B);

    // Logic funcions: Logical ops

    /**
      *  @brief Compute the truth value of ``A and B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor to store the results of the operation
      *  @return    void
    */
    static void logical_and(Tensor *A, Tensor *B, Tensor *C);

    /**
      *  @brief Compute the truth value of ``A or B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor to store the results of the operation
      *  @return    void
    */
    static void logical_or(Tensor *A, Tensor *B, Tensor *C);

    /**
      *  @brief Compute the truth value of ``not A`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor to store the results of the operation
      *  @return    void
    */
    static void logical_not(Tensor *A, Tensor *B);

    /**
      *  @brief Compute the truth value of ``A xor B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor to store the results of the operation
      *  @return    void
    */
    static void logical_xor(Tensor *A, Tensor *B, Tensor *C);

    // Logic funcions: Comparison ops *****************************

    /**
      *  @brief Returns True if two arrays are element-wise equal within a tolerance.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param rtol
      *  @param atol
      *  @param equal_nan
      *  @return    void
    */
    static bool allclose(Tensor *A, Tensor *B, float rtol=1e-05, float atol=1e-08, bool equal_nan=false);  // Returns true or false

    /**
      *  @brief Returns a boolean array where two arrays are element-wise equal within a tolerance.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor
      *  @param rtol
      *  @param atol
      *  @param equal_nan
      *  @return    void
    */
    static void isclose(Tensor *A, Tensor *B, Tensor *C, float rtol=1e-05, float atol=1e-08, bool equal_nan=false);  // Returns a boolean tensor

    /**
      *  @brief Return the truth value of ``A > B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void greater(Tensor *A, Tensor *B, Tensor *C);

    /**
      *  @brief Return the truth value of ``A >= B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void greater_equal(Tensor *A, Tensor *B, Tensor *C);

    /**
      *  @brief Return the truth value of ``A < B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void less(Tensor *A, Tensor *B, Tensor *C);

    /**
      *  @brief Return the truth value of ``A <= B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void less_equal(Tensor *A, Tensor *B, Tensor *C);

    /**
      *  @brief Return the truth value of ``A == B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void equal(Tensor *A, Tensor *B, Tensor *C);

    /**
      *  @brief Return the truth value of ``A != B`` element-wise.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @param C   Tensor store the results of the operation.
      *  @return    void
    */
    static void not_equal(Tensor *A, Tensor *B, Tensor *C);


    // Math operations: Other ops
    // TODO: cross, diag, einsum, flip, trace, dot, etc


    // Indexing, Slicing, Joining, Mutating Ops *************
    static Tensor* concat(vector<Tensor*> A, unsigned int axis=0, Tensor* output=nullptr);
    static void concat_back(Tensor *A, vector<Tensor*> t, unsigned int axis);

    /**
      *  @brief Returns an array with the selected indices of the tensor.
      *
      *  @param indices  Vector of strings representing the indices to be selected. These indices must follow a Python-like syntax. Some examples: ``"0"`` , ``":5"`` , ``":"`` , ``"3:6"``.
      *  @return     Tensor
    */
    Tensor* select(const vector<string>& indices);
    static void select(Tensor *A, Tensor *B, SelDescriptor *sd);
    static void select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

    /**
      *  @brief Sets the elements in the array using the selected indices. The indices must be specified as a vector of strings ({“0”, “:5”, “:”, “3:6”}).
      *
      *  @param indices  Vector of strings representing the indices to be selected. These indices must follow a Python-like syntax. Some examples: ``"0"``, ``":5"``, ``":"``, ``"3:6"``.
      *  @param A
      *  @return     void
    */
    void set_select(const vector<string>& indices, Tensor *A);
    static void set_select(Tensor *A, Tensor *B, SelDescriptor *sd);
    static void set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

    /**
     *  @brief Clone a tensor (same device). Similar to copy, but returning a new instance
     *
     *  @return    Tensor
    */
    Tensor* clone();

    /**
      *  @brief Reallocates a tensor into this one. Deprecated.
      *
      *  @return
    */
    void reallocate(Tensor* old_t, vector<int> *s = nullptr);

    /**
      *  @brief Resizes a tensor ({2, 2, 2} => {10, 2, 2}).
      *
      *  @return
    */
    void resize(int b, float *fptr=nullptr);


    // ***********************************************************
    // ***********************************************************
    // ***********************************************************
    // ************************ LEGACY ***************************
    // ***********************************************************
    // ***********************************************************
    // ***********************************************************

    // Generators (In-place) *************************************
    // TODO: Rethink names + static
    void rand_bernoulli(); // Todo
    void rand_multinomial(); // Todo
    void rand_uniform(float v);
    void rand_signed_uniform(float v);
    void rand_normal(float m, float s, bool fast_math=true);
    void rand_binary(float v);

    // ***** Overload operators *****************************
    // Tensor and Tensor (Element wise)
    friend Tensor& operator+ (Tensor &A, Tensor &B);
    friend Tensor& operator- (Tensor &A, Tensor &B);
    friend Tensor& operator* (Tensor &A, Tensor &B);
    friend Tensor& operator/ (Tensor &A, Tensor &B);

    // Tensor op= Tensor
    friend void operator+= (Tensor &A, Tensor &B);
    friend void operator-= (Tensor &A, Tensor &B);
    friend void operator*= (Tensor &A, Tensor &B);
    friend void operator/= (Tensor &A, Tensor &B);

    // Tensor op= scalar
    friend void operator+= (Tensor &A, float v);
    friend void operator-= (Tensor &A, float v);
    friend void operator*= (Tensor &A, float v);
    friend void operator/= (Tensor &A, float v);

    // Tensor and scalar
    friend Tensor& operator+ (Tensor &A, float v);
    friend Tensor& operator- (Tensor &A, float v);
    friend Tensor& operator* (Tensor &A, float v);
    friend Tensor& operator/ (Tensor &A, float v);

    // scalar and Tensor
    friend Tensor& operator+ (float v, Tensor &A);
    friend Tensor& operator- (float v, Tensor &A);
    friend Tensor& operator* (float v, Tensor &A);
    friend Tensor& operator/ (float v, Tensor &A);


    // *******************************************
    // Legacy ************************************
    // *******************************************


    /**
      *  @brief Copy data from tensor A to B.
      *
      *  @param A   Tensor
      *  @param B   Tensor
      *  @return    void
    */
    static void copy(Tensor *A, Tensor *B);
    static void fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);
    static void select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end, bool mask_zeros=false);
    static void deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,int inc=0, bool mask_zeros=false);
    static void tile(Tensor *A, Tensor *B);

    // TODO: REFACTOR!!! ************************
    static void transpose(Tensor *A, Tensor *B, vector<int> dims);
    static void add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC); // C = a*A+b*B
    static void inc(Tensor *A, Tensor *B);
    static void el_div(Tensor *A, Tensor *B, Tensor *C, int incC);
    static void el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);
    static void mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
    static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
    static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);
    static void reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
    static int eqsize(Tensor *A, Tensor *B);
    static int equal2(Tensor *A, Tensor *B, float epsilon=1e-3);

};



template<typename T>
Tensor* Tensor::load_from_numpy(const string &filename, const string &format){
    Tensor* t = nullptr;

    cnpy::NpyArray arr = cnpy::npy_load(filename);
    auto* loaded_data = arr.data<T>();

    // Get shape
    vector<int> arr_shape;
    for(unsigned long i : arr.shape){
        arr_shape.push_back(i);
    }

    // Initialize tensor
    t = new Tensor(arr_shape, DEV_CPU);

    // Fill tensor
    for(int i=0; i<arr.num_vals; i++){
        t->ptr[i] = (float)loaded_data[i];
    }

    return t;
}

template<typename T>
Tensor* Tensor::load(const string& filename, string format){
    // Infer format from filename
    if(format.empty()){
        format = get_extension(filename);
    }

    // Check if file exists (open file stream)
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.good()){
        msg("File not found. Check the file name and try again.", "Tensor::load");
    }
    // Load tensor
    Tensor* t;
    if(format=="jpg" || format=="jpeg" || format=="png" || format=="bmp" ||
       format=="hdr" || format=="psd" || format=="tga" || format=="gif" ||
       format=="pic"  || format=="pgm"  || format=="ppm") { // Images
        t = Tensor::load_from_img(filename, format);
    }else if(format=="bin" || format=="onnx"){
        t = Tensor::loadfs(ifs, format);
    }else if(format=="npy" || format=="npz"){
        t = Tensor::load_from_numpy<T>(filename, format);
    }else if(format=="csv" || format=="tsv" || format=="txt"){
        t = Tensor::loadfs(ifs, format);
    }else{
        msg("Format not implemented: *.'" + format + "'", "Tensor::load");
    }

    // Close file stream and return tensor
    ifs.close();
    return t;
}



#endif //EDDL_TENSOR_H
