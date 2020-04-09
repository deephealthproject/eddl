/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef _EDDLT_
#define _EDDLT_


#include "tensor/tensor.h"
#include "tensor/tensor_reduction.h"

typedef Tensor* tensor;
typedef vector<int> tshape;

namespace eddlT{

    // Creation ops ***********************************
    Tensor* create(const vector<int> &shape);
    Tensor* create(const vector<int> &shape, int dev);
    Tensor* create(const vector<int> &shape, float *ptr);
    Tensor* create(const vector<int> &shape, float *ptr, int dev);


    /**
    *   @brief Create a tensor of the specified shape and fill it with zeros.
    *
    *   @param shape Shape of the tensor to create.
    *   @param dev  Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
    *   @return The zero-initialized tensor
    */
    Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);

    /**
    *   @brief Create a tensor of the specified shape and fill it with ones.
    *
    *   @param shape Shape of the tensor to create.
    *   @param dev  Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
    *   @return The ones-initialized tensor
    */
    Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);

    /**
    *   @brief Create a tensor of the specified shape and fill it with a value.
    *
    *   @param shape Shape of the tensor to create.
    *   @param value  Value to fill the tensor with.
    *   @param dev  Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
    *   @return The initialized tensor
    */
    Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);

    
    Tensor* arange(float start, float end, float step, int dev=DEV_CPU);
    Tensor* range(float start, float end, float step, int dev=DEV_CPU);
    Tensor* linspace(float start, float end, int steps, int dev=DEV_CPU);
    Tensor* logspace(float start, float end, int steps, float base, int dev=DEV_CPU);
    Tensor* eye(int size, int dev=DEV_CPU);
    Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);


    // Copy data     c   ********************************
    void toCPU_(Tensor *A);
    void toGPU_(Tensor *A);
    Tensor * toCPU(Tensor *A);
    Tensor * toGPU(Tensor *A);
    Tensor* clone(Tensor *A);
    Tensor* select(Tensor *A, int i);
    void copyTensor(Tensor *A,Tensor *B);

    // Core inplace    **********************************
    void fill_(Tensor *A,float v);
    void set_(Tensor *A,vector<int> indices, float value);
    void reshape_(Tensor *A, vector<int> indices);

    // Pointer functions ********************************
    float *getptr(Tensor *A);

    // Other functions   ********************************
    void print(Tensor *A);
    void info(Tensor *A);
    tshape getShape(Tensor *A);

    // Serialization ***********************************
    Tensor* load(string fname, string format="");

    template<typename T>
    Tensor* load_from_numpy(const string &filename, const string &format=""){
        return Tensor::load_from_numpy<T>(filename, format);
    }
    template<typename T>
    Tensor* load(const string& filename, string format=""){
        return Tensor::load<T>(filename, format);
    }

    void save(Tensor* A, string fname, string format="");

    // Math ops       ***********************************

    void abs_(Tensor *A);
    Tensor* abs(Tensor *A);

    void acos_(Tensor *A);
    Tensor* acos(Tensor *A);

    void add_(Tensor *A,float v);
    Tensor *add(Tensor *A,float v);
    void add_(Tensor *A,Tensor *B);
    Tensor* add(Tensor *A, Tensor *B);



    void asin_(Tensor *A);
    Tensor* asin(Tensor *A);

    void atan_(Tensor *A);
    Tensor* atan(Tensor *A);

    void ceil_(Tensor *A);
    Tensor* ceil(Tensor *A);

    void clamp_(Tensor *A,float min, float max);
    Tensor* clamp(Tensor *A, float min, float max);

    void clampmax_(Tensor *A,float max);
    Tensor* clampmax(Tensor *A, float max);

    void clampmin_(Tensor *A,float min);
    Tensor* clampmin(Tensor *A, float min);

    void cos_(Tensor *A);
    Tensor* cos(Tensor *A);

    void cosh_(Tensor *A);
    Tensor* cosh(Tensor *A);

    void inv_(Tensor *A);

    void div_(Tensor *A,float v);
    Tensor* div(Tensor *A, float v);
    void div_(Tensor *A, Tensor *B);
    Tensor *div(Tensor *A, Tensor *B);

    void exp_(Tensor *A);
    Tensor* exp(Tensor *A);

    void floor_(Tensor *A);
    Tensor* floor(Tensor *A);

    void inc_(Tensor *A,Tensor *B);

    void log_(Tensor *A);
    Tensor* log(Tensor *A);

    void log2_(Tensor *A);
    Tensor* log2(Tensor *A);

    void log10_(Tensor *A);
    Tensor* log10(Tensor *A);

    void logn_(Tensor *A,float n);
    Tensor* logn(Tensor *A, float n);

    float max(Tensor *A);
    float min(Tensor *A);

    void mod_(Tensor *A,float v);
    Tensor* mod(Tensor *A, float v);

    void mult_(Tensor *A,float v);
    Tensor* mult(Tensor *A, float v);
    void mult_(Tensor *A, Tensor *B);
    Tensor *mult(Tensor *A, Tensor *B);
    Tensor *mult2D(Tensor *A, Tensor *B);

    void neg_(Tensor *A);
    Tensor* neg(Tensor *A);

    void normalize_(Tensor *A,float min, float max);
    Tensor* normalize(Tensor *A, float min, float max);

    void pow_(Tensor *A,float exp);
    Tensor* pow(Tensor *A, float exp);

    void reciprocal_(Tensor *A);
    Tensor* reciprocal(Tensor *A);

    void remainder_(Tensor *A,float v);
    Tensor* remainder(Tensor *A, float v);

    void round_(Tensor *A);
    Tensor* round(Tensor *A);

    void rsqrt_(Tensor *A);
    Tensor* rsqrt(Tensor *A);

    void sigmoid_(Tensor *A);
    Tensor* sigmoid(Tensor *A);

    void sign_(Tensor *A);
    Tensor* sign(Tensor *A);


    void sin_(Tensor *A);
    Tensor* sin(Tensor *A);

    void sinh_(Tensor *A);
    Tensor* sinh(Tensor *A);

    void sqr_(Tensor *A);
    Tensor* sqr(Tensor *A);

    void sqrt_(Tensor *A);
    Tensor* sqrt(Tensor *A);

    void sub_(Tensor *A,float v);
    Tensor* sub(Tensor *A, float v);
    void sub_(Tensor *A, Tensor *B);
    Tensor *sub(Tensor *A, Tensor *B);

    float sum(Tensor *A);
    float sum_abs(Tensor *A);

    void tan_(Tensor *A);
    Tensor* tan(Tensor *A);

    void tanh_(Tensor *A);
    Tensor* tanh(Tensor *A);

    void trunc_(Tensor *A);
    Tensor* trunc(Tensor *A);

    //reductions
    tensor reduce_mean(tensor A,vector<int> axis);
    tensor reduce_variance(tensor A,vector<int> axis);



}

#endif
