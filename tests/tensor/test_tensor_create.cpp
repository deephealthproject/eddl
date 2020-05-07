#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"

using namespace std;


TEST(TensorTestSuite, tensor_create_zeros){
    // Reference
    float* ptr_ref = new float[2*4]{0.0f, 0.0f, 0.0f, 0.0f,
                                    0.0f, 0.0f, 0.0f, 0.0f};
    vector<int> shape = {2, 4};

    Tensor* t0_ref = new Tensor(shape, ptr_ref);
    Tensor* t1 = Tensor::zeros(shape);

    ASSERT_TRUE(Tensor::equal2(t0_ref, t1, 10e-0));
    ASSERT_TRUE(Tensor::equal2(t0_ref, t1, 10e-0));
}



//
///**
//  *  @brief Create a tensor of the specified shape and fill it with zeros.
//  *
//  *  @param shape  Shape of the tensor to create.
//  *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
//  *  @return     Tensor of the specified shape filled with zeros
//*/
//static Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);
//
///**
//  *  @brief Create a tensor of the specified shape and fill it with ones.
//  *
//  *  @param shape  Shape of the tensor to create.
//  *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
//  *  @return     Tensor of the specified shape filled with ones
//*/
//static Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);
//
///**
//  *  @brief Create a tensor of the specified shape and fill it with a specific value.
//  *
//  *  @param shape  Shape of the tensor to create.
//  *  @param value  Value to use to fill the tensor.
//  *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
//  *  @return     Tensor of the specified shape filled with the value
//*/
//static Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);
//
//static Tensor* arange(float start, float end, float step=1.0f, int dev=DEV_CPU);
//static Tensor* range(float start, float end, float step=1.0f, int dev=DEV_CPU);
//static Tensor* linspace(float start, float end, int steps=100, int dev=DEV_CPU);
//static Tensor* logspace(float start, float end, int steps=100, float base=10.0f, int dev=DEV_CPU);
//static Tensor* geomspace(float start, float end, int steps=100, int dev=DEV_CPU);
//
///**
//  *  @brief
//  *
//  *  @param rows  Number of rows of the tensor.
//  *  @param offset
//  *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
//  *  @return     Tensor of the specified shape filled with the value
//*/
//static Tensor* eye(int rows, int offset=0, int dev=DEV_CPU);
//
///**
//  *  @brief Create a tensor representing the identity matrix. Equivalent to calling function ``eye`` with ``offset = 0``.
//  *
//  *  @param shape  Shape of the tensor to create.
//  *  @param value  Value to use to fill the tensor.
//  *  @param dev    Device to use. The possible values are: ``DEV_CPU`` and ``DEV_GPU``.
//  *  @return     Tensor of the specified shape filled with the value
//*/
//static Tensor* identity(int rows, int dev=DEV_CPU);
//static Tensor* diag(Tensor* A, int k=0, int dev=DEV_CPU);
//static Tensor* randu(const vector<int> &shape, int dev=DEV_CPU);
//static Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);
