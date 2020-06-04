#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;


TEST(TensorTestSuite, tensor_indexing_nonzero){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0, 1, 2, 4};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {5};
    vector<float> d_t1 = {1, 1, 1, 0, 1};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->nonzero(true);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
}


TEST(TensorTestSuite, tensor_indexing_where){
    // Test #1
    vector<int> t1_shape_ref = {3, 2};
    vector<float> d_t1_ref = {1.0000,  0.3139,
                              0.3898,  1.0000,
                              0.0478,  1.0000};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape_A = {3, 2};
    vector<float> d_t1_A = {-0.4620,  0.3139,
                            0.3898, -0.7197,
                            0.0478, -0.1657};
    Tensor* t1_A = new Tensor(t1_shape_A, d_t1_A.data(), DEV_CPU);


    Tensor* t1_B = Tensor::ones({3, 2}, DEV_CPU);

    Tensor* condition = t1_A->greater(0);

    Tensor* new_t = Tensor::where(condition, t1_A, t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
}
