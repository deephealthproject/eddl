#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;


TEST(TensorTestSuite, tensor_nn_full_softmax){
    // Test #1
    Tensor* t1 = new Tensor({0.0303,  0.2418, -1.9007, -4.7348, -0.7624, -0.5518}, {2, 3}, DEV_CPU);
    Tensor* new_t = Tensor::empty_like(t1);

    // Dim= 0
    Tensor* t1_ref = new Tensor({0.4201, 0.5190, 0.0609, 0.0084, 0.4438, 0.5478}, {2, 3}, DEV_CPU);
    tensorNN::FullSoftmax(t1, new_t);

    t1_ref->print(4);
    new_t->print(4);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Delta
    Tensor* new_delta = Tensor::zeros_like(t1);
    Tensor* parent_delta = Tensor::zeros_like(t1);
    tensorNN::D_FullSoftmax(new_delta, new_t, parent_delta);

    new_delta->print(4);

    // Deletes
    delete t1;
    delete t1_ref;
    delete new_t;
    delete new_delta;
    delete parent_delta;
}
