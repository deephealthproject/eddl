#include <gtest/gtest.h>
#include <random>
#include <string>
#include <ctime>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;


TEST(TensorTestSuite, tensor_math_reduction_sum) {
    // Test #1
    Tensor *t1_ref = new Tensor({4.0f, 4.0f, 4.0f,
                                      4.0f, 4.0f, 4.0f}, {3, 2}, DEV_CPU);
    Tensor *t1 = Tensor::ones({3, 2, 4}, DEV_CPU);
    t1->print();

    Tensor *new_t = t1->sum({2}, false);
    new_t->print();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

}

