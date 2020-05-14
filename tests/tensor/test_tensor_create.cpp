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

    Tensor* t0_ref = new Tensor(shape, ptr_ref, DEV_CPU);
    Tensor* t1 = Tensor::zeros(shape);

    ASSERT_TRUE(Tensor::equal2(t0_ref, t1, 10e-0));
    ASSERT_TRUE(Tensor::equal2(t0_ref, t1, 10e-0));
}

