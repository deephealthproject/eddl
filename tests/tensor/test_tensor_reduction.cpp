#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;

//
//TEST(TensorTestSuite, tensor_math_reduction_sum) {
//    // Test #1
//
//    Tensor *t1 = Tensor::ones({3, 2, 4}, DEV_CPU);
//    t1->print();
//
//    auto RD = new ReduceDescriptor(t1, {2}, "sum", false);
//    reduction(RD);
//    RD->O->print();
//}
//
