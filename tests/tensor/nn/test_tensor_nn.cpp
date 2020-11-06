#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;


TEST(TensorTestSuite, tensor_nn_full_softmax){
    // Test #1: Forward
    Tensor* t1 = new Tensor({0.0303,  0.2418, -1.9007,
                             -4.7348, -0.7624, -0.5518}, {2, 3}, DEV_CPU);
    Tensor* t1_ref = new Tensor({0.4201, 0.5190, 0.0609,
                                 0.0084, 0.4438, 0.5478}, {2, 3}, DEV_CPU);
    Tensor* t1_out = Tensor::empty_like(t1);

    tensorNN::FullSoftmax(t1, t1_out);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1_out, 10e-4));

    // Test #1: Backward
    Tensor* t1_delta = new Tensor({1.0f, 2.0f, 3.0f,
                                   3.0f, 2.0f, 1.0f}, {2, 3}, DEV_CPU);
    Tensor* t1_delta_ref = new Tensor({-0.26919939,  0.18641007,  0.08278932,
                                       0.01286397,  0.23942514, -0.25228911}, {2, 3}, DEV_CPU);
    Tensor* t1_parent_delta = Tensor::zeros_like(t1);
    tensorNN::D_FullSoftmax(t1_delta, t1_out, t1_parent_delta);
    ASSERT_TRUE(Tensor::equivalent(t1_delta_ref, t1_parent_delta, 10e-4));


    // Deletes
    delete t1;
    delete t1_ref;
    delete t1_out;
    delete t1_delta;
    delete t1_delta_ref;
    delete t1_parent_delta;


    // Test GPU
#ifdef cGPU
    // Forward
    Tensor* t_cpu_in = Tensor::randn({100, 1000});
    Tensor* t_gpu_in = t_cpu_in->clone(); t_gpu_in->toGPU();

    // Forward: CPU
    Tensor* t_cpu_out = Tensor::empty_like(t_cpu_in);
    tensorNN::FullSoftmax(t_cpu_in, t_cpu_out);

    // Forward: GPU
    Tensor* t_gpu_out = Tensor::empty_like(t_gpu_in);
    tensorNN::FullSoftmax(t_gpu_in, t_gpu_out);


    // Backward: CPU
    Tensor* t_cpu_delta = t_cpu_in;  // Whatever
    Tensor* t_cpu_parent_delta = Tensor::zeros_like(t_cpu_in);
    tensorNN::D_FullSoftmax(t_cpu_delta, t_cpu_out, t_cpu_parent_delta);

    // Backward: GPU
    Tensor* t_gpu_delta = t_gpu_in;  // Whatever
    Tensor* t_gpu_parent_delta = Tensor::zeros_like(t_gpu_in);
    tensorNN::D_FullSoftmax(t_gpu_delta, t_gpu_out, t_gpu_parent_delta);

    t_gpu_out->toCPU();  // Keep in GPU until comparison
    ASSERT_TRUE(Tensor::equivalent(t_cpu_out, t_gpu_out, 10e-4));

    t_gpu_parent_delta->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu_parent_delta, t_gpu_parent_delta, 10e-4));

    delete t_cpu_in;
    delete t_gpu_in;
    delete t_cpu_out;
    delete t_gpu_out;
    delete t_cpu_parent_delta;
    delete t_gpu_parent_delta;
//    delete t_cpu_delta;  // Aliases
//    delete t_gpu_delta;  // Aliases

#endif
}
