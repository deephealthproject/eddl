#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;


TEST(TensorTestSuite, tensor_math_sort) {
    // Test #1
    Tensor *t1_ref = new Tensor({-0.5792, -0.1372,  0.5962,  1.2097},
                                {4}, DEV_CPU);
    Tensor *t1 = new Tensor({1.2097, -0.1372, -0.5792,  0.5962},
                            {4}, DEV_CPU);

    Tensor *new_t = t1->sort(false, true);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({1.2097, 0.5962, -0.1372, -0.5792,},
                                {4}, DEV_CPU);

    new_t = t1->sort(true, true);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t, 10e-4));


    // Test GPU
#ifdef cGPU
    // Test #1
    Tensor* t1_cpu = Tensor::randu({10000});
    Tensor* t1_gpu = t1_cpu->clone(); t1_gpu->toGPU();
    t1_cpu->sort_(false, true);
    t1_gpu->sort_(false, true); t1_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t1_cpu, t1_gpu, 10e-4));

    // Test #2
    Tensor* t2_cpu = Tensor::randu({10000});
    Tensor* t2_gpu = t2_cpu->clone(); t2_gpu->toGPU();
    t2_cpu->sort_(true, false);
    t2_gpu->sort_(true, false); t2_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t2_cpu, t2_gpu, 10e-4));
#endif
}




TEST(TensorTestSuite, tensor_math_argsort) {
    // Test #1
    Tensor *t1_ref = new Tensor({2, 1, 3, 0},{4}, DEV_CPU);
    Tensor *t1 = new Tensor({1.2097, -0.1372, -0.5792,  0.5962},
                            {4}, DEV_CPU);

    Tensor *new_t = t1->argsort(false, true);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({0, 3, 1, 2}, {4}, DEV_CPU);

    new_t = t1->argsort(true, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t, 10e-4));


    // Test GPU
#ifdef cGPU
    // Test #1
    Tensor* t1_cpu = Tensor::randn({10000});
    Tensor* t1_gpu = t1_cpu->clone(); t1_gpu->toGPU();
    t1_cpu = t1_cpu->argsort(false, true);
    t1_gpu = t1_gpu->argsort(false, true); t1_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t1_cpu, t1_gpu, 10e-4));

    // Test #2
    Tensor* t2_cpu = Tensor::randn({10000});
    Tensor* t2_gpu = t2_cpu->clone(); t2_gpu->toGPU();
    t2_cpu = t2_cpu->argsort(true, true);
    t2_gpu = t2_gpu->argsort(true, true); t2_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t2_cpu, t2_gpu, 10e-4));

    // Note: I don't test the unstable sort here, because similar (or equal) float values could be
    // sorted into different positions
#endif
}


