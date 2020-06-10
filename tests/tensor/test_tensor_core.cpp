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


//    // Test GPU
//#ifdef cGPU
//    Tensor* t_cpu = Tensor::randu({1000});
//        Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
//        t_cpu->sort_();
//        t_gpu->sort_(); t_gpu->toCPU();
//        ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
//#endif
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


//    // Test GPU
//#ifdef cGPU
//    Tensor* t_cpu = Tensor::randu({1000});
//        Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
//        t_cpu->sort_();
//        t_gpu->sort_(); t_gpu->toCPU();
//        ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
//#endif
}