#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"

using namespace std;


TEST(TensorTestSuite, tensor_create_trace){
    // Test #1
//    vector<int> t1_shape = {3, 3};
//    vector<float> d_t1_ref = {1.0f, 0.0f, 0.0f,
//                              0.0f, 5.0f, 0.0f,
//                              0.0f, 0.0f, 9.0f};
//    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);

    Tensor* t1 = Tensor::range(1, 9); t1->reshape_({3, 3});

    float t1_sum = t1->trace(0);
    ASSERT_TRUE(t1_sum == 15.0f);

    // Test #2
//    vector<int> t2_shape = {3, 3};
//    vector<float> d_t2_ref = {0.0f, 2.0f, 0.0f,
//                              0.0f, 0.0f, 6.0f,
//                              0.0f, 0.0f, 0.0f};
//    Tensor* t2_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);

    Tensor* t2 = Tensor::range(1, 9); t2->reshape_({3, 3});

    float t2_sum = t2->trace(1);
    ASSERT_TRUE(t2_sum == 8.0f);

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    float t_cpu_sum = t_cpu->trace(0);
    float t_gpu_sum = t_gpu->trace(0);

    ASSERT_TRUE(abs(t_cpu_sum - t_gpu_sum) < 10e-4);
#endif
}