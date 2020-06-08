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

    Tensor* condition = t1_A->greater(0.0f);

    Tensor* new_t = Tensor::where(condition, t1_A, t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu_A = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu_A = t_cpu_A->clone(); t_gpu_A->toGPU();

    Tensor* t_cpu_B = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu_B = t_cpu_B->clone(); t_gpu_B->toGPU();

    Tensor* t_cpu_condition = Tensor::randu({3, 1000, 1000}); t_cpu_condition->round();
    Tensor* t_gpu_condition = t_cpu_condition->clone(); t_gpu_condition->toGPU();

    Tensor* new_t_cpu = Tensor::where(t_cpu_condition, t_cpu_A, t_cpu_B);
    Tensor* new_t_gpu =Tensor::where(t_gpu_condition, t_gpu_A, t_gpu_B);; new_t_gpu->toCPU();

    ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 10e-4));
#endif
}
