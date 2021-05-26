#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"

using namespace std;


TEST(TensorTestSuite, tensor_create_data){
    // Test #1
    vector<int> t1_shape = {2, 3};
    vector<float> d_t1_ref = {1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);

    Tensor* t1 = new Tensor( {1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f}, {2, 3}, DEV_CPU);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;

}



TEST(TensorTestSuite, tensor_create_zeros){
    // Test #1
    vector<int> t1_shape = {2, 3};
    vector<float> d_t1_ref = {0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::zeros(t1_shape);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    // Test #2
    vector<int> t2_shape = {5};
    vector<float> d_t2_ref = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Tensor* t2_ref = new Tensor(t2_shape, d_t2_ref.data(), DEV_CPU);
    Tensor* t2 = Tensor::zeros(t2_shape);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
    delete t2_ref;
    delete t2;
}

TEST(TensorTestSuite, tensor_create_zeros_like){
    Tensor* t1_ref = Tensor::zeros({2, 3}, DEV_CPU);
    Tensor* t1 = Tensor::zeros_like(t1_ref);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
}


TEST(TensorTestSuite, tensor_create_ones){
    // Test #1
    vector<int> t1_shape = {2, 3};
    vector<float> d_t1_ref = {1.0f, 1.0f, 1.0f,
                              1.0f, 1.0f, 1.0f};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::ones(t1_shape);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    // Test #2
    vector<int> t2_shape = {5};
    vector<float> d_t2_ref = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Tensor* t2_ref = new Tensor(t2_shape, d_t2_ref.data(), DEV_CPU);
    Tensor* t2 = Tensor::ones(t2_shape);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
    delete t2_ref;
    delete t2;
}

TEST(TensorTestSuite, tensor_create_ones_like){
    Tensor* t1_ref = Tensor::ones({2, 3}, DEV_CPU);
    Tensor* t1 = Tensor::ones_like(t1_ref);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
}

TEST(TensorTestSuite, tensor_create_full){
    // Test #1
    vector<int> t1_shape = {2, 3};
    vector<float> d_t1_ref = {3.141592f, 3.141592f, 3.141592f,
                              3.141592f, 3.141592f, 3.141592f};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::full(t1_shape, 3.141592f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;

}

TEST(TensorTestSuite, tensor_create_full_like){
    Tensor* t1_ref = Tensor::full({2, 3}, 15.0f, DEV_CPU);
    Tensor* t1 = Tensor::full_like(t1_ref, 15.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
}    

TEST(TensorTestSuite, tensor_create_arange){
    // Test #1
    vector<int> t1_shape = {5};
    vector<float> d_t1_ref = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::arange(0.0f, 5.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    // Test #2
    vector<int> t2_shape = {3};
    vector<float> d_t2_ref = {1.0f, 1.5f, 2.0f};
    Tensor* t2_ref = new Tensor(t2_shape, d_t2_ref.data(), DEV_CPU);
    Tensor* t2 = Tensor::arange(1.0f, 2.5f, 0.5f);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
    delete t2_ref;
    delete t2;
}


TEST(TensorTestSuite, tensor_create_range){
    // Test #1
    vector<int> t1_shape = {4};
    vector<float> d_t1_ref = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::range(1.0f, 4.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    // Test #2
    vector<int> t2_shape = {7};
    vector<float> d_t2_ref = {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};
    Tensor* t2_ref = new Tensor(t2_shape, d_t2_ref.data(), DEV_CPU);
    Tensor* t2 = Tensor::range(1.0f, 4.0f, 0.5f);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
    delete t2_ref;
    delete t2;
}


TEST(TensorTestSuite, tensor_create_linspace){
    // Test #1
    vector<int> t1_shape = {5};
    vector<float> d_t1_ref = {3.0000,   4.7500,   6.5000,   8.2500,  10.0000};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::linspace(3, 10, 5);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    // Test #2
    vector<int> t2_shape = {5};
    vector<float> d_t2_ref = {-10.,  -5.,   0.,   5.,  10.};
    Tensor* t2_ref = new Tensor(t2_shape, d_t2_ref.data(), DEV_CPU);
    Tensor* t2 = Tensor::linspace(-10.0f, 10.0f, 5);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, t2, 1e-3f, 0.0f, true, true));

    // Test #3
    vector<int> t3_shape = {1};
    vector<float> d_t3_ref = {-10};
    Tensor* t3_ref = new Tensor(t3_shape, d_t3_ref.data(), DEV_CPU);
    Tensor* t3 = Tensor::linspace(-10.0f, 10.0f, 1);
    ASSERT_TRUE(Tensor::equivalent(t3_ref, t3, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
    delete t2_ref;
    delete t2;
    delete t3_ref;
    delete t3;
}


TEST(TensorTestSuite, tensor_create_logspace){
    // Test #1
    vector<int> t1_shape = {5};
    vector<float> d_t1_ref = {1.0000e-10,  1.0000e-05,  1.0000e+00,  1.0000e+05,  1.0000e+10};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::logspace(-10.0f, 10.0f, 5);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    // Test #2
    vector<int> t2_shape = {5};
    vector<float> d_t2_ref = {1.2589,   2.1135,   3.5481,   5.9566,  10.0000};
    Tensor* t2_ref = new Tensor(t2_shape, d_t2_ref.data(), DEV_CPU);
    Tensor* t2 = Tensor::logspace(0.1f, 1.0, 5);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, t2, 1e-3f, 0.0f, true, true));

    // Test #3
    vector<int> t3_shape = {1};
    vector<float> d_t3_ref = {1.2589};
    Tensor* t3_ref = new Tensor(t3_shape, d_t3_ref.data(), DEV_CPU);
    Tensor* t3 = Tensor::logspace(0.1, 1.0, 1);
    ASSERT_TRUE(Tensor::equivalent(t3_ref, t3, 1e-3f, 0.0f, true, true));


    // Test #4
    vector<int> t4_shape = {1};
    vector<float> d_t4_ref = {4.0};
    Tensor* t4_ref = new Tensor(t4_shape, d_t4_ref.data(), DEV_CPU);
    Tensor* t4 = Tensor::logspace(2, 2, 1, 2);
    ASSERT_TRUE(Tensor::equivalent(t4_ref, t4, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
    delete t2_ref;
    delete t2;
    delete t3_ref;
    delete t3;
    delete t4_ref;
    delete t4;
}


TEST(TensorTestSuite, tensor_create_geomspace){
    // Test #1
    vector<int> t1_shape = {4};
    vector<float> d_t1_ref = {1.,    10.,   100.,  1000.};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::geomspace(1.0f, 1000.0f, 4);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    // Test #2
    vector<float> d_t2_ref = { 1.,    2.,    4.,    8.,   16.,   32.,   64.,  128.,  256.};
    vector<int> t2_shape = {9};
    Tensor* t2_ref = new Tensor(t2_shape, d_t2_ref.data(), DEV_CPU);
    Tensor* t2 = Tensor::geomspace(1.0f, 256.0f, 9);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
    delete t2_ref;
    delete t2;
}

TEST(TensorTestSuite, tensor_create_eye){
    // Test #1
    vector<int> t1_shape = {3, 3};
    vector<float> d_t1_ref = {1.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 1.0f};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::eye(3);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, t1, 1e-3f, 0.0f, true, true));

    // Test #2
    vector<int> t2_shape = {3, 3};
    vector<float> d_t2_ref = {0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 1.0f,
                              0.0f, 0.0f, 0.0f};
    Tensor* t2_ref = new Tensor(t2_shape, d_t2_ref.data(), DEV_CPU);
    Tensor* t2 = Tensor::eye(3, 1);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
    delete t2_ref;
    delete t2;
}


TEST(TensorTestSuite, tensor_create_diag){
    // Test #1
    vector<int> t1_shape = {3, 3};
    vector<float> d_t1_ref = {1.0f, 0.0f, 0.0f,
                              0.0f, 5.0f, 0.0f,
                              0.0f, 0.0f, 9.0f};
    Tensor* t1_ref = new Tensor(t1_shape, d_t1_ref.data(), DEV_CPU);
    Tensor* t1 = Tensor::range(1, 9); t1->reshape_({3, 3});

    Tensor* new_t1 = t1->diag(0);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t1, 1e-3f, 0.0f, true, true));

    // Test #2
    vector<int> t2_shape = {3, 3};
    vector<float> d_t2_ref = {0.0f, 2.0f, 0.0f,
                              0.0f, 0.0f, 6.0f,
                              0.0f, 0.0f, 0.0f};
    Tensor* t2_ref = new Tensor(t2_shape, d_t2_ref.data(), DEV_CPU);
    Tensor* t2 = Tensor::range(1, 9); t2->reshape_({3, 3});

    Tensor* new_t2 = t2->diag(1);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1;
    delete new_t1;
    delete t2_ref;
    delete t2;
    delete new_t2;

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor* new_t_cpu = t_cpu->diag(0);
    Tensor* new_t_gpu = t_gpu->diag(0); new_t_gpu->toCPU();

    ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 1e-3f, 0.0f, true, true));

    delete t_cpu;
    delete t_gpu;
    delete new_t_cpu;
    delete new_t_gpu;

#endif
}