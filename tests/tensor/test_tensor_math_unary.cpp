#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;


TEST(TensorTestSuite, tensor_math_unary_abs){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->abs();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));


    // Test GPU
    #ifdef cGPU
        Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
        Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
        t_cpu->abs_();
        t_gpu->abs_(); t_gpu->toCPU();
        ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
    #endif
}


TEST(TensorTestSuite, tensor_math_unary_acos){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {1.2294,  2.2004,  1.3690,  1.7298};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.3348, -0.5889,  0.2005, -0.1584};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->acos();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->acos_();
    t_gpu->acos_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_add){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {20.0202,  21.0985,  21.3506,  19.3944};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.0202,  1.0985,  1.3506, -0.6056};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->add(20.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->add_(100.0f);
    t_gpu->add_(100.0f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_addT){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {2, 4, -6};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {1, 2, -3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->add(t1);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->add_(t_cpu);
    t_gpu->add_(t_gpu); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_asin){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-0.6387, std::numeric_limits<double>::quiet_NaN(), -0.4552,  std::numeric_limits<double>::quiet_NaN()};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {-0.5962,  1.4985, -0.4396,  1.4525};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->asin();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->asin_();
    t_gpu->asin_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_atan){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0.2299,  0.2487, -0.5591, -0.5727};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.2341,  0.2539, -0.6256, -0.6448};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->atan();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->atan_();
    t_gpu->atan_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_ceil){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-0., -1., -1.,  1.};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {-0.6341, -1.4208, -1.0900,  0.5826};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->ceil();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->ceil_();
    t_gpu->ceil_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_clamp){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-0.5000,  0.1734, -0.0478, -0.0922};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {-1.7120,  0.1734, -0.0478, -0.0922};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->clamp(-0.5f,0.5f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->clamp_(0.1, 0.3);
    t_gpu->clamp_(0.1, 0.3); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_clampmax){
    // Test #1
    vector<int> t1_shape_ref = {9};
    vector<float> d_t1_ref = {-4, -3, -2, -1, 0, 1, 2, 2, 2};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {9};
    vector<float> d_t1 = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->clampmax(2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->clampmax_(0.3f);
    t_gpu->clampmax_(0.3f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_clampmin){
    // Test #1
    vector<int> t1_shape_ref = {9};
    vector<float> d_t1_ref = {-2, -2, -2, -1, 0, 1, 2, 3, 4};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {9};
    vector<float> d_t1 = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->clampmin(-2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->clampmin_(0.1f);
    t_gpu->clampmin_(0.1f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_cos){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0.1395,  0.2957,  0.6553,  0.5574};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {1.4309,  1.2706, -0.8562,  0.9796};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->cos();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->cos_();
    t_gpu->cos_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_cosh){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {1.0133,  1.7860,  1.2536,  1.2805};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.1632,  1.1835, -0.6979, -0.7325};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->cosh();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->cosh_();
    t_gpu->cosh_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_div){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0.7620,  2.5548, -0.5944, -0.7439,  0.9275};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.3810,  1.2774, -0.2972, -0.3719,  0.4637};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->div(0.5);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->div_(2.0f);
    t_gpu->div_(2.0f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_divT){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {1, 1, 1, 1};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {1, 2, 3, 4};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->div(t1);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->div_(t_cpu);
    t_gpu->div_(t_gpu); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_exp){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = { 1., 2., 3., 4.};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {::logf(1), ::logf(2), ::logf(3), ::logf(4)};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->exp();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->exp_();
    t_gpu->exp_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_floor){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-1.,  1., -1., -1.};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {-0.8166,  1.5308, -0.2530, -0.2091};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->floor();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->floor_();
    t_gpu->floor_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_inv){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {1, 2, 3, 4};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {1.0000, 0.5000, 0.3333, 0.2500};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->inv();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->inv_(5.0f);
    t_gpu->inv_(5.0f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_log){
    // Test #1
    vector<int> t1_shape_ref = {5};
    vector<float> d_t1_ref = { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), -0.1128,  0.3666, -2.1286};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {5};
    vector<float> d_t1 = {-0.7168, -0.5471,  0.8933,  1.4428,  0.1190};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->log();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->log_();
    t_gpu->log_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_log2){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-0.2483, -0.3213, -0.0042, -0.9196, -4.3504};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.8419,  0.8003,  0.9971,  0.5287,  0.0490};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->log2();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->log2_();
    t_gpu->log2_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_log10){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-0.2820, -0.0290, -0.1392, -0.8857, -0.6476};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.5224,  0.9354,  0.7257,  0.1301,  0.2251};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->log10();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->log10_();
    t_gpu->log10_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_logn){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {1, 2, 3, 4};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {3, 9, 27, 81};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->logn(3);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->logn_(3.0f);
    t_gpu->logn_(3.0f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_mod){
    // Test #1
    vector<int> t1_shape_ref = {6};
    vector<float> d_t1_ref = {-0.0000, -0.5000, -1.0000,  1.0000,  0.5000,  0.0000};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {6};
    vector<float> d_t1 = {-3., -2, -1, 1, 2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->mod(1.5f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    vector<int> t2_shape_ref = {6};
    vector<float> d_t2_ref = {-0.0000, -0.5000, -1.0000,  1.0000,  0.5000,  0.0000};
    Tensor* t2_ref = new Tensor(t2_shape_ref, d_t2_ref.data(), DEV_CPU);

    vector<int> t2_shape = {6};
    vector<float> d_t2 = {-3., -2, -1, 1, 2, 3};
    Tensor* t2 = new Tensor(t2_shape, d_t2.data(), DEV_CPU);

    Tensor* new_t2 = t2->mod(-1.5f);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->mod_(1.5f);
    t_gpu->mod_(1.5f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_mult){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = { 18.3800, -264.3900,  108.2000,   99.3300};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.1838, -2.6439,  1.0820,  0.9933};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->mult(100.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->mult_(100.0f);
    t_gpu->mult_(100.0f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_multT){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0.0338, 6.9903, 1.1708, 0.9866};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.1838, -2.6439,  1.0820,  0.9933};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->mult(t1);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->mult_(t_cpu);
    t_gpu->mult_(t_gpu); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_neg){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-1, -2, 3, 4};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {1, 2, -3, -4};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->neg();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->neg();
    t_gpu->neg(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_normalize){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {0.0f, 127.5, 255.0f};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1.0f, 0.0, 1.0f};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->normalize(0.0f,255.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->normalize_();
    t_gpu->normalize_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_pow){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0.1875,  1.5561,  0.4670,  0.0779};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.4331,  1.2475,  0.6834, -0.2791};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->pow(2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->pow_(3.5f);
    t_gpu->pow_(3.5f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_powb){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {5, 25, 125, 625};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {1.,  2.,  3.,  4.};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->powb(5);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->powb_(4);
    t_gpu->powb_(4); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_reciprocal){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-2.1763, -0.4713, -0.6986,  1.3702};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {-0.4595, -2.1219, -1.4314,  0.7298};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->reciprocal();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->reciprocal_();
    t_gpu->reciprocal_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_remainder){
    // Test #1
    vector<int> t1_shape_ref = {6};
    vector<float> d_t1_ref = {0.0000, 1.0000, 0.5000, 1.0000, 0.5000, 0.0000};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {6};
    vector<float> d_t1 = {-3., -2, -1, 1, 2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->remainder(1.5f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    vector<int> t2_shape_ref = {6};
    vector<float> d_t2_ref = { 0.0000, -0.5000, -1.0000, -0.5000, -1.0000,  0.0000};
    Tensor* t2_ref = new Tensor(t2_shape_ref, d_t2_ref.data(), DEV_CPU);

    vector<int> t2_shape = {6};
    vector<float> d_t2 = {-3., -2, -1, 1, 2, 3};
    Tensor* t2 = new Tensor(t2_shape, d_t2.data(), DEV_CPU);

    Tensor* new_t2 = t2->remainder(-1.5f);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->remainder_(2.3f);
    t_gpu->remainder_(2.3f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_round){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {1.,  1.,  1., -1.};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.9920,  0.6077,  0.9734, -1.0362};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->round();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->round();
    t_gpu->round(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_rsqrt){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {std::numeric_limits<double>::quiet_NaN(),  1.8351,  0.8053, std::numeric_limits<double>::quiet_NaN()};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {-0.0370,  0.2970,  1.5420, -0.9105};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->rsqrt();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->rsqrt_();
    t_gpu->rsqrt_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_sigmoid){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0.7153,  0.7481,  0.2920,  0.1458};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.9213,  1.0887, -0.8858, -1.7683};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sigmoid();
    new_t->sigmoid();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->sigmoid_();
    t_gpu->sigmoid_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_sign){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {1., -1.,  0.,  1.};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.7000, -1.2000,  0.0000,  2.3000};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sign();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->sign_();
    t_gpu->sign_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_sin){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-0.5194,  0.1343, -0.4032, -0.2711};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {-0.5461,  0.1347, -2.7266, -0.2746};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sin();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->sin_();
    t_gpu->sin_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_sinh){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0.5644, -0.9744, -0.1268,  1.0845};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.5380, -0.8632, -0.1265,  0.9399};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sinh();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->sinh_();
    t_gpu->sinh_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_sqr){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0.3594, 0.1461, 1.4741, 6.0706};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = { 0.5995, -0.3823, -1.2141, -2.4639};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sqr();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->sqr_();
    t_gpu->sqr_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_sqrt){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {std::numeric_limits<double>::quiet_NaN(),  1.0112,  0.2883,  0.6933};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {-2.0755,  1.0226,  0.0831,  0.4806};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sqrt();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->sqrt_();
    t_gpu->sqrt_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_sub){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-9.4202,  -9.2406,  -9.3453, -10.3650};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.5798,  0.7594,  0.6547, -0.3650};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);//TODO: doo

    Tensor* new_t = t1->sub(10);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->sub_(3.0f);
    t_gpu->sub_(3.0f); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_subT){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0., 0., 0., 0.};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.5798,  0.7594,  0.6547, -0.3650};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);//TODO: doo

    Tensor* new_t = t1->sub(t1);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->sub_(t_cpu);
    t_gpu->sub_(t_gpu); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_tan){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {-2.5928, 4.9868, 0.4722, -5.3378};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {-1.2027, -1.7687,  0.4412, -1.3856};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->tan();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->tan_();
    t_gpu->tan_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_tanh){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {0.7156, -0.6218,  0.8257,  0.2553};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.8986, -0.7279,  1.1745,  0.2611};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->tanh();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->tanh_();
    t_gpu->tanh_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_trunc){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = {3.,  0., -0., -0.};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {3.4742,  0.5466, -0.8008, -0.9079};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->trunc();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
    t_cpu->trunc_();
    t_gpu->trunc_(); t_gpu->toCPU();
    ASSERT_TRUE(Tensor::equivalent(t_cpu, t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_maximum){
    // Test #1
    Tensor *t1_ref = Tensor::full({2, 3}, 10);

    Tensor *t1 = Tensor::full({2, 3}, 5);

    Tensor* new_t = Tensor::maximum(t1, 10.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor* new_t_cpu = Tensor::maximum(t_cpu, 0.75f);
    Tensor* new_t_gpu = Tensor::maximum(t_gpu, 0.75f);; new_t_gpu->toCPU();

    ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_maximumT){
    // Test #1
    Tensor *t1_ref = Tensor::full({2, 3}, 10);

    Tensor *t1_A = Tensor::full({2, 3}, 5);
    Tensor *t1_B = Tensor::full({2, 3}, 10);

    Tensor* new_t = Tensor::maximum(t1_A, t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu_A = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu_A = t_cpu_A->clone(); t_gpu_A->toGPU();

    Tensor* t_cpu_B = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu_B = t_cpu_B->clone(); t_gpu_B->toGPU();

    Tensor* new_t_cpu = Tensor::maximum(t_cpu_A, t_cpu_B);
    Tensor* new_t_gpu =Tensor::maximum(t_gpu_A, t_gpu_B);; new_t_gpu->toCPU();

    ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 10e-4));
#endif
}

TEST(TensorTestSuite, tensor_math_minimum){
    // Test #1
    Tensor *t1_ref = Tensor::full({2, 3}, 5);

    Tensor *t1 = Tensor::full({2, 3}, 10);

    Tensor* new_t = Tensor::minimum(t1, 5.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor* new_t_cpu = Tensor::minimum(t_cpu, 0.25f);
    Tensor* new_t_gpu = Tensor::minimum(t_gpu, 0.25f);; new_t_gpu->toCPU();

    ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 10e-4));
#endif
}

TEST(TensorTestSuite, tensor_math_minimumT){
    // Test #1
    Tensor *t1_ref = Tensor::full({2, 3}, 5);

    Tensor *t1_A = Tensor::full({2, 3}, 5);
    Tensor *t1_B = Tensor::full({2, 3}, 10);

    Tensor* new_t = Tensor::minimum(t1_A, t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu_A = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu_A = t_cpu_A->clone(); t_gpu_A->toGPU();

    Tensor* t_cpu_B = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu_B = t_cpu_B->clone(); t_gpu_B->toGPU();

    Tensor* new_t_cpu = Tensor::minimum(t_cpu_A, t_cpu_B);
    Tensor* new_t_gpu =Tensor::minimum(t_gpu_A, t_gpu_B);; new_t_gpu->toCPU();

    ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_unary_sum){
    // Test #1
    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.8986, -0.7279,  1.1745,  0.2611};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    float t_sum = t1->sum();
    ASSERT_NEAR(t_sum, 1.6063f, 10e-4f);

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    float t_cpu_sum = t_cpu->sum();
    float t_gpu_sum = t_gpu->sum(); t_gpu->toCPU();

    ASSERT_NEAR(t_cpu_sum, t_gpu_sum, 10e-4f);
#endif
}


TEST(TensorTestSuite, tensor_math_unary_abs_sum){
    // Test #1
    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.8986, -0.7279,  1.1745,  0.2611};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    float t_sum = t1->sum_abs();
    ASSERT_NEAR(t_sum, 3.0621f, 10e-4f);

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    float t_cpu_sum = t_cpu->sum_abs();
    float t_gpu_sum = t_gpu->sum_abs(); t_gpu->toCPU();

    ASSERT_NEAR(t_cpu_sum, t_gpu_sum, 10e-4f);
#endif
}


TEST(TensorTestSuite, tensor_math_unary_max){
    // Test #1
    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.8986, -0.7279,  1.1745,  0.2611};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    float t_max = t1->max();
    ASSERT_NEAR(t_max, 1.1745, 10e-4f);

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    float t_cpu_max = t_cpu->max();
    float t_gpu_max = t_gpu->max(); t_gpu->toCPU();

    ASSERT_NEAR(t_cpu_max, t_gpu_max, 10e-4f);
#endif
}


TEST(TensorTestSuite, tensor_math_unary_min){
    // Test #1
    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.8986, -0.7279,  1.1745,  0.2611};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    float t_min = t1->min();
    ASSERT_NEAR(t_min, -0.7279, 10e-4f);

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randu({3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    float t_cpu_min = t_cpu->min();
    float t_gpu_min = t_gpu->min(); t_gpu->toCPU();

    ASSERT_NEAR(t_cpu_min, t_gpu_min, 10e-4f);
#endif
}