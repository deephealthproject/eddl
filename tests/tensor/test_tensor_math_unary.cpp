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
}


TEST(TensorTestSuite, tensor_math_unary_mult){
    // Test #1
    vector<int> t1_shape_ref = {4};
    vector<float> d_t1_ref = { 18.3800, -264.3900,  108.2000,   99.3300};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {4};
    vector<float> d_t1 = {0.1838, -2.6439,  1.0820,  0.9933};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->mult(100);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
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
}


//TEST(TensorTestSuite, tensor_math_unary_round){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->round();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_rsqrt){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->rsqrt();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_sigmoid){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->sigmoid();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_sign){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->sign();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_sin){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->sin();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_sinh){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->sinh();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_sqr){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->sqr();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_sqrt){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->sqrt();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_sub){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->sub(3);
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_subT){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->sub(t1);
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_tan){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->tan();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_tanh){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->tanh();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
//
//
//TEST(TensorTestSuite, tensor_math_unary_trunc){
//    // Test #1
//    vector<int> t1_shape_ref = {4};
//    vector<float> d_t1_ref = {1, 2, 3};
//    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);
//
//    vector<int> t1_shape = {4};
//    vector<float> d_t1 = {-1, -2, 3};
//    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);
//
//    Tensor* new_t = t1->trunc();
//    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));
//}
