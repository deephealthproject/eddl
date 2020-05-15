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
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_acos){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->acos();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_add){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->add(1.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_addT){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->add(t1);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_asin){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->asin();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_atan){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->atan();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_ceil){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->ceil();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_clamp){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->clamp(1,1);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_clampmax){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->clampmax(2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_clampmin){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->clampmin(2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_cos){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->cos();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_cosh){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->cosh();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_div){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->div(3);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_divT){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->div(t1);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_exp){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->exp();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_floor){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->floor();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_inv){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->inv();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_log){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->log();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_log2){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->log2();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_log10){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->log10();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_logn){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->logn(2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_mod){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->mod(3);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_mult){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->mult(2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_multT){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->mult(t1);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_neg){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->neg();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_normalize){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->normalize(1,2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_pow){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->pow(2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_powb){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->powb(2);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_reciprocal){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->reciprocal();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_remainder){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->remainder(3);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_round){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->round();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_rsqrt){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->rsqrt();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_sigmoid){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sigmoid();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_sign){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sign();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_sin){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sin();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_sinh){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sinh();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_sqr){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sqr();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_sqrt){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sqrt();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_sub){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sub(3);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_subT){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->sub(t1);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_tan){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->tan();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_tanh){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->tanh();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}


TEST(TensorTestSuite, tensor_math_unary_trunc){
    // Test #1
    vector<int> t1_shape_ref = {3};
    vector<float> d_t1_ref = {1, 2, 3};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    vector<int> t1_shape = {3};
    vector<float> d_t1 = {-1, -2, 3};
    Tensor* t1 = new Tensor(t1_shape, d_t1.data(), DEV_CPU);

    Tensor* new_t = t1->trunc();
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-0));
}
