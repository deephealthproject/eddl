#include <gtest/gtest.h>
#include <random>
#include <string>
#include <ctime>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;



TEST(TensorTestSuite, tensor_math_reduction_max) {
    // Test #1
    Tensor *t1_ref = new Tensor({5.0f, 6.0f, 9.0f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, -3.0f, 9.0f,
                                    1.0f, 6.0f, 0.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t = t1->max({0}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({4.0f, 8.0f, 9.0f, 6.0f},  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, -3.0f, 9.0f,
                                    1.0f, 6.0f, 0.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->max({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randn({3, 1000, 1000});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_max = t_cpu->max({1}, false);
    Tensor *t_gpu_max = t_gpu->max({1}, false); t_gpu_max->toCPU();
    t_cpu_max->print();
    t_gpu_max->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_max, t_gpu_max, 10e-4));
#endif
}

TEST(TensorTestSuite, tensor_math_reduction_argmax) {
    // Test #1
    Tensor *t1_ref = new Tensor({1.0f, 3.0f, 2.0f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, -3.0f, 9.0f,
                                    1.0f, 6.0f, 0.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t = t1->argmax({0}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({1.0f, 2.0f, 2.0f, 1.0f},  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, -3.0f, 9.0f,
                                    1.0f, 6.0f, 0.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->argmax({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randn({3, 1000, 1000});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_argmax = t_cpu->argmax({1}, false);
    Tensor *t_gpu_argmax = t_gpu->argmax({1}, false); t_gpu_argmax->toCPU();
    t_cpu_argmax->print();
    t_gpu_argmax->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_argmax, t_gpu_argmax, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_reduction_min) {
    // Test #1
    Tensor *t1_ref = new Tensor({1.0f, -3.0f, 0.0f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, -3.0f, 9.0f,
                                    1.0f, 6.0f, 0.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t = t1->min({0}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({1.0f, 4.0f, -3.0f, 0.0f},  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, -3.0f, 9.0f,
                                    1.0f, 6.0f, 0.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->min({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randn({3, 1000, 1000});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_min = t_cpu->min({1}, false);
    Tensor *t_gpu_min = t_gpu->min({1}, false); t_gpu_min->toCPU();
    t_cpu_min->print();
    t_gpu_min->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_min, t_gpu_min, 10e-4));
#endif
}

TEST(TensorTestSuite, tensor_math_reduction_argmin) {
    // Test #1
    Tensor *t1_ref = new Tensor({0.0f, 2.0f, 3.0f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, -3.0f, 9.0f,
                                    1.0f, 6.0f, 0.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t = t1->argmin({0}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({0.0f, 1.0f, 1.0f, 2.0f},  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, -3.0f, 9.0f,
                                    1.0f, 6.0f, 0.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->argmin({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::randn({3, 1000, 1000});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_argmin = t_cpu->argmin({1}, false);
    Tensor *t_gpu_argmin = t_gpu->argmin({1}, false); t_gpu_argmin->toCPU();
//    t_cpu_argmin->print();
//    t_gpu_argmin->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_argmin, t_gpu_argmin, 10e-4));
#endif
}

TEST(TensorTestSuite, tensor_math_reduction_sum) {
    // Test #1
    Tensor *t1_ref = new Tensor({
                                        4.0f, 4.0f,
                                        4.0f, 4.0f,
                                        4.0f, 4.0f}, {3, 2}, DEV_CPU);
    Tensor *t1 = Tensor::ones({3, 2, 4}, DEV_CPU);

    Tensor *new_t = t1->sum({2}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({
                                        2.0f, 2.0f, 2.0f, 2.0f,
                                        2.0f, 2.0f, 2.0f, 2.0f,
                                        2.0f, 2.0f, 2.0f, 2.0f}, {3, 4}, DEV_CPU);
    Tensor *t2 = Tensor::ones({3, 2, 4}, DEV_CPU);

    Tensor *new_t2 = t2->sum({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::ones({3, 1000, 1000});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_sum = t_cpu->sum({1}, false);
    Tensor *t_gpu_sum = t_gpu->sum({1}, false); t_gpu_sum->toCPU();
//    t_cpu_sum->print();
//    t_gpu_sum->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_sum, t_gpu_sum, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_reduction_sum_abs) {
    // Test #1
    Tensor *t1_ref = new Tensor({10.0f, 11.0f, 7.0f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({-4.0f, 7.0f, 3.0f,
                             6.0f, 4.0f, -4.0f}, {2, 3}, DEV_CPU);

    Tensor *new_t = t1->sum_abs({0}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #1
    Tensor *t2_ref = new Tensor({14.0f, 14.0f},  {2}, DEV_CPU);
    Tensor *t2 = new Tensor({-4.0f, 7.0f, 3.0f,
                             6.0f, 4.0f, -4.0f}, {2, 3}, DEV_CPU);

    Tensor *new_t2 = t2->sum_abs({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu = Tensor::ones({3, 1000, 1000});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_sum_abs = t_cpu->sum_abs({1}, false);
    Tensor *t_gpu_sum_abs = t_gpu->sum_abs({1}, false); t_gpu_sum_abs->toCPU();
//    t_cpu_sum_abs->print();
//    t_gpu_sum_abs->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_sum_abs, t_gpu_sum_abs, 10e-4));
#endif
}

TEST(TensorTestSuite, tensor_math_reduction_prod) {
    // Test #1
    Tensor *t1_ref = new Tensor({24.0f, 28.0f, 12.0f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({4.0f, 7.0f, 3.0f,
                             6.0f, 4.0f, 4.0f}, {2, 3}, DEV_CPU);

    Tensor *new_t = t1->prod({0}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({ 16., 160.,  48.,  32. },  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, 3.0f, 8.0f,
                                    1.0f, 4.0f, 8.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->prod({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

#ifdef cGPU
    Tensor* t_cpu = Tensor::full({10, 10}, 2.0f);  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_prod = t_cpu->prod({1}, false);
    Tensor *t_gpu_prod = t_gpu->prod({1}, false); t_gpu_prod->toCPU();
    t_cpu_prod->print();
    t_gpu_prod->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_prod, t_gpu_prod, 10e-4));
#endif
}

TEST(TensorTestSuite, tensor_math_reduction_mean) {
    // Test #1
    Tensor *t1_ref = new Tensor({5.0f, 5.5f, 3.5f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({4.0f, 7.0f, 3.0f,
                             6.0f, 4.0f, 4.0f}, {2, 3}, DEV_CPU);

    Tensor *new_t = t1->mean({0}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({ 3.0000, 5.6667, 4.3333, 4.3333},  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, 3.0f, 8.0f,
                                    1.0f, 4.0f, 8.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->mean({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

#ifdef cGPU
    Tensor* t_cpu = Tensor::randn({3, 1000, 1000});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_mean = t_cpu->mean({1}, false);
    Tensor *t_gpu_mean = t_gpu->mean({1}, false); t_gpu_mean->toCPU();
    t_cpu_mean->print();
    t_gpu_mean->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_mean, t_gpu_mean, 10e-4));
#endif
}

TEST(TensorTestSuite, tensor_math_reduction_var) {
    // Test #1
    Tensor *t1_ref = new Tensor({2.0f, 4.5f, 0.5f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({4.0f, 7.0f, 3.0f,
                             6.0f, 4.0f, 4.0f}, {2, 3}, DEV_CPU);

    Tensor *new_t = t1->var({0}, false, true);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({ 3.0000,  4.3333, 10.3333, 12.3333},  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, 3.0f, 8.0f,
                                    1.0f, 4.0f, 8.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->var({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

#ifdef cGPU
    Tensor* t_cpu = Tensor::randn({3, 1000, 1000});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_var = t_cpu->var({1}, false);
    Tensor *t_gpu_var = t_gpu->var({1}, false); t_gpu_var->toCPU();
    t_cpu_var->print();
    t_gpu_var->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_var, t_gpu_var, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_reduction_std) {
    // Test #1
    Tensor *t1_ref = new Tensor({1.4142f, 2.1213f, 0.7071f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({4.0f, 7.0f, 3.0f,
                                 6.0f, 4.0f, 4.0f}, {2, 3}, DEV_CPU);

    Tensor *new_t = t1->std({0}, false, true);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({1.7321, 2.0817, 3.2146, 3.5119},  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, 3.0f, 8.0f,
                                    1.0f, 4.0f, 8.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->std({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

#ifdef cGPU
    Tensor* t_cpu = Tensor::randn({3, 1000, 1000});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_std = t_cpu->std({1}, false);
    Tensor *t_gpu_std = t_gpu->std({1}, false); t_gpu_std->toCPU();
    t_cpu_std->print();
    t_gpu_std->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_std, t_gpu_std, 10e-4));
#endif
}



TEST(TensorTestSuite, tensor_math_reduction_mode) {
    // Test #1
    Tensor *t1_ref = new Tensor({1.0f, 4.0f, 8.0f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({
                             1.0f, 4.0f, 4.0f,
                             5.0f, 4.0f, 8.0f,
                             2.0f, 3.0f, 8.0f,
                             1.0f, 4.0f, 8.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t = t1->mode({0}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({4., 5., 2., 1.},  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 5.0f, 8.0f,
                                    2.0f, 3.0f, 2.0f,
                                    1.0f, 1.0f, 1.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->mode({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

#ifdef cGPU
    // TODO:: THERE ARE PROBLEMS WITH BIGGER TENSORS
    Tensor* t_cpu = Tensor::randn({3, 100, 100});  t_cpu->round_(); // High mismatch CPU/GPU; make either 0 or 1
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_mode = t_cpu->mode({1}, false);
    Tensor *t_gpu_mode = t_gpu->mode({1}, false); t_gpu_mode->toCPU();
//    t_cpu_mode->print();
//    t_gpu_mode->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_mode, t_gpu_mode, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_reduction_norm) {
    // Test #1
    Tensor *t1_ref = new Tensor({5.5678f,  7.5498f, 14.4222f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, 3.0f, 8.0f,
                                    1.0f, 4.0f, 8.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t = t1->norm({0}, false, "fro");
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({5.7446f, 10.2470f, 8.7750f, 9.0000f},  {4}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 4.0f, 4.0f,
                                    5.0f, 4.0f, 8.0f,
                                    2.0f, 3.0f, 8.0f,
                                    1.0f, 4.0f, 8.0f}, {4, 3}, DEV_CPU);

    Tensor *new_t2 = t2->norm({1}, false, "fro");
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

#ifdef cGPU
    // TODO:: THERE ARE PROBLEMS WITH BIGGER TENSORS
    Tensor* t_cpu = Tensor::randn({3, 100, 100});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_norm = t_cpu->norm({1}, false, "fro");
    Tensor *t_gpu_norm = t_gpu->norm({1}, false, "fro"); t_gpu_norm->toCPU();
//    t_cpu_norm->print();
//    t_gpu_norm->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_norm, t_gpu_norm, 10e-4));
#endif
}


TEST(TensorTestSuite, tensor_math_reduction_median) {
    // Test #1
    Tensor *t1_ref = new Tensor({5.0f, 4.0f, 3.0f},  {3}, DEV_CPU);
    Tensor *t1 = new Tensor({4.0f, 7.0f, 9.0f,
                                  6.0f, 4.0f, 1.0f,
                                  5.0f, 2.0f, 3.0f,}, {3, 3}, DEV_CPU);

    Tensor *new_t = t1->median({0}, false);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 10e-4));

    // Test #2
    Tensor *t2_ref = new Tensor({ 1.5f, 4.0f, 4.5f},  {3}, DEV_CPU);
    Tensor *t2 = new Tensor({
                                    1.0f, 5.0f, 2.0f, 1.0f,
                                    4.0f, 4.0f, 3.0f, 4.0f,
                                    4.0f, 5.0f, 8.0f, 1.0f}, {3, 4}, DEV_CPU);

    Tensor *new_t2 = t2->median({1}, false);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 10e-4));

#ifdef cGPU
    // TODO:: THERE ARE PROBLEMS WITH BIGGER TENSORS
    Tensor* t_cpu = Tensor::randn({3, 100, 100});  // High mismatch CPU/GPU
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    Tensor *t_cpu_median = t_cpu->median({1}, false);
    Tensor *t_gpu_median = t_gpu->median({1}, false); t_gpu_median->toCPU();
    t_cpu_median->print();
    t_gpu_median->print();

    ASSERT_TRUE(Tensor::equivalent(t_cpu_median, t_gpu_median, 10e-4));
#endif
}