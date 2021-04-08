#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;



TEST(TensorTestSuite, tensor_da_pad){
    int pad;

    // Test #1
    pad = 1;
    Tensor* t1 = Tensor::ones({1, 1, 3, 3});

    Tensor* t1_ref = new Tensor({
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {1, 1, 3+pad*2, 3+pad*2});

    Tensor* t_new = t1->pad({pad, pad});
//    t_new->squeeze_(); t_new->print(2.0f);
//    t1_ref->squeeze_(); t1_ref->print(2.0f);
    ASSERT_TRUE(Tensor::equivalent(t_new, t1_ref, 1e-3f, 0.0f, true, true));

    delete t1;
    delete t1_ref;

    // Test GPU
#ifdef cGPU
    pad = 5;
    Tensor* t1_cpu = Tensor::randn({1, 3, 100, 100});
    Tensor* t1_gpu = t1_cpu->clone(); t1_gpu->toGPU();

    Tensor* t1_cpu_out = t1_cpu->pad({pad, pad});
    Tensor* t1_gpu_out = t1_gpu->pad({pad, pad}); t1_gpu_out->toCPU();
//    t1_gpu_out->squeeze_(); t1_gpu_out->print(2.0f);
    ASSERT_TRUE(Tensor::equivalent(t1_cpu_out, t1_gpu_out, 1e-3f, 0.0f, true, true));

    delete t1_cpu;
    delete t1_gpu;
    delete t1_cpu_out;
    delete t1_gpu_out;
#endif

}
