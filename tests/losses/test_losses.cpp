#include <gtest/gtest.h>


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace eddl;


TEST(NetTestSuite, losses_categorical_cross_entropy){

    auto loss = LCategoricalCrossEntropy();

    Tensor* t1_y_pred = new Tensor({0.7, 0.2, 0.1,
                                          0.3, 0.5, 0.2}, {2, 3});
    Tensor* t1_y = new Tensor({1.0, 0.0, 0.0,
                                    0.0, 1.0, 0.0}, {2, 3});

    // Compute loss
    float value = loss.value(t1_y, t1_y_pred);
    ASSERT_NEAR(value, 0.524911046f, 10e-4f);


    // Compute delta
    Tensor* t1_delta = Tensor::zeros_like(t1_y);
    Tensor* t1_delta_ref = new Tensor({-1.4285f, 0.0, 0.0,
                                           0.0, -2.0000f, 0.0}, {2, 3});
    loss.delta(t1_y, t1_y_pred, t1_delta);
    ASSERT_TRUE(Tensor::equivalent(t1_delta_ref, t1_delta, 10e-4));

    // Deletes
    delete t1_y_pred;
    delete t1_y;
    delete t1_delta;
    delete t1_delta_ref;

    // Test GPU
#ifdef cGPU
    // Test: Loss value
    // Generate default predictions
    int rows = 1000;
    Tensor* t_cpu_y_pred = Tensor::randu({rows, rows}); // Default values between 0 and 1
    t_cpu_y_pred->clamp_(0.0f, 1.0f); // Clamp between 0.0 a 1 (just in case)
    Tensor* t_gpu_y_pred = t_cpu_y_pred->clone(); t_gpu_y_pred->toGPU();

    // Generate one-hots
    Tensor* t_cpu_y_true = Tensor::eye(rows);
    Tensor* t_gpu_y_true = t_cpu_y_true->clone(); t_gpu_y_true->toGPU();

    // Compute loss
    float cpu_loss = loss.value(t_cpu_y_true, t_cpu_y_pred);
    float gpu_loss = loss.value(t_gpu_y_true, t_gpu_y_pred);
    ASSERT_NEAR(cpu_loss-gpu_loss, 0.0f, 10e-4f);

    // Test: Deltas
    // Generate matrices
    Tensor* t_cpu_delta = Tensor::zeros_like(t_cpu_y_pred);
    Tensor* t_gpu_delta = t_cpu_delta->clone(); t_gpu_delta->toGPU();

    // Compute deltas
    loss.delta(t_cpu_y_true, t_cpu_y_pred, t_cpu_delta);
    loss.delta(t_gpu_y_true, t_gpu_y_pred, t_gpu_delta);

    t_gpu_delta->toCPU();  // Send to CPU
    ASSERT_TRUE(Tensor::equivalent(t_cpu_delta, t_gpu_delta, 10e-4));

    // Deletes
    delete t_cpu_y_pred;
    delete t_gpu_y_pred;
    delete t_cpu_y_true;
    delete t_gpu_y_true;
    delete t_cpu_delta;
    delete t_gpu_delta;
#endif
}

