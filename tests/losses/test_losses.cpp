#include <gtest/gtest.h>


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace eddl;


TEST(NetTestSuite, losses_full_cross_entropy){

    auto loss = LFullCrossEntropy();

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

}

