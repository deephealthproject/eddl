#include <gtest/gtest.h>


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace eddl;


TEST(NetTestSuite, net1_memory_leaks){
    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));

    layer out = Softmax(Dense(l, 10));  // num_classes
    model net = Model({in}, {out});

    optimizer opt = rmsprop(0.01);
    compserv cs = CS_CPU();

    // Build model
    build(net,
          opt, // Optimizer
        {"soft_cross_entropy"}, // Losses
        {"categorical_accuracy"}, // Metrics
        cs
    );

//    delete opt;
//    delete cs;
    delete net;
    std::cout << "end" << std::endl;

    ASSERT_TRUE(true);
}
