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
    layer in=Input({3, 32, 32});
    layer l=in;

    l = Conv(l,32,{3,3},{1,1});
    l = ReLu(l);
    l = LeakyReLu(l);
    l = Flatten(l);
    l = Dense(l, 10);
    layer out = Softmax(l);  // num_classes
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
