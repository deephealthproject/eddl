#include <gtest/gtest.h>


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace eddl;


TEST(NetTestSuite, memory_leaks_select){
    layer in=Input({3, 32, 32});
    auto l = new LSelect(in, {":", "0:31", "0:31"}, "mylayer", DEV_CPU, 0);

    delete l;
    std::cout << "layer deleted" << std::endl;
    delete in;

    ASSERT_TRUE(true);
}

TEST(NetTestSuite, net_delete_mnist_mlp){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;

    ASSERT_TRUE(true);
}


TEST(NetTestSuite, net_delete_mnist_initializers){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = ReLu(GlorotNormal(Dense(l, 1024)));
    l = ReLu(GlorotUniform(Dense(l, 1024)));
    l = ReLu(RandomNormal(Dense(l, 1024),0.0,0.1));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;

    ASSERT_TRUE(true);
}



TEST(NetTestSuite, net_delete_mnist_regularizers){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = ReLu(L2(Dense(l, 1024),0.0001));
    l = ReLu(L1(Dense(l, 1024),0.0001));
    l = ReLu(L1L2(Dense(l, 1024),0.00001,0.0001));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;
    ASSERT_TRUE(true);
}
