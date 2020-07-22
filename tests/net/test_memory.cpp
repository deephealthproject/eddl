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

TEST(NetTestSuite, net_delete_mnist_da){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    // Data augmentation assumes 3D tensors... images:
    l=Reshape(l,{1,28,28});

    // Data augmentation
    l = RandomCropScale(l, {0.9f, 1.0f});

    // Come back to 1D tensor for fully connected:
    l=Reshape(l,{-1});
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    //l = ReLu(Dense(l, 1024));
    //l = ReLu(Dense(l, 1024));

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

// mnist_mlp_train_batch => No needed. Redundant
// mnist_mlp_auto_encoder => No needed. Redundant

TEST(NetTestSuite, net_delete_mnist_conv){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Reshape(l,{1,28,28});
    l = MaxPool(ReLu(Conv(l,32, {3,3},{1,1})),{3,3}, {1,1}, "same");
    l = MaxPool(ReLu(Conv(l,64, {3,3},{1,1})),{2,2}, {2,2}, "same");
    l = MaxPool(ReLu(Conv(l,128,{3,3},{1,1})),{3,3}, {2,2}, "none");
    l = MaxPool(ReLu(Conv(l,256,{3,3},{1,1})),{2,2}, {2,2}, "none");
    l = Reshape(l,{-1});

    layer out = Activation(Dense(l, num_classes), "softmax");
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


TEST(NetTestSuite, net_delete_mnist_rnn){
    int num_classes = 10;

    // Define network
    layer in = Input({28});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 32));
    //l = L2(RNN(l, 128, "relu"),0.001);
    l = L2(LSTM(l, 128),0.001);
    l = LeakyReLu(Dense(l, 32));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          rmsprop(0.001), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;

    ASSERT_TRUE(true);
}

TEST(NetTestSuite, net_delete_mnist_conv1D){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Reshape(l,{1,784}); //image as a 1D signal with depth=1
    l = MaxPool1D(ReLu(Conv1D(l,16, {3},{1})),{4},{4});  //MaxPool 4 stride 4
    l = MaxPool1D(ReLu(Conv1D(l,32, {3},{1})),{4},{4});
    l = MaxPool1D(ReLu(Conv1D(l,64,{3},{1})),{4},{4});
    l = MaxPool1D(ReLu(Conv1D(l,64,{3},{1})),{4},{4});
    l = Reshape(l,{-1});

    layer out = Activation(Dense(l, num_classes), "softmax");
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
