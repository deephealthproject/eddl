/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include <mpi.h>

using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic Conv1D for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    int id;

 init_distributed(&argc, &argv,8, &id);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download mnist
    download_mnist();

    // Settings
    int epochs = testing ? 2 : 5;
    int batch_size = 100;
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

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU({1}, "low_mem"); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
        // cs = CS_FPGA({1});
    }

    // Build model
    build(net,
          rmsprop(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          cs);

    // View model
    summary(net);

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * batch_size);
        Tensor* x_mini_train = x_train->select({_range_, ":"});
        Tensor* y_mini_train = y_train->select({_range_, ":"});
        Tensor* x_mini_test  = x_test->select({_range_, ":"});
        Tensor* y_mini_test  = y_test->select({_range_, ":"});

        delete x_train;
        delete y_train;
        delete x_test;
        delete y_test;

        x_train = x_mini_train;
        y_train = y_mini_train;
        x_test  = x_mini_test;
        y_test  = y_mini_test;
    }

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Evaluate
    evaluate(net, {x_test}, {y_test});

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;
    
      end_distributed();
    
    return EXIT_SUCCESS;
}
