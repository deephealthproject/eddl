/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"
#include "eddl/apis/eddlT.h"

using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic MLP for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {
    // Download mnist
    download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({28});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 32));
    l = LeakyReLu(RNN(l, 32));
    l = LeakyReLu(Dense(l, 32));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          rmsprop(0.001), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1}, "low_mem") // one GPU
          CS_CPU(-1, "low_mem") // CPU with maximum threads availables
    );
    //toGPU(net,{1},100,"low_mem"); // In two gpus, syncronize every 100 batches, low_mem setup

    // View model
    summary(net);


    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");
    tensor x_test = eddlT::load("tsX.bin");
    tensor y_test = eddlT::load("tsY.bin");

    x_train->reshape_({60000,28,28});
    tensor x_train2=Tensor::permute(x_train,{1,0,2});

    x_train2->info();

    // Preprocessing
    eddlT::div_(x_train2, 255.0);
    eddlT::div_(x_test, 255.0);

    setlogfile(net,"recurrent_mnist");

    // Train model
    fit(net,{x_train2}, {y_train}, batch_size, epochs);

    // Evaluate
    //evaluate(net, {x_test}, {y_test});


}
