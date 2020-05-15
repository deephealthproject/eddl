/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// mnist_mlp.cpp:
// A very basic Conv1D for mnist
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

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1}, "full_mem") // one GPU
          CS_CPU(-1) // CPU with maximum threads availables
    );

    // View model
    summary(net);

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Evaluate
    evaluate(net, {x_test}, {y_test});

}
