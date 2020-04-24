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
// mnist_rnn.cpp:
// A recurrent NN for mnist
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
          //CS_GPU({1}) // one GPU
          CS_CPU(-1,"low_mem") // CPU with maximum threads availables
    );

    // View model
    summary(net);


    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");
    tensor x_test = eddlT::load("tsX.bin");
    tensor y_test = eddlT::load("tsY.bin");

    x_train->reshape_({60000,28,28});
    tensor x_trainp=Tensor::permute(x_train,{1,0,2});
    delete x_train;

    x_test->reshape_({10000,28,28});
    tensor x_testp=Tensor::permute(x_test,{1,0,2});
    delete x_test;

    // Preprocessing
    eddlT::div_(x_trainp, 255.0);
    eddlT::div_(x_testp, 255.0);

    setlogfile(net,"recurrent_mnist");

    // Train model

    for(int i=0;i<epochs;i++) {
      fit(net,{x_trainp}, {y_train}, batch_size, 1);
      evaluate(net, {x_testp}, {y_test});
    }


}
