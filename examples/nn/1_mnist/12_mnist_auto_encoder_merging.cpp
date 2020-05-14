/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
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
// mnist_auto_encoder.cpp:
// An autoencoder for mnist
// merging two networs
//////////////////////////////////

int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 100;


    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Activation(Dense(l, 256), "relu");
    l = Activation(Dense(l, 128), "relu");
    layer out = Activation(Dense(l, 64), "relu");

    model net = Model({in}, {out});

    summary(net);

    // Define network
    in = Input({64});
    l = Activation(Dense(in, 128), "relu");
    l = Activation(Dense(l, 256), "relu");

    out = Dense(l, 784);

    model net2 = Model({in}, {out});

    summary(net2);

    model netf = Model({net,net2});


    // View model
    summary(netf);
    plot(netf, "model.pdf");

    // Build model
    build(netf,
          sgd(0.001, 0.9), // Optimizer
          {"mean_squared_error"}, // Losses
          {"mean_squared_error"}, // Metrics
          CS_GPU({1}, "low_mem")
          //CS_CPU(-1)
    );

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    // Preprocessing
    x_train->div_(255.0f);

    // Train model
    fit(netf, {x_train}, {x_train}, batch_size, epochs);

}
