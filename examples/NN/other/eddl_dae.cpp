/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"
#include "apis/eddlT.h"

using namespace eddl;


// DENOISSING-AUTOENCODER
int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 100;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    // Inject noise in the input
    l=GaussianNoise(l,0.5);

    l = Activation(Dense(l, 256), "relu");
    l = Activation(Dense(l, 128), "relu");
    l = Activation(Dense(l, 64), "relu");
    l = Activation(Dense(l, 128), "relu");
    l = Activation(Dense(l, 256), "relu");
    layer out = Dense(l, 784);

    model net = Model({in}, {out});

    // View model
    summary(net);
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"mean_squared_error"}, // Losses
          {"mean_squared_error"}, // Metrics
          CS_CPU()
          //CS_GPU({1})
    );

    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    // Preprocessing
    eddlT::div_(x_train, 255.0);

    // Train model
    fit(net, {x_train}, {x_train}, batch_size, epochs);
}


///////////
