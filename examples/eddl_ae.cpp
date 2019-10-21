/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"

using namespace eddl;

// AUTOENCODER
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
    l = Activation(Dense(l, 64), "relu");
    l = Activation(Dense(l, 128), "relu");
    l = Activation(Dense(l, 256), "relu");

    layer out = Dense(l, 784);

    model net = Model({in}, {out});

    // View model
  

    cout<<summary(net);
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.001, 0.9), // Optimizer
          {"mean_squared_error"}, // Losses
          {"mean_squared_error"}, // Metrics
          CS_CPU(4) // CPU with 4 threads
          //CS_GPU({1})
    );

    // Load dataset
    tensor x_train = T_load("trX.bin");
    // Preprocessing
    div(x_train, 255.0);

    // Train model
    fit(net, {x_train}, {x_train}, batch_size, epochs);


}


///////////
