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
#include "apis/eddlT.h"


using namespace eddl;

// DENOISSING-AUTOENCODER
int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 1000;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    // Inject noise in the input
    l=GaussianNoise(l,0.5);

    l = Activation(Dense(l, 256), "relu");
    l = Activation(Dense(l, 128), "relu");
    layer lc= l = Activation(Dense(l, 64), "relu");

    // Autoencoder branch
    l = Activation(Dense(l, 128), "relu");
    l = Activation(Dense(l, 256), "relu");
    layer outdae = Dense(l, 784);

    // Classification branch
    layer outclass = Activation(Dense(lc, num_classes), "softmax");

    // model with two outpus
    model net = Model({in}, {outdae,outclass});

    // View model
    summary(net);
    plot(net, "model.pdf");

    // Build model with two losses and metrics
    build(net,
          sgd(0.001, 0.9), // Optimizer
          {"mean_squared_error","soft_cross_entropy"}, // Losses
          {"mean_squared_error","categorical_accuracy"}, // Metrics
          //CS_CPU()
          CS_GPU({1})
    );

    /// Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");

    // Preprocessing
    eddlT::div_(x_train, 255.0);


    // Train model
    fit(net, {x_train}, {x_train,y_train}, batch_size, epochs);


}


///////////
