
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "eddl.h"


// DENOISSING-AUTOENCODER

int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 1000;

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
               {LossFunc("mean_squared_error")}, // Losses
               {MetricFunc("mean_squared_error")}, // Metrics
               CS_CPU(4) // CPU with 4 threads
    );

    // Load dataset
    tensor x_train = T_load("trX.bin");
    // Preprocessing
    div(x_train, 255.0);

    // Train model
    fit(net, {x_train}, {x_train}, batch_size, epochs);


}


///////////
