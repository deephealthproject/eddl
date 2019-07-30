
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
          CS_CPU(4) // CPU with 4 threads
    );

    // Load dataset
    tensor x_train = T_load("trX.bin");
    tensor y_train = T_load("trY.bin");

    // Preprocessing
    div(x_train, 255.0);

    // Train model
    fit(net, {x_train}, {x_train,y_train}, batch_size, epochs);


}


///////////
