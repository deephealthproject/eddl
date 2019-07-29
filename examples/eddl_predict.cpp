
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

int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 1;
    int batch_size = 1000;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Activation(Dense(l, 1024), "relu");
    l = Activation(Dense(l, 1024), "relu");
    l = Activation(Dense(l, 1024), "relu");
    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});

    // View model
    summary(net);
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU(4) // CPU with 4 threads
    );

    // Load dataset
    tensor x_train = T_load("trX.bin");
    tensor y_train = T_load("trY.bin");
    tensor x_test = T_load("tsX.bin");
    tensor y_test = T_load("tsY.bin");

    // Preprocessing
    div(x_train, 255.0);
    div(x_test, 255.0);

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    ///  Predict *one* sample
    auto *X=new float[1*784];
    tensor TX=T({1,784},X);

    auto *Y=new float[1*10];
    tensor TY=T({1,10},Y);

    predict(net,{TX},{TY});

    // The result is in float *Y
    // but in general you can get the pointer to
    // tensor data by:
    float *result=T_getptr(TY);


}


///////////
