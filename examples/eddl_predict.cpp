
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
#include "eddl.h"

int main(int argc, char **argv) {

    // Download dataset
    eddl.download_mnist();

    // Settings
    int epochs = 1;
    int batch_size = 1000;
    int num_classes = 10;

    // Define network
    layer in = eddl.Input({784});
    layer l = in;  // Aux var

    l = eddl.Activation(eddl.Dense(l, 1024), "relu");
    l = eddl.Activation(eddl.Dense(l, 1024), "relu");
    l = eddl.Activation(eddl.Dense(l, 1024), "relu");
    layer out = eddl.Activation(eddl.Dense(l, num_classes), "softmax");
    model net = eddl.Model({in}, {out});

    // View model
    eddl.summary(net);
    eddl.plot(net, "model.pdf");

    // Build model
    eddl.build(net,
               eddl.sgd(0.01, 0.9), // Optimizer
               {eddl.LossFunc("soft_cross_entropy")}, // Losses
               {eddl.MetricFunc("categorical_accuracy")}, // Metrics
               eddl.CS_CPU(4) // CPU with 4 threads
    );

    // Load dataset
    tensor x_train = eddl.T("trX.bin");
    tensor y_train = eddl.T("trY.bin");
    tensor x_test = eddl.T("tsX.bin");
    tensor y_test = eddl.T("tsY.bin");

    // Preprocessing
    eddl.div(x_train, 255.0);
    eddl.div(x_test, 255.0);

    // Train model
    eddl.fit(net, {x_train}, {y_train}, batch_size, epochs);

    ///  Predict *one* sample
    auto *X=new float[1*784];
    tensor TX=eddl.T({1,784},X);

    auto *Y=new float[1*10];
    tensor TY=eddl.T({1,10},Y);

    eddl.predict(net,{TX},{TY});

    // The result is in float *Y
    // but in general you can get the pointer to
    // tensor data by:
    float *result=eddl.T_getptr(TY);


}


///////////
