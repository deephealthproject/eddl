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


int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 100;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var


    l = BatchNormalization(Activation(L2(Dense(l, 1024),0.0001f), "relu"));
    l = BatchNormalization(Activation(L2(Dense(l, 1024),0.0001f), "relu"));
    l = BatchNormalization(Activation(L2(Dense(l, 1024),0.0001f), "relu"));
    //l = BatchNormalization(Activation(Dense(l, 1024, true, L2(0.0001f)), "relu"));
    
    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});

    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1,1},10) // 2 GPUs with local_sync_batches=10
          //CS_GPU({1}) // 1 GPU
          //CS_CPU(4) // 4 CPU threads
          CS_CPU() // CPU with maximum threads availables
          //CS_COMPSS("../config/compss/resources.xml")
    );

    // View model
    cout<<summary(net);


    // Load dataset
    tensor x_train = T_load("trX.bin");
    tensor y_train = T_load("trY.bin");
    tensor x_test = T_load("tsX.bin");
    tensor y_test = T_load("tsY.bin");

    // Preprocessing
    div(x_train, 255.0);
    div(x_test, 255.0);

    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 1);

      // Evaluate test
      std::cout << "Evaluate test:" << std::endl;
      evaluate(net, {x_test}, {y_test});
    }
}


///////////
