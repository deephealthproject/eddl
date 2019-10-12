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
    int epochs = 1;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Activation(Dense(l, 1024), "relu");
    l = Activation(Dense(l, 1024), "relu");
    l = Activation(Dense(l, 1024), "relu");
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
          CS_CPU(4) // 4 CPU threads
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


    // Prepare data
    // Input/target data {x1, x2,...} => {y1, y2,...}
    vector<Tensor *> tin{x_train->data};  // Get input from LTensor (LTensor->data)
    vector<Tensor *> tout{y_train->data}; // Get input from LTensor (LTensor->data)

    int num_samples = tin[0]->shape[0];  //arg1
    int num_batches = num_samples / batch_size; //arg2

    // Set batch size
    resize_model(net, batch_size);  // Bind this function

    // Start training
    set_mode(net, TRMODE);  // Bind this function


    // Train model (fine-grained)
    for(int i=0;i<epochs;i++) {

        // For each batch
        for (int j = 0; j < num_batches; j++) {
            fprintf(stdout, "Epoch %d/%d (batch %d/%d)\n", i + 1, epochs, j+1, num_batches);

            // Set random indices
            vector<int> indices = random_indices(batch_size, num_samples); // Should declared from python

            // COMPS: wait for weights()

            // Train batch
            train_batch(net, tin, tout, indices);  // Bind this function

            // COMPS: send grads()
        }
    }


      // Evaluate test
      std::cout << "Evaluate test:" << std::endl;
      evaluate(net, {x_test}, {y_test});
}


///////////
