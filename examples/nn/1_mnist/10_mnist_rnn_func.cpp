/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"
#include "eddl/apis/eddlT.h"

using namespace eddl;

//////////////////////////////////
// mnist_rnn.cpp:
// A recurrent NN for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {
    // Download mnist
    download_mnist();

    // Settings
    int epochs = 10;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({28});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 32));
    l = RNN(l, 32, "relu");
    l = RNN(l, 32, "relu");

    l = LeakyReLu(Dense(l, 32));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});


    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          rmsprop(0.001), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1,1},100) // one GPU
          CS_CPU(-1) // CPU with maximum threads availables
    );

    // View model
    summary(net);


    // Load dataset
    tensor x_train = eddlT::load("mnist_trX.bin");
    tensor y_train = eddlT::load("mnist_trY.bin");
    tensor x_test = eddlT::load("mnist_tsX.bin");
    tensor y_test = eddlT::load("mnist_tsY.bin");



    // Preprocessing
    eddlT::div_(x_train, 255.0);
    eddlT::div_(x_test, 255.0);

    setlogfile(net,"recurrent_mnist");

    tensor x_train_batch=eddlT::create({batch_size,784});
    tensor y_train_batch=eddlT::create({batch_size,10});

    // Train model
    int num_batches=x_train->shape[0]/batch_size;
    for(int i=0;i<epochs;i++) {
      printf("Epoch %d\n",i+1);
      reset_loss(net);
      for(int j=0;j<num_batches;j++) {
        // get a batch
        next_batch({x_train,y_train},{x_train_batch,y_train_batch});

        x_train_batch->reshape_({batch_size,28,28}); // time x dim

        zeroGrads(net);
        forward(net,{x_train_batch});
        backward(net,{y_train_batch});
        update(net);


        print_loss(net,j);
        printf("\r");

        x_train_batch->reshape_({batch_size,784});

    }
    printf("\n");
  }

}
















/////
