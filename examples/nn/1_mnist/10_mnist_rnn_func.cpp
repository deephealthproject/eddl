/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


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
    int epochs = 5;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({28});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 32));
    l = LSTM(l, 32, "relu");
    l = LSTM(l, 32, "relu");

    l = LeakyReLu(Dense(l, 32));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});


    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          rmsprop(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
	  //CS_FPGA({1})
    );

    // View model
    summary(net);


    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");


    // Preprocessing
    x_train->div_(255.0);
    x_test->div_(255.0);

    setlogfile(net,"recurrent_mnist");

    Tensor* x_train_batch=new Tensor({batch_size,784});
    Tensor* y_train_batch=new Tensor({batch_size,10});

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
