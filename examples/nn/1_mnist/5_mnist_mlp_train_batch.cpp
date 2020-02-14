/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
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

//////////////////////////////////
// mnist_mlp_train_batch.cpp:
// A very basic MLP for mnist
// Using train_batch for training
// and eval_batch fot test
//////////////////////////////////

int main(int argc, char **argv) {

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 1;
    int batch_size = 128;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}, "low_mem") // one GPU
          //CS_CPU(-1, "low_mem") // CPU with maximum threads availables
    );

    // View model
    summary(net);

    setlogfile(net,"mnist");

    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");
    tensor x_test = eddlT::load("tsX.bin");
    tensor y_test = eddlT::load("tsY.bin");

    tensor xbatch = eddlT::create({batch_size,784});
    tensor ybatch = eddlT::create({batch_size,10});

    // Preprocessing
    eddlT::div_(x_train, 255.0);
    eddlT::div_(x_test, 255.0);


    // Train model
    int i,j;
    tshape s=eddlT::getShape(x_train);
    int num_batches=s[0]/batch_size;

    for(i=0;i<epochs;i++) {
      reset_loss(net);
      fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
      for(j=0;j<num_batches;j++)  {

        next_batch({x_train,y_train},{xbatch,ybatch});
        train_batch(net, {xbatch}, {ybatch});

        //OR:
        //vector<int> indices = random_indices(batch_size, s[0]);
        //train_batch(net, {x_train}, {y_train}, indices);

        print_loss(net,j);
        printf("\r");
      }
      printf("\n");
    }


    // Evaluate model
    printf("Evaluate:\n");
    s=eddlT::getShape(x_test);
    num_batches=s[0]/batch_size;

    reset_loss(net);
    for(j=0;j<num_batches;j++)  {
        vector<int> indices(batch_size);
        for(int i=0;i<indices.size();i++)
          indices[i]=(j*batch_size)+i;

        eval_batch(net, {x_test}, {y_test}, indices);

        print_loss(net,j);
        printf("\r");
      }

    //last batch
    if (s[0]%batch_size) {
      int last_batch_size=s[0]%batch_size;
      vector<int> indices(last_batch_size);
      for(int i=0;i<indices.size();i++)
        indices[i]=(j*batch_size)+i;

      eval_batch(net, {x_test}, {y_test}, indices);

      print_loss(net,j);
      printf("\r");
    }
    printf("\n");




}
