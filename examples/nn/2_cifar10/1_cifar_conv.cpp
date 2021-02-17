/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// cifar_conv.cpp:
// A very basic Conv for cifar10
// Using fit for training
//////////////////////////////////

layer Normalization(layer l)
{

  return l;
  //return BatchNormalization(l);
}

int main(int argc, char **argv){
    bool testing = false;
    bool use_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

  // download CIFAR data
  download_cifar10();

  // Settings
  int epochs = testing ? 2 : 3;
  int batch_size = 100;
  int num_classes = 10;

  // network
  layer in=Input({3,32,32});
  layer l=in;

  l=MaxPool(ReLu(Normalization(Conv(l,32,{3,3},{1,1}))),{2,2});
  l=MaxPool(ReLu(Normalization(Conv(l,64,{3,3},{1,1}))),{2,2});
  l=MaxPool(ReLu(Normalization(Conv(l,128,{3,3},{1,1}))),{2,2});
  l=MaxPool(ReLu(Normalization(Conv(l,256,{3,3},{1,1}))),{2,2});

  l=GlobalMaxPool(l);


  l=Flatten(l);

  l=Activation(Dense(l,128),"relu");

  layer out= Softmax(Dense(l, num_classes));

  // net define input and output layers list
  model net=Model({in},{out});

  compserv cs = nullptr;
  if (use_cpu) {
      cs = CS_CPU();
  } else {
      cs = CS_GPU({1}); // one GPU
      // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
      // cs = CS_CPU();
      // cs = CS_FPGA({1});
  }

  // Build model
  build(net,
    adam(0.001), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    cs);

  // plot the model
  plot(net,"model.pdf");

  // get some info from the network
  summary(net);

  // Load and preprocess training data
  Tensor* x_train = Tensor::load("cifar_trX.bin");
  Tensor* y_train = Tensor::load("cifar_trY.bin");
  x_train->div_(255.0f);

  // Load and preprocess test data
  Tensor* x_test = Tensor::load("cifar_tsX.bin");
  Tensor* y_test = Tensor::load("cifar_tsY.bin");
  x_test->div_(255.0f);

  if (testing) {
      std::string _range_ = "0:" + std::to_string(2 * batch_size);
      Tensor* x_mini_train = x_train->select({_range_, ":"});
      Tensor* y_mini_train = y_train->select({_range_, ":"});
      Tensor* x_mini_test  = x_test->select({_range_, ":"});
      Tensor* y_mini_test  = y_test->select({_range_, ":"});

      delete x_train;
      delete y_train;
      delete x_test;
      delete y_test;

      x_train = x_mini_train;
      y_train = y_mini_train;
      x_test  = x_mini_test;
      y_test  = y_mini_test;
  }

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Evaluate
    evaluate(net, {x_test}, {y_test});

//  for(int i=0;i<epochs;i++) {
//    // training, list of input and output tensors, batch, epochs
//    fit(net,{x_train},{y_train},batch_size, 1);
//
//    // Evaluate train
//    std::cout << "Evaluate test:" << std::endl;
//    evaluate(net,{x_test},{y_test});
//  }

  delete x_train;
  delete y_train;
  delete x_test;
  delete y_test;
  delete net;

  return EXIT_SUCCESS;
}
