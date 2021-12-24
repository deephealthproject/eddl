/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// cifar_resnet_da_bn.cpp:
// Resnet18 with
// BatchNorm
// Data Augmentation
// Using fit for training
//////////////////////////////////

layer BG(layer l) {
  return GaussianNoise(BatchNormalization(l),0.3);
  //return l;
}

layer ResBlock(layer l, int filters,int nconv,int half) {
  layer in=l;

  if (half)
      l=ReLu(BG(Conv(l,filters,{3,3},{2,2})));
  else
      l=ReLu(BG(Conv(l,filters,{3,3},{1,1})));


  for(int i=0;i<nconv-1;i++)
    l=ReLu(BG(Conv(l,filters,{3,3},{1,1})));

  if (half)
    return Add(BG(Conv(in,filters,{1,1},{2,2})),l);
  else
    return Add(l,in);
}

int main(int argc, char **argv){
  bool testing = false;
  bool use_cpu = false;
  int id=0;
    
  id = init_distributed();

  for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "--testing") == 0) testing = true;
      else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
  }

  // download CIFAR data
  download_cifar10();

  // Settings
  int epochs = testing ? 2 : 5;
  int batch_size = 100;
  int num_classes = 10;

  // network
  layer in=Input({3,32,32});
  layer l=in;

  // Data augmentation
  l = RandomCropScale(l, {0.8f, 1.0f});
  l = RandomFlip(l,1);

  // Resnet-18
  l=ReLu(BG(Conv(l,64,{3,3},{1,1})));

  // Add explicit padding to avoid the asymmetric padding in the Conv layers
  l = Pad(l, {0, 1, 1, 0});

  l=ResBlock(l, 64,2,1);//<<<-- output half size
  l=ResBlock(l, 64,2,0);

  l=ResBlock(l, 128,2,1);//<<<-- output half size
  l=ResBlock(l, 128,2,0);

  l=ResBlock(l, 256,2,1);//<<<-- output half size
  l=ResBlock(l, 256,2,0);

  l=ResBlock(l, 256,2,1);//<<<-- output half size
  l=ResBlock(l, 256,2,0);

  l=Reshape(l,{-1});
  l=ReLu(BG(Dense(l,512)));

  layer out= Softmax(Dense(l, num_classes));

  // net define input and output layers list
  model net=Model({in},{out});

  compserv cs = nullptr;
  if (use_cpu) {
      cs = CS_CPU();
  } else {
      cs = CS_GPU(); // one GPU
      // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
      // cs = CS_CPU();
      // cs = CS_FPGA({1});
  }

 // Build model
  build(net,
    sgd(lr, 0.9), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    cs);

  // plot the model
  plot(net,"model.pdf","TB");

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

  for(int i=0;i<epochs;i++) {
    // training, list of input and output tensors, batch, epochs
    fit(net,{x_train},{y_train},batch_size, 1);
    // Evaluate train
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net,{x_test},{y_test});
  }

  delete x_train;
  delete y_train;
  delete x_test;
  delete y_test;
  delete net;
  
  end_distributed();

  return EXIT_SUCCESS;
}


///////////
