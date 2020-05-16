/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
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
// cifar_resnet_da_bn.cpp:
// Resnet18 with
// BatchNorm
// Data Augmentation
// Using fit for training
//////////////////////////////////

layer BG(layer l) {
  return GaussianNoise(BatchNormalization(l),0.3);
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
    return Sum(BG(Conv(in,filters,{1,1},{2,2})),l);
  else
    return Sum(l,in);
}

int main(int argc, char **argv){

  // download CIFAR data
  download_cifar10();

  // Settings
  int epochs = 500;
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

  layer out=Softmax(Dense(l,num_classes));

  // net define input and output layers list
  model net=Model({in},{out});


  // Build model
  build(net,
    sgd(0.01, 0.9), // Optimizer
    {"soft_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    CS_GPU({1}) // one GPU
    //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
    //CS_CPU()
  );

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

  for(int i=0;i<epochs;i++) {
    // training, list of input and output tensors, batch, epochs
    fit(net,{x_train},{y_train},batch_size, 1);
    // Evaluate train
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net,{x_test},{y_test});
  }


}


///////////
