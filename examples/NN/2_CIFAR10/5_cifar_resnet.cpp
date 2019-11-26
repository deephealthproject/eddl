/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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
// cifar_resnet.cpp:
// Resnet18 without BatchNorm
// Using fit for training
//////////////////////////////////

layer ResBlock(layer l, int filters,int nconv,int half) {
  layer in=l;

  if (half)
      l=ReLu(Conv(l,filters,{3,3},{2,2}));
  else
      l=ReLu(Conv(l,filters,{3,3},{1,1}));


  for(int i=0;i<nconv-1;i++)
    l=ReLu(Conv(l,filters,{3,3},{1,1}));

  if (half)
    return Sum(Conv(in,filters,{1,1},{2,2}),l);
  else
    return Sum(l,in);
}

int main(int argc, char **argv){

  // download CIFAR data
  download_cifar10();

  // Settings
  int epochs = 25;
  int batch_size = 100;
  int num_classes = 10;

  // network
  layer in=Input({3,32,32});
  layer l=in;

  l=ReLu(Conv(l,64,{3,3},{1,1}));

  l=ResBlock(l, 64,2,1);//<<<-- output half size
  l=ResBlock(l, 64,2,0);

  l=ResBlock(l, 128,2,1);//<<<-- output half size
  l=ResBlock(l, 128,2,0);

  l=ResBlock(l, 256,2,1);//<<<-- output half size
  l=ResBlock(l, 256,2,0);

  l=ResBlock(l, 512,2,1);//<<<-- output half size
  l=ResBlock(l, 512,2,0);

  l=Reshape(l,{-1});
  l=Activation(Dense(l,512),"relu");

  layer out=Activation(Dense(l,num_classes),"softmax");

  // net define input and output layers list
  model net=Model({in},{out});


  // Build model
  build(net,
    sgd(0.01, 0.9), // Optimizer
    {"soft_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    CS_CPU() // CPU with maximum threads availables
    //CS_GPU({1}) // GPU with only one gpu
  );

  // plot the model
  plot(net,"model.pdf");

  // get some info from the network
  summary(net);

  // Load and preprocess training data
  tensor x_train = eddlT::load("cifar_trX.bin");
  tensor y_train = eddlT::load("cifar_trY.bin");
  eddlT::div_(x_train, 255.0);

  // Load and preprocess test data
  tensor x_test = eddlT::load("cifar_tsX.bin");
  tensor y_test = eddlT::load("cifar_tsY.bin");
  eddlT::div_(x_test, 255.0);

  for(int i=0;i<epochs;i++) {
    // training, list of input and output tensors, batch, epochs
    fit(net,{x_train},{y_train},batch_size, 1);
    // Evaluate train
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net,{x_test},{y_test});
  }


}


///////////
