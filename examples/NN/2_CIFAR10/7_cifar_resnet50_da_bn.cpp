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
// cifar_resnet50_da_bn.cpp:
// Resnet50 with
// BatchNorm
// Data Augmentation
// Using fit for training
//////////////////////////////////

layer BN(layer l)
{
  return BatchNormalization(l);
}

layer BG(layer l) {
  return GaussianNoise(BN(l),0.3);
}

layer ResBlock(layer l, int filters,int half) {
  layer in=l;

  if (half)
      l=ReLu(BG(Conv(l,filters,{1,1},{2,2})));
  else
      l=ReLu(BG(Conv(l,filters,{1,1},{1,1})));


  l=ReLu(BG(Conv(l,filters,{3,3},{1,1})));
  l=ReLu(BG(Conv(l,4*filters,{1,1},{1,1})));

  if (half)
    return Sum(BG(Conv(in,4*filters,{1,1},{2,2})),l);
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

  // Data augmentation
  l = CropScaleRandom(l, {0.8f, 1.0f});
  l = Flip(l,1);

  // Resnet-50
  l=ReLu(BG(Conv(l,64,{3,3},{1,1})));

  for(int i=0;i<3;i++)
    l=ResBlock(l, 64,i==0);

  for(int i=0;i<4;i++)
    l=ResBlock(l, 128,i==0);

  for(int i=0;i<6;i++)
    l=ResBlock(l, 256,i==0);

  for(int i=0;i<3;i++)
    l=ResBlock(l, 256,i==0); // <-- should be 512, check MAX_THR in gpu problem

  l=Reshape(l,{-1});
  l=ReLu(BG(Dense(l,512)));

  layer out=Activation(Dense(l,num_classes),"softmax");

  // net define input and output layers list
  model net=Model({in},{out});


  // Build model
  build(net,
    sgd(0.01, 0.9), // Optimizer
    {"soft_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    //CS_CPU() // CPU with maximum threads availables
    CS_GPU({1}) // GPU with only one gpu
  );

  // plot the model
  plot(net,"model.pdf","TB");  // TB --> Top-Bottom mode for dot (graphviz)

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
    // Evaluate test
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net,{x_test},{y_test});
  }


}


///////////
