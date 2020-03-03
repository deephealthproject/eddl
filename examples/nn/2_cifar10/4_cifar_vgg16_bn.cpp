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
// cifar_vgg16_bn.cpp:
// vgg16 with BatchNorm
// Using fit for training
//////////////////////////////////


layer Normalization(layer l)
{
  return GroupNormalization(l,4);
}

layer Block1(layer l,int filters) {
  return ReLu(Normalization(Conv(l,filters,{1,1},{1,1},"same",false)));
}
layer Block3_2(layer l,int filters) {
  l=ReLu(Normalization(Conv(l,filters,{3,3},{1,1},"same",false)));
  l=ReLu(Normalization(Conv(l,filters,{3,3},{1,1},"same",false)));
  return l;
}


int main(int argc, char **argv){

  // download CIFAR data
  download_cifar10();

  // Settings
  int epochs = 25;
  int batch_size = 4;
  int num_classes = 10;

  // network
  layer in=Input({3,32,32});
  layer l=in;

  // Data augmentation
  l = RandomCropScale(l, {0.8f, 1.0f});
  l = RandomFlip(l,1);

  l=MaxPool(Block3_2(l,64));
  l=MaxPool(Block3_2(l,128));
  l=MaxPool(Block1(Block3_2(l,256),256));
  l=MaxPool(Block1(Block3_2(l,512),512));
  l=MaxPool(Block1(Block3_2(l,512),512));

  l=Reshape(l,{-1});
  l=ReLu(BatchNormalization(Dense(l,512)));

  layer out=Softmax(Dense(l,num_classes));

  // net define input and output layers list
  model net=Model({in},{out});


  // Build model
  build(net,
    sgd(0.01, 0.9), // Optimizer
    {"soft_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    CS_GPU({1}, "full_mem") // GPU with only one gpu
	//CS_CPU(-1, "low_mem")  // CPU with maximum threads availables
  );

  // plot the model
  plot(net,"model.pdf","TB");  //Top Bottom plot

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
