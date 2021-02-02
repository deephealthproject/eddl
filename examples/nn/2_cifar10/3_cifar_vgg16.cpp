/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
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
// cifar_vgg16.cpp:
// VGG-16
// Using fit for training
//////////////////////////////////
layer Block1(layer l,int filters) {
  return ReLu(Conv(l,filters,{1,1},{1,1}));
}
layer Block3_2(layer l,int filters) {
  l=ReLu(Conv(l,filters,{3,3},{1,1}));
  l=ReLu(Conv(l,filters,{3,3},{1,1}));
  return l;
}

int main(int argc, char **argv){

  // download CIFAR data
  download_cifar10();

  // Settings
  int epochs = 5;
  int batch_size = 100;
  int num_classes = 10;

  // network
  layer in=Input({3,32,32});
  layer l=in;


  l=MaxPool(Block3_2(l,64));
  l=MaxPool(Block3_2(l,128));
  l=MaxPool(Block1(Block3_2(l,256),256));
  l=MaxPool(Block1(Block3_2(l,512),512));
  l=MaxPool(Block1(Block3_2(l,512),512));

  l=Reshape(l,{-1});
  l=Activation(Dense(l,512),"relu");

  layer out= Softmax(Dense(l, num_classes));

  // net define input and output layers list
  model net=Model({in},{out});


  // Build model
  build(net,
    sgd(0.01, 0.9), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    CS_GPU({1}) // one GPU
    //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
    //CS_CPU()
  );



  setlogfile(net,"vgg16");

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



  for(int i=0;i<epochs;i++) {
    // training, list of input and output tensors, batch, epochs
    fit(net,{x_train},{y_train},batch_size, 1);
    // Evaluate test
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net,{x_test},{y_test});
  }


}


///////////
