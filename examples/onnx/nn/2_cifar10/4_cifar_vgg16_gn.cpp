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

#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed

using namespace eddl;

//////////////////////////////////
// cifar_vgg16_bn.cpp:
// vgg16 with BatchNorm
// Using fit for training
//////////////////////////////////

layer Block1(layer l,int filters) {
  return ReLu(GroupNormalization(Conv(l,filters,{1,1},{1,1}),4));
}
layer Block3_2(layer l,int filters) {
  l=ReLu(GroupNormalization(Conv(l,filters,{3,3},{1,1}),4));
  l=ReLu(GroupNormalization(Conv(l,filters,{3,3},{1,1}),4));
  return l;
}


int main(int argc, char **argv){

  // download CIFAR data
  download_cifar10();

  // Settings
  int epochs = 5;
  int batch_size = 8; // very small batch to test GroupNormalization
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

  layer out=Activation(Dense(l,num_classes),"softmax", {1});

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
  plot(net,"model.pdf","TB");  //Top Bottom plot

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
  // Export
  string path("trained_model.onnx");
  save_net_to_onnx_file(net, path);

  cout << "Saved net to onnx file" << endl;

  // Import 
  Net* imported_net = import_net_from_onnx_file(path, DEV_CPU);
	
  build(imported_net,
        adam(0.001), // Optimizer
        {"soft_cross_entropy"}, // Losses
        {"categorical_accuracy"}, // Metrics
	    CS_GPU({1}, "low_mem"), // one GPU
        //CS_CPU(), // CPU with maximum threads availables
	    false       // Parameter that indicates that the weights of the net must be initialized to random values.
  );

  cout << "Evaluating with imported net" << endl;
  evaluate(imported_net, {x_test}, {y_test});



}


///////////
