/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
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


layer Block(layer l,int filters, vector<int> kernel, vector<int> stride)
{
  return MaxPool(Activation(Conv(l, filters, kernel,stride),"relu"),{2,2});
}

int main(int argc, char **argv){
  // download MNIST data
  download_mnist();

  // Settings
  int epochs = 5;
  int batch_size = 100;
  int num_classes = 10;

  // network
  layer in=Input({784});
  layer l=in;

  l=Reshape(l,{1,28,28});
  l=Block(l,16,{3,3},{1,1});
  l=Block(l,32,{3,3},{1,1});
  l=Block(l,64,{3,3},{1,1});
  l=Block(l,128,{3,3},{1,1});

  l=Reshape(l,{-1});

  l=Activation(Dense(l,64),"relu");

  layer out=Activation(Dense(l,num_classes),"softmax");

  // net define input and output layers list
  model net=Model({in},{out});


  // Build model
  build(net,
    sgd(0.01, 0.9), // Optimizer
    {"soft_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    //CS_CPU(4) // 4 CPU threads
    CS_CPU() // CPU with maximum threads availables
    //CS_GPU({1}) // GPU with only one gpu
  );

  // plot the model
  plot(net,"model.pdf");

  // get some info from the network
  cout << summary(net) << endl;

  // Load and preprocess training data
  // Load dataset
  tensor x_train = eddlT::load("trX.bin");
  tensor y_train = eddlT::load("trY.bin");
  eddlT::div_(x_train, 255.0);


  // training, list of input and output tensors, batch, epochs
  fit(net,{x_train},{y_train},batch_size, epochs);

  // Evaluate train
  std::cout << "Evaluate train:" << std::endl;
  evaluate(net,{x_train},{y_train});

  // Load and preprocess test data
  tensor x_test = eddlT::load("tsX.bin");
  tensor y_test = eddlT::load("tsY.bin");
  eddlT::div_(x_test, 255.0);


  // Evaluate test
  std::cout << "Evaluate test:" << std::endl;
  evaluate(net,{x_test},{y_test});

}


///////////
