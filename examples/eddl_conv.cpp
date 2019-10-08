
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"

using namespace eddl;


layer Block(layer l,int filters, vector<int> kernel, vector<int> stride)
{
  return MaxPool(BatchNormalization(Activation(Conv(l, filters, kernel,stride),"relu")),{2,2});
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

  //l = GaussianNoise(l,0.3);

  l=Reshape(l,{1,28,28});
    l=UpSampling(l, {2,2});
//  l=Block(l,16,{3,3},{1,1});
//  l=Block(l,32,{3,3},{1,1});
//  l=Block(l,64,{3,3},{1,1});
//  l=Block(l,128,{3,3},{1,1});

  l=Reshape(l,{-1});

  l=Activation(Dense(l,256),"relu");

  layer out=Activation(Dense(l,num_classes),"softmax");

  // net define input and output layers list
  model net=Model({in},{out});

  // plot the model
  plot(net,"model.pdf");

  // get some info from the network
  cout << summary(net) << endl;

  // Build model
  build(net,
    sgd(0.01, 0.9), // Optimizer
    {"soft_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    //CS_CPU(1) // CPU with 4 threads
    CS_GPU({1}) // GPU with only one gpu
  );

  // Load and preprocess training data
  tensor X=T_load("trX.bin");
  tensor Y=T_load("trY.bin");
  div(X,255.0);

  // training, list of input and output tensors, batch, epochs
  fit(net,{X},{Y},batch_size, epochs);

  // Evaluate train
  std::cout << "Evaluate train:" << std::endl;
  evaluate(net,{X},{Y});

  // Load and preprocess test data
  tensor tX=T_load("tsX.bin");
  tensor tY=T_load("tsY.bin");
  div(tX,255.0);

  // Evaluate test
  std::cout << "Evaluate test:" << std::endl;
  evaluate(net,{tX},{tY});

}


///////////
