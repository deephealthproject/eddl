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
// cifar_resnet50_da_bn.cpp:
// Resnet50 with
// BatchNorm
// Data Augmentation
// Using fit for training
//////////////////////////////////

layer BN(layer l)
{
  return BatchNormalization(l);
  //return l;
}

layer BG(layer l) {
  //return GaussianNoise(BN(l),0.3);
  return BN(l);
}


layer ResBlock(layer l, int filters,int half, int expand=0) {
  layer in=l;

  l=ReLu(BG(Conv(l,filters,{1,1},{1,1},"same",false)));

  if (half)
    l=ReLu(BG(Conv(l,filters,{3,3},{2,2},"same",false)));
  else
    l=ReLu(BG(Conv(l,filters,{3,3},{1,1},"same",false)));

  l=BG(Conv(l,4*filters,{1,1},{1,1},"same",false));

  if (half)
    return ReLu(Sum(BG(Conv(in,4*filters,{1,1},{2,2},"same",false)),l));
  else
    if (expand) return ReLu(Sum(BG(Conv(in,4*filters,{1,1},{1,1},"same",false)),l));
    else return ReLu(Sum(in,l));
}

int main(int argc, char **argv){

  // download CIFAR data
  download_cifar10();

  // Settings
  int epochs = 150;
  int batch_size =128;
  int num_classes = 10;

  // network
  layer in=Input({3,32,32});
  layer l=in;

  // Data augmentation

  l = RandomCropScale(l, {0.8f, 1.0f});

  // l = RandomShift(l,{-0.1,0.1},{-0.1,0.1});//CropScale(l, {0.8f, 1.0f});
  l = RandomHorizontalFlip(l);
  //l = RandomCutout(l, {0.1,0.3},{0.1,0.3});

  // Resnet-50

  // For Imagenet size:

  l=ReLu(BG(Conv(l,64,{3,3},{2,2},"same",false))); //{1,1}

  for(int i=0;i<3;i++)
    l=ResBlock(l, 64, 0, i==0); // not half but expand the first

  for(int i=0;i<4;i++)
    l=ResBlock(l, 128,i==0);

  for(int i=0;i<6;i++)
    l=ResBlock(l, 256,i==0);

  for(int i=0;i<3;i++)
    l=ResBlock(l,512,i==0);

  l=MaxPool(l,{2,2});  // should be avgpool

  l=Reshape(l,{-1});

  layer out=Activation(Dense(l,num_classes),"softmax");

  // net define input and output layers list
  model net=Model({in},{out});


  // Build model
  build(net,
	sgd(0.001,0.9), // Optimizer
    {"soft_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
	//CS_GPU({1,1},250,"full_mem")// GPU with only one gpu
    CS_CPU(-1)  // CPU with maximum threads availables
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


  float lr=0.01;
  for(int j=0;j<3;j++) {
    lr/=10.0;

    setlr(net,{lr,0.9});

    for(int i=0;i<epochs;i++) {
      // training, list of input and output tensors, batch, epochs
      fit(net,{x_train},{y_train},batch_size, 1);

      // Evaluate test
      std::cout << "Evaluate test:" << std::endl;
      evaluate(net,{x_test},{y_test});
    }
  }



}


///////////
