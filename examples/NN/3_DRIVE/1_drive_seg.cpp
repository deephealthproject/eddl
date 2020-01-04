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
// Drive segmentation
// https://drive.grand-challenge.org/DRIVE/
//////////////////////////////////


int main(int argc, char **argv){

  // download CIFAR data
  //download_cifar10();

  // Settings
  int epochs = 25;
  int batch_size = 100;


  // network
  /*
  layer in=Input({3,32,32});
  layer l=in;


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
  plot(net,"model.pdf");

  // get some info from the network
  summary(net);
*/
// Load and preprocess training data
cout<<"Reading train numpy\n";
tensor x_train = eddlT::load("x_train.npy");
cout<<"Reading test numpy\n";
tensor y_train = eddlT::load("y_train.npy");

x_train->info();
x_train->print();
getchar();

y_train->info();

y_train->print();
getchar();


eddlT::reshape_(y_train,{20,1,584,565});
y_train->info();

tensor t_output = y_train->select({"0", ":", ":", ":"});

t_output->info();
t_output->print();
//t_output->save("np_select.png");


}
