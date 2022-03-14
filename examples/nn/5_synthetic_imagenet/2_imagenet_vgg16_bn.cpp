/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// cifar_vgg16_bn.cpp:
// vgg16 with BatchNorm
// Using fit for training
//////////////////////////////////


layer Normalization(layer l)
{
  return BatchNormalization(l);
  //return LayerNormalization(l);
  //return GroupNormalization(l,8);
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
  bool testing = false;
  bool use_cpu = false;
  for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "--testing") == 0) testing = true;
      else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
  }


  // Settings
  int epochs = testing ? 2 : 5;
  int batch_size = testing ? 2 : 16;
  int num_classes = 1000;
//This factor indicates the number of batches of batch_size for the execution
  //If the NODE/GPU runs out of memory, just decrease the factor valur
  int factor = testing ? 1 : 20000;


  // network
  layer in=Input({3,224,224});
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
  l=ReLu(Dense(l,4096));

  layer out= Softmax(Dense(l, num_classes));

  // net define input and output layers list
  model net=Model({in},{out});

  compserv cs = nullptr;
  if (use_cpu) {
      cs = CS_CPU();
  } else {
      cs = CS_GPU({1}); // one GPU
      // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
      // cs = CS_CPU();
      // cs = CS_FPGA({1});
  }

  // Build model
  build(net,
      adam(0.001), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    cs);

  // plot the model
  plot(net,"model.pdf","TB");  //Top Bottom plot

  // get some info from the network
  summary(net);

  Tensor * x_train = new Tensor({batch_size*factor,3, 224,224});
  Tensor * y_train = new Tensor({batch_size*factor,1000});

  for(int i=0;i<epochs;i++) {
    // training, list of input and output tensors, batch, epochs
    fit(net,{x_train},{y_train},batch_size, 1);
  }

    delete x_train;
    delete y_train;
    delete net;

    return EXIT_SUCCESS;
}


///////////
