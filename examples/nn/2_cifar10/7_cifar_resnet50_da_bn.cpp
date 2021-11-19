/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


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
    return ReLu(Add(BG(Conv(in,4*filters,{1,1},{2,2},"same",false)),l));
  else
    if (expand) return ReLu(Add(BG(Conv(in,4*filters,{1,1},{1,1},"same",false)),l));
    else return ReLu(Add(in,l));
}

int main(int argc, char **argv){
  bool testing = false;
  bool use_cpu = false;
  for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "--testing") == 0) testing = true;
      else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
  }

  // download CIFAR data
  download_cifar10();

  // Settings
  int epochs = testing ? 2 : 5;
  int batch_size =16;
  int num_classes = 10;

  // network
  layer in=Input({3,32,32});
  layer l=in;

  // Data augmentation

  l = RandomCropScale(l, {0.8f, 1.0f});
  l = RandomHorizontalFlip(l);

  // Resnet-50

  l=ReLu(BG(Conv(l,64,{3,3},{1,1},"same",false))); //{1,1}
  //l=MaxPool(l,{3,3},{1,1},"same");

  // Add explicit padding to avoid the asymmetric padding in the Conv layers
  l = Pad(l, {0, 1, 1, 0});

  for(int i=0;i<3;i++)
    l=ResBlock(l, 64, 0, i==0); // not half but expand the first

  for(int i=0;i<4;i++)
    l=ResBlock(l, 128,i==0);

  for(int i=0;i<6;i++)
    l=ResBlock(l, 256,i==0);

  for(int i=0;i<3;i++)
    l=ResBlock(l,512,i==0);

  l=MaxPool(l,{4,4});  // should be avgpool

  l=Reshape(l,{-1});

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
  plot(net,"model.pdf","TB");  // TB --> Top-Bottom mode for dot (graphviz)

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

  if (testing) {
      std::string _range_ = "0:" + std::to_string(2 * batch_size);
      Tensor* x_mini_train = x_train->select({_range_, ":"});
      Tensor* y_mini_train = y_train->select({_range_, ":"});
      Tensor* x_mini_test  = x_test->select({_range_, ":"});
      Tensor* y_mini_test  = y_test->select({_range_, ":"});

      delete x_train;
      delete y_train;
      delete x_test;
      delete y_test;

      x_train = x_mini_train;
      y_train = y_mini_train;
      x_test  = x_mini_test;
      y_test  = y_mini_test;
  }

  float lr=0.001;
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
  delete x_train;
  delete y_train;
  delete x_test;
  delete y_test;
  delete net;

  return EXIT_SUCCESS;
}
