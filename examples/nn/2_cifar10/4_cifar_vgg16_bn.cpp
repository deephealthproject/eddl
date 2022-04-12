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
  bool use_mpi = false;
  
  int batch_size = 800;
  int epochs = 10;
  
  // Process arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
        else if (strcmp(argv[i], "--mpi") == 0) use_mpi= true;
        else if (strcmp(argv[i], "--batch-size") == 0) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0) {
            epochs = atoi(argv[++i]);
        }
    }
    if (use_mpi)
      init_distributed("MPI");
    else 
      init_distributed("NCCL");  
  
    vector<string> hws = Tensor::hardware_supported();
    for (auto i: hws)
    std::cout << i << ' ';
      
      printf ("DeviceID %d) \n", Tensor::getDeviceID("cuda"));
      
  // download CIFAR data
  download_cifar10();

  // Settings
  epochs = (testing) ? 2 : epochs;   
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
  l=ReLu(Dense(l,512));

  layer out= Softmax(Dense(l, num_classes));

  // net define input and output layers list
  model net=Model({in},{out});

  compserv cs = nullptr;
  if (use_cpu) {
      cs = CS_CPU();
  } else {
      cs = CS_GPU({1}); // one GPU
      // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
//       cs = CS_CPU();
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

  for(int i=0;i<epochs;i++) {
    printf("== Epoch: %d\n",i);
    // training, list of input and output tensors, batch, epochs
    fit(net,{x_train},{y_train},batch_size, 1);
    // Evaluate test
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net,{x_test},{y_test});
   
    if (early_stopping_on_loss_var (net, 0, 0.2, 1, i)) break;
    if (early_stopping_on_metric (net, 0, 0.9, 1, i)) break;
  }
  
   save(net, "weights.model");
   // Quantization
   //CPU_quantize_network_distributed(net, 1, 6);
   //std::cout << "Evaluate test w/quantization:" << std::endl;
   //evaluate(net,{x_test},{y_test});

    for (int bits = 10; bits > 2; bits--) {
        load(net, "weights.model");

        // Quantization
        GPU_quantize_network_distributed(net, 1, bits);
        std::cout << "Evaluate test w/quantization, GPU:" << std::endl;
        evaluate(net,{x_test},
        {
            y_test
        });
    }
    
  
    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;

    return EXIT_SUCCESS;
}


///////////
