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
#include "eddl/serialization/onnx/eddl_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/serialization/onnx/export_helpers.h"

using namespace eddl;

//////////////////////////////////
// cifar_resnet.cpp:
// Resnet18 without BatchNorm
// Using fit for training
//////////////////////////////////

layer ResBlock(layer l, int filters,int nconv,int half) {
  layer in=l;

  if (half)
      l=ReLu(Conv(l,filters,{3,3},{2,2}));
  else
      l=ReLu(Conv(l,filters,{3,3},{1,1}));


  for(int i=0;i<nconv-1;i++)
    l=ReLu(Conv(l,filters,{3,3},{1,1}));

  if (half)
    return Add(Conv(in,filters,{1,1},{2,2}),l);
  else
    return Add(l,in);
}

int main(int argc, char **argv){

  int epochs = -1;
  //args
  int use_cpu=0;
  int use_quant=0;
  int testing=0;
  int testingquant=0;
  int savecheckpoint=0;
  int loadcheckpoint=0;
  float alpha = 0;
  int roundbits=0;
  int clipbits=0;
  int just_evaluate=0;
  string checkpoint = "";
  for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--cpu")) {
          use_cpu=1;
      }
      else if (!strcmp(argv[i], "--test")) {
          epochs=1;
          testing=1;
          use_quant=0;
      }
      else if (!strcmp(argv[i], "--epoch")) {
          epochs = atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "--testquantization")) {
          epochs=0;
          testing=1;
          use_quant=1;
          testingquant=1;
      }
      else if (!strcmp(argv[i], "--quantization")) {
          use_quant=1;
      }
      else if (!strcmp(argv[i], "--savecheckpoint")) {
          savecheckpoint = 1;
          loadcheckpoint = 0;
          use_quant=0;
          testing=0;
          testingquant=0;
      }
      else if (!strcmp(argv[i], "--loadcheckpoint")) {
          savecheckpoint = 0;
          loadcheckpoint = 1;
          checkpoint = argv[++i];
      }
      else if (!strcmp(argv[i], "--alpha")) {
          alpha = atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "--bits")) {
          clipbits = atoi(argv[++i]); 
          roundbits = atoi(argv[++i]);
      }
      else if(!strcmp(argv[i], "--evaluate")){
          just_evaluate = 1;
          use_quant=0;
          testing=0;
          testingquant=0;
          savecheckpoint=0;
          loadcheckpoint=1;
          alpha = 0;
      }
  }

  if(testing)printf("Testing ON\n");
  if(testingquant)printf("Testing ON\n");
  if(use_quant) {
    printf("Quantization ON with %d %d bits %f alpha\n",clipbits, roundbits, alpha);
  }
  if(savecheckpoint)printf("Save model ON\n");
  if(loadcheckpoint)printf("Load model from checkpoint ON\n");
  // download CIFAR data
  download_cifar10();

  // Settings
  if (epochs==-1) epochs = testing ? 2 : 15;
  int batch_size = 100;
  int num_classes = 10;

  // network
  layer in=Input({3,32,32});
  layer l=in;

  l=ReLu(Conv(l,64,{3,3},{1,1}));

  // Add explicit padding to avoid the asymmetric padding in the Conv layers
  l = Pad(l, {0, 1, 1, 0});

  l=ResBlock(l, 64,2,1);//<<<-- output half size
  l=ResBlock(l, 64,2,0);

  l=ResBlock(l, 128,2,1);//<<<-- output half size
  l=ResBlock(l, 128,2,0);

  l=ResBlock(l, 256,2,1);//<<<-- output half size
  l=ResBlock(l, 256,2,0);

  l=ResBlock(l, 512,2,1);//<<<-- output half size
  l=ResBlock(l, 512,2,0);

  l=Reshape(l,{-1});
  l=Activation(Dense(l,512),"relu");

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
      //sgd(0.001, 0.9), // Optimizer
      {"softmax_cross_entropy"}, // Losses
      {"categorical_accuracy"}, // Metrics
      cs);

  if (loadcheckpoint) {
      cout << " Loading new from checkpoint ... " << checkpoint <<std::endl;

      if(!checkpoint.empty()) load(net, checkpoint, "bin");
      else {
          printf("[ERROR] Empty checkpoint\n");
          exit(0);
      }
  }

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

//set_quantized_mode(net, 0, (2^bits), alpha);

int i;
if(!testingquant && !loadcheckpoint && !just_evaluate){
  std::cout << "Floating point training with "<< epochs << std::endl;
  for(i=0;i<epochs;i++) {
    // training, list of input and output tensors, batch, epochs
    fit(net,{x_train},{y_train},batch_size, 1);
    // Evaluate train
    std::cout << "Evaluate test:" << std::endl;
    evaluate(net,{x_test},{y_test});
  }
}

if(savecheckpoint) {
    cout << "Saving weights..." << endl;
    save(net, "cifar_resnet_checkpoint_epoch_" + to_string(i) + ".bin", "bin");
}

if(use_quant || testingquant) {
    int quantepochs = 10;
    if(testingquant) quantepochs = 1;

    std::cout << "Quantization training with "<< quantepochs << " epochs: " << clipbits << "_" << roundbits <<  " bits and " << alpha << " alpha" <<std::endl;
    set_quantized_mode(net, 1, clipbits, roundbits, alpha);
    //set_quantized_mode(net, 1, pow(2,bits), alpha);
    for(int i=0;i<quantepochs;i++) {
      // training, list of input and output tensors, batch, epochs
      fit(net,{x_train},{y_train},batch_size, 1);
      // Evaluate train
      std::cout << "QEvaluate test:" << std::endl;
      evaluate(net,{x_test},{y_test});
    }

    end_quantization(net);
    save_net_to_onnx_file(net,"cifar_resnet_" + to_string(clipbits) +"_" + to_string(roundbits) + "bits_"+to_string(alpha)+"alpha"+".onnx");
}

//just evalutate
if(just_evaluate) {
    printf("Evaluate net\n");
    evaluate(net,{x_test},{y_test});
}
printf("end\n");

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;

    return EXIT_SUCCESS;
}


///////////
