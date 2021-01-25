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
// cifar_pretrained.cpp:
// from pretrained onnx resnet18
//////////////////////////////////

int main(int argc, char **argv){

  // download CIFAR data
  download_cifar10();

  string path("resnet18-v1-7.onnx");

  // Download from:
  //https://www.dropbox.com/s/tn0d87dr035yhol/resnet18-v1-7.onnx

	Net* net_onnx = import_net_from_onnx_file(path, DEV_CPU);

  // Remove last layer
  removeLayer(net_onnx, "resnetv15_dense0_fwd");

  // create a new graph to adapt the output for CIFAR
  layer in=Input({512});
  layer l=Dense(in,10);
  l=Softmax(l);

  model net_adap=Model({in},{l});

  // cat both models
  model net=Model({net_onnx,net_adap});

  build(net,
        sgd(0.01), // Optimizer
        {"soft_cross_entropy"}, // Losses
        {"accuracy"}, // Metrics
        //CS_GPU({1}) // one GPU
        //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
        CS_CPU()
  );

  summary(net);
  plot(net,"model.pdf");

  setTrainable(net,"flatten_170",false); //from "flatten_170" to the begining

  // Load dataset
  Tensor *x_train=Tensor::load("cifar_trX.bin","bin");
  Tensor* xs = x_train->select({":1000", ":"});
  Tensor* xtrain = Tensor::zeros({1000, 3, 224, 224});

  Tensor::scale(xs, xtrain, {224, 224}); //Scale the image to 100x100 px
  xtrain->div(255.0);
  delete x_train;
  delete xs;

  Tensor *y_train=Tensor::load("cifar_trY.bin","bin");
  Tensor* ytrain = y_train->select({":1000", ":"});
  //
  Tensor* x_test = Tensor::load("cifar_tsX.bin");
  Tensor* y_test = Tensor::load("cifar_tsY.bin");
  x_test->div_(255.0f);

  int epochs = 100;
  int batch_size = 16;

  for(int i=0;i<epochs;i++)
    fit(net,{xtrain},{ytrain},batch_size, 1);

  evaluate(net, {x_test}, {y_test}, batch_size);
  // Export
  string import_path("trained_model.onnx");
  save_net_to_onnx_file(net, path);

  cout << "Saved net to onnx file" << endl;

  // Import 
  Net* imported_net = import_net_from_onnx_file(import_path, DEV_CPU);
	
  build(imported_net,
        adam(0.001), // Optimizer
        {"soft_cross_entropy"}, // Losses
        {"categorical_accuracy"}, // Metrics
	    CS_GPU({1}, "low_mem"), // one GPU
        //CS_CPU(), // CPU with maximum threads availables
	    false       // Parameter that indicates that the weights of the net must be initialized to random values.
  );

  cout << "Evaluating with imported net" << endl;
  evaluate(imported_net, {x_test}, {y_test}, batch_size);




}
