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
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace eddl;

///////////////////////////////////////////////////
// 5_onnx_import_net_and_reshape.cpp:
// An example of loading an ONNX model (resnet18)
// and reshape its input to train on cifar10.
// It also changes the output layer of the model
// to classify over 10 classes.
///////////////////////////////////////////////////


int main(int argc, char **argv) { 
    bool testing = false;
    bool use_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download cifar
    download_cifar10();

    // Settings
    int freeze_epochs = 2;
    int unfreeze_epochs = 5;
    int batch_size = 100;
    int num_classes = 10; // 10 labels in cifar10
    vector<string> names;
	string path("resnet18.onnx");

    // Import resnet18 model and reshape input for cifar
	Net* net = import_net_from_onnx_file(path, {3, 32, 32}, DEV_CPU);

    for(auto l : net->layers)
       names.push_back(l->name);

    removeLayer(net, "resnetv15_dense0_fwd"); // Remove dense layer of output

    layer l = getLayer(net, "flatten_170"); // Get last layer to connect the new dense
    layer out = Softmax(Dense(l, num_classes, true, "new_dense")); // true is for the bias.

    layer in = getLayer(net, "data"); // Get input layer of the model
    net = Model({in}, {out}); // Create a new model from input output

    compserv cs = nullptr;
    if (use_cpu) cs = CS_CPU();
    else cs = CS_GPU({1});

    // Build model
    build(net,
          adam(0.0001),              // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"},  // Metrics
          cs,                        // Computing service (CPU or GPU)
		  false                      // Parameter that indicates that the weights of the net must not be initialized to random values.
    );

    // View model
    summary(net);

    // Force initialization of new layers
    initializeLayer(net, "new_dense");
    
    // Load training data
    Tensor* x_train = Tensor::load("cifar_trX.bin");
    Tensor* y_train = Tensor::load("cifar_trY.bin");
    // Load test data
    Tensor* x_test = Tensor::load("cifar_tsX.bin");
    Tensor* y_test = Tensor::load("cifar_tsY.bin");
  
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

    // Preprocess data
    x_train->div_(255.0f);
    x_test->div_(255.0f);

    // Freeze pretrained weights
    for(auto n : names)
      setTrainable(net, n, false);

    // Train few epochs the new layers
    fit(net, {x_train}, {y_train}, batch_size, freeze_epochs);

    // Unfreeze weights
    for(auto n:names)
      setTrainable(net,n,true);

    // Train few epochs all layers
    fit(net, {x_train}, {y_train}, batch_size, unfreeze_epochs);

    // Evaluate
    evaluate(net, {x_test}, {y_test}, batch_size);

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;

	return EXIT_SUCCESS;
}
