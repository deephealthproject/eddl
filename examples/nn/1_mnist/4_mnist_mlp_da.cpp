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


using namespace eddl;

//////////////////////////////////
// mnist_mlp_da.cpp:
// A very basic MLP for mnist
// Playing with Data Augmentation
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 5;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    // Data augmentation assumes 3D tensors... images:
    l=Reshape(l,{1,28,28});
    // Data augmentation
    l = RandomCropScale(l, {0.9f, 1.0f});

    // Come back to 1D tensor for fully connected:
    l=Reshape(l,{-1});
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    //l = ReLu(Dense(l, 1024));
    //l = ReLu(Dense(l, 1024));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
	      //CS_FPGA({1})
    );

    // View model
    summary(net);

    setlogfile(net,"mnist_bn_da_lra");

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);


    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);
    // Evaluate
    printf("Evaluate:\n");
    evaluate(net, {x_test}, {y_test});


    // LR annealing
    setlr(net,{0.005,0.9});
    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs/2);
    // Evaluate
    printf("Evaluate:\n");
    evaluate(net, {x_test}, {y_test});


    // LR annealing
    setlr(net,{0.001,0.9});
    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs/2);
    // Evaluate
    printf("Evaluate:\n");
    evaluate(net, {x_test}, {y_test});

    // LR annealing
    setlr(net,{0.0001,0.9});
    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs/4);
    // Evaluate
    printf("Evaluate:\n");
    evaluate(net, {x_test}, {y_test});



}
