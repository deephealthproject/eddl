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
// mnist_mlp_da.cpp:
// A very basic MLP for mnist
// Playing with Data Augmentation
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 100;
    int batch_size = 100;
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    // Data augmentation assumes 3D tensors... images:
    l=Reshape(l,{1,28,28});
    // Data augmentation
    l = RandomCropScale(l, {0.9f, 1.0f});
    l = RandomShift(l, {-0.1, 0.1}, {-0.1, 0.1});
    l = RandomRotation(l, {-10, 10});

    // Come back to 1D tensor for fully connected:
    l=Reshape(l,{-1});
    l = ReLu(GaussianNoise(LayerNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(LayerNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(LayerNormalization(Dense(l, 1024)),0.3));
    //l = ReLu(Dense(l, 1024));
    //l = ReLu(Dense(l, 1024));
 cout<<"OK1\n";
    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
 cout<<"OK1\n";
    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1}) // one GPU
          CS_CPU(-1, "low_mem") // CPU with maximum threads availabçles
    );

    // View model
    summary(net);

    setlogfile(net,"mnist_bn_da_lra");

    // Load dataset
    tensor x_train = eddlT::load("trX.bin");
    tensor y_train = eddlT::load("trY.bin");
    tensor x_test = eddlT::load("tsX.bin");
    tensor y_test = eddlT::load("tsY.bin");

    // Preprocessing
    eddlT::div_(x_train, 255.0);
    eddlT::div_(x_test, 255.0);


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
