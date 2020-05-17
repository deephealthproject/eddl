/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
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
// cifar_conv_da.cpp:
// A very basic Conv for cifar10
// Data augmentation
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv){

    // download CIFAR data
    download_cifar10();

    // Settings
    int epochs = 5;
    int batch_size = 100;
    int num_classes = 10;

    // network
    layer in=Input({3,32,32});
    layer l=in;

    // Data augmentation
//   l = RandomShift(l, {-0.2f, +0.2f}, {-0.2f, +0.2f});
//   l = RandomRotation(l, {-30.0f, +30.0f});
//   l = RandomScale(l, {0.85f, 2.0f});
//   l = RandomFlip(l, 1);
//   l = RandomCrop(l, {28, 28});
//   l = RandomCropScale(l, {0.f, 1.0f});
//   l = RandomCutout(l, {0.0f, 0.3f}, {0.0f, 0.3f});

    // l=Select(l, {"1", "1:31", "1:31"});
    l=MaxPool(ReLu(Conv(l,32,{3,3},{1,1})),{2,2});
    l=MaxPool(ReLu(Conv(l,64,{3,3},{1,1})),{2,2});
    l=MaxPool(ReLu(Conv(l,128,{3,3},{1,1})),{2,2});
    l=MaxPool(ReLu(Conv(l,256,{3,3},{1,1})),{2,2});

    l=Reshape(l,{-1});

    l=Activation(Dense(l,128),"relu");

    layer out=Activation(Dense(l,num_classes),"softmax");

    // net define input and output layers list
    model net=Model({in},{out});


    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
    );


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

    for(int i=0;i<epochs;i++) {
        // training, list of input and output tensors, batch, epochs
        fit(net,{x_train},{y_train},batch_size, 1);
        // Evaluate train
        std::cout << "Evaluate test:" << std::endl;
        evaluate(net,{x_test},{y_test});
    }


}
