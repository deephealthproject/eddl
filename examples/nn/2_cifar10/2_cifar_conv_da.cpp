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
// cifar_conv_da.cpp:
// A very basic Conv for cifar10
// Data augmentation
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv){

    // download CIFAR data
    download_cifar10();

    // Settings
    int epochs = 100;
    int batch_size = 128;
    int num_classes = 10;

    // network
    layer in=Input({3,32,32});
    layer l=in;

    // Data augmentation
    l = RandomHorizontalFlip(l);
    l = RandomCropScale(l, {0.8f, 1.0f});
    l = RandomCutout(l,{0.1,0.5},{0.1,0.5});
    ////

    l=MaxPool(ReLu(BatchNormalization(HeUniform(Conv(l,32,{3,3},{1,1},"same",false)))),{2,2});

    l=MaxPool(ReLu(BatchNormalization(HeUniform(Conv(l,64,{3,3},{1,1},"same",false)))),{2,2});

    l=MaxPool(ReLu(BatchNormalization(HeUniform(Conv(l,128,{3,3},{1,1},"same",false)))),{2,2});

    l=MaxPool(ReLu(BatchNormalization(HeUniform(Conv(l,256,{3,3},{1,1},"same",false)))),{2,2});

    l=Reshape(l,{-1});

    l=Activation(BatchNormalization(Dense(l,128)),"relu");

    layer out=FullSoftmax(BatchNormalization(Dense(l,num_classes)));

    // net define input and output layers list
    model net=Model({in},{out});


    // Build model
    build(net,
          adam(0.001), // Optimizer
          {"categorical_cross_entropy"}, // Losses
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

    setlr(net,{0.0001});

    for(int i=0;i<epochs;i++) {
        // training, list of input and output tensors, batch, epochs
        fit(net,{x_train},{y_train},batch_size, 1);
        // Evaluate train
        std::cout << "Evaluate test:" << std::endl;
        evaluate(net,{x_test},{y_test});
    }
}
