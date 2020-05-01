/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"
#include "eddl/apis/eddlT.h"

using namespace eddl;

//////////////////////////////////
// Embeding+Dense for
// aclImdb sentiment anañysis
//////////////////////////////////

int main(int argc, char **argv) {
    // Download aclImdb


    // Settings
    int epochs = 1000;
    int batch_size = 100;
    int num_classes = 2;

    int length=80;
    int embdim=50;
    int vocsize=75180;

    // Define network
    layer in = Input({length});
    layer l = in;

    l = GlorotUniform(L2(Embedding(l, vocsize, length, embdim),0.001));
    l = Reshape(l,{1,length,embdim});
    layer l1 = ReLu(BatchNormalization(Conv(l,128,{1,embdim},{1,1},"same,none")));
    layer l3 = ReLu(BatchNormalization(Conv(l,128,{2,embdim},{1,1},"same,none")));
    layer l5 = ReLu(BatchNormalization(Conv(l,128,{3,embdim},{1,1},"same,none")));
    l=GlobalMaxPool(Concat({l1,l3,l5}));

    l = Reshape(l,{-1});

    l = ReLu(BatchNormalization(Dense(l,256)));
    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          adam(0.0001), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          //CS_GPU({1}) // one GPU
          CS_GPU({1,1},100) // two GPU
          //CS_CPU(-1) // CPU with maximum threads availables
    );
    //toGPU(net,{1},100,"low_mem"); // In two gpus, syncronize every 100 batches, low_mem setup

    // View model
    summary(net);

    // Load dataset
    tensor x_train=eddlT::load("xtrain.bin");
    //x_train->info();
    tensor y_train=eddlT::load("ytrain.bin");
    //y_train->info();

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);


}
