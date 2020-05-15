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
// Embeding+CNN for
// aclImdb sentiment analysis
// using all vocabulary
//////////////////////////////////

int main(int argc, char **argv) {
    // Download aclImdb
    download_imdb();

    // Settings
    int epochs = 100;
    int batch_size = 100;
    int num_classes = 2;

    int length=100;
    int embdim=250;
    int vocsize=72682;

    // Define network
    layer in = Input({length});
    layer l = in;

    layer lE=Embedding(l, vocsize, length, embdim, true); // mask_zeros

    l = Reshape(lE,{1,length,embdim});

    layer l1 = ReLu(BatchNormalization(Conv(l,128,{1,embdim},{1,1},"same,none",false)));
    layer l2 = ReLu(BatchNormalization(Conv(l,128,{2,embdim},{1,1},"same,none",false)));
    layer l3 = ReLu(BatchNormalization(Conv(l,128,{3,embdim},{1,1},"same,none",false)));

    layer lc1=l=Concat({l1,l2,l3});

    l=GlobalMaxPool(l);

    l = Reshape(l,{-1});

    l = ReLu(BatchNormalization(Dense(l,256)));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    // Build model
    build(net,
          adam(0.001), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU, sync:100 batches
          //CS_CPU(-1) // CPU with maximum threads availables
    );
    // View model
    summary(net);

    // Load dataset
    Tensor* x_train=Tensor::load("imdb_trX.bin");
    Tensor* y_train=Tensor::load("imdb_trY.bin");
    Tensor* x_test=Tensor::load("imdb_tsX.bin");
    Tensor* y_test=Tensor::load("imdb_tsY.bin");

    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 1);
      evaluate(net,{x_test},{y_test});
    }



}
