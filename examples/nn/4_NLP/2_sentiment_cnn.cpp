/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
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
//////////////////////////////////

int main(int argc, char **argv) {
    // Download aclImdb
    download_imdb();

    // Settings
    int epochs = 100;
    int batch_size = 100;
    int num_classes = 2;

    int length=10;
    int embdim=250;
    int vocsize=72682;

    // Define network
    layer in = Input({length});
    layer l = in;

    layer lE=Embedding(l, vocsize, length, embdim, true); // mask_zeros

    l=Dropout(lE,0.1); //5% of training words treated as unknown

    l = Reshape(l,{1,length,embdim});
    layer l1 = ReLu(BatchNormalization(Conv(l,128,{1,embdim},{1,1},"same,none",false)));
    layer l2 = ReLu(BatchNormalization(Conv(l,128,{2,embdim},{1,1},"same,none",false)));
    layer l3 = ReLu(BatchNormalization(Conv(l,128,{3,embdim},{1,1},"same,none",false)));

    layer lc1=l=Concat({l1,l2,l3});


    l1 = ReLu(BatchNormalization(Conv(l,128,{1,1},{1,1},"same,none",false)));
    l2 = ReLu(BatchNormalization(Conv(l,128,{2,1},{1,1},"same,none",false)));
    l3 = ReLu(BatchNormalization(Conv(l,128,{3,1},{1,1},"same,none",false)));

    layer lc2=l=Concat({l1,l2,l3});


    l1 = ReLu(BatchNormalization(Conv(l,128,{1,1},{1,1},"same,none",false)));
    l2 = ReLu(BatchNormalization(Conv(l,128,{2,1},{1,1},"same,none",false)));
    l3 = ReLu(BatchNormalization(Conv(l,128,{3,1},{1,1},"same,none",false)));

    layer lc3=l=Concat({l1,l2,l3});

    l=Add({lc1,lc2,lc3});


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
          adam(0.0001), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU
          //CS_CPU(-1) // CPU with maximum threads availables
    );
    //toGPU(net,{1},100,"low_mem"); // In two gpus, syncronize every 100 batches, low_mem setup

    // View model
    summary(net);

    // Load dataset
    Tensor* x_train=Tensor::load("imdb_trX.bin");
    //x_train->info();
    Tensor* y_train=Tensor::load("imdb_trY.bin");
    //y_train->info();
    // Load dataset
    Tensor* x_test=Tensor::load("imdb_tsX.bin");
    //x_train->info();
    Tensor* y_test=Tensor::load("imdb_tsY.bin");
    //y_train->info();

    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 1);
      evaluate(net,{x_test},{y_test});
    }


    Tensor* E=getParam(lE,0);

    E->save("embedding.bin");


}
