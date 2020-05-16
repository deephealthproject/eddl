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
// Embeding+RNN for
// aclImdb sentiment analysis
//////////////////////////////////

int main(int argc, char **argv) {
    // Download aclImdb
    download_imdb_1000();

    // Settings
    int epochs = 1000;
    int batch_size = 100;
    int num_classes = 2;

    int length=100;
    int embdim=250;
    int vocsize=1001;  //1000 most frequent words + padding

    // Define network
    layer in = Input({1}); //1 word
    layer l = in;

    layer lE = Embedding(l, vocsize, 1,embdim,true); //mask_zeros=true

    l = L2(LSTM(lE,256),0.001);
    l = LeakyReLu(BatchNormalization(Dense(l,128)));


    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    optimizer opt=rmsprop(0.00001);
    //opt->set_clip_val(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
    );

    // View model
    summary(net);

    // Load dataset
    Tensor* x_train=Tensor::load("imdb_1000_trX.bin");
    Tensor* y_train=Tensor::load("imdb_1000_trY.bin");
    Tensor* x_test=Tensor::load("imdb_1000_tsX.bin");
    Tensor* y_test=Tensor::load("imdb_1000_tsY.bin");


    x_train->reshape_({x_train->shape[0],length,1}); //batch x timesteps x input_dim
    x_test->reshape_({x_test->shape[0],length,1}); //batch x timesteps x input_dim

    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 5);
      evaluate(net,{x_test},{y_test});
    }


}
