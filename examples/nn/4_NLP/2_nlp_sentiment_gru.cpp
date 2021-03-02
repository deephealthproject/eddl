/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// Embeding+GRU
// using imdb preprocessed from keras
// 2000 words
//////////////////////////////////

int main(int argc, char **argv) {

    bool testing = false;
    bool use_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download Imdb
    download_imdb_2000();


    // Settings
    int epochs = testing ? 2 : 10;
    int batch_size = 32;

    int length = 250;
    int embdim = 33;
    int vocsize = 2000;

    // Define network
    layer in = Input({1}); //1 word
    layer l = in;

    layer lE = RandomUniform(Embedding(l, vocsize, 1, embdim), -0.05, 0.05);

    l = GRU(lE, 37);
    l = ReLu(Dense(l, 256));

    layer out = Sigmoid(Dense(l, 1));
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    //optimizer opt = sgd(0.001);
    //optimizer opt = rmsprop(0.001);
    optimizer opt = adam(0.001);
    //opt->set_clip_val(0.01);

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        //cs = CS_GPU({1}, "low_mem"); // one GPU
        cs = CS_GPU({1}, "full_mem"); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
    }

    // Build model
    build(net,
          opt, // Optimizer
          {"binary_cross_entropy"}, // Losses
          {"binary_accuracy"}, // Metrics
          cs
    );

    // View model
    summary(net);

    // Load dataset
    Tensor* x_train=Tensor::load("imdb_2000_trX.bin");
    Tensor* y_train=Tensor::load("imdb_2000_trY.bin");
    Tensor* x_test=Tensor::load("imdb_2000_tsX.bin");
    Tensor* y_test=Tensor::load("imdb_2000_tsY.bin");


    x_train->reshape_({x_train->shape[0], length, 1}); //batch x timesteps x input_dim
    x_test->reshape_({x_test->shape[0], length, 1}); //batch x timesteps x input_dim

    y_train->reshape_({y_train->shape[0], 1, 1}); //batch x timesteps x input_dim
    y_test->reshape_({y_test->shape[0], 1, 1}); //batch x timesteps x input_dim

    Tensor* x_mini_train = x_train;
    Tensor* y_mini_train = y_train;
    Tensor* x_mini_test  = x_test;
    Tensor* y_mini_test  = y_test;
    if (testing) {
        x_mini_train = x_train->select({"0:64", ":", ":"});
        y_mini_train = y_train->select({"0:64", ":", ":"});
        x_mini_test  = x_test->select({"0:64", ":", ":"});
        y_mini_test  = y_test->select({"0:64", ":", ":"});
        epochs = 2;
    }


    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {x_mini_train}, {y_mini_train}, batch_size, 1);
      evaluate(net, {x_mini_test}, {y_mini_test});
    }

    if (x_mini_train != x_train) delete x_mini_train;
    if (y_mini_train != y_train) delete y_mini_train;
    if (x_mini_test != x_test) delete x_mini_test;
    if (y_mini_test != y_test) delete y_mini_test;
    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;
    //delete cs; -- Net object is in charge of free the memory
    //delete opt; -- Net object is in charge of free the memory

    return EXIT_SUCCESS;
}
