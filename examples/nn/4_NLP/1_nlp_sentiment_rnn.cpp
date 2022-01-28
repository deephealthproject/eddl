/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// Embeding+RNN
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
    int epochs = testing ? 1 : 10;
    int batch_size = 128;

    int length=250;
    int embdim=33;
    int vocsize= 2000;

    // Define network
    layer in = Input({1}); //1 word
    layer l = in;

    layer lE = RandomUniform(Embedding(l, vocsize, 1,embdim),-0.05,0.05);

    l = RNN(lE,37);
    l = ReLu(Dense(l,256));


    layer out = Sigmoid(Dense(l, 1));
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    optimizer opt=adam(0.001);
    //opt->set_clip_val(0.01);

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU({1}); // one GPU
        // cs = CS_GPU({1}, "low_mem"); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
    }

    // Build model
    build(net,
          opt, // Optimizer
          {"binary_cross_entropy"}, // Losses
          {"binary_accuracy"}, // Metrics
          cs);

    // View model
    summary(net);

    // Load dataset
    Tensor* x_train=Tensor::load("imdb_2000_trX.bin");
    Tensor* y_train=Tensor::load("imdb_2000_trY.bin");

    Tensor* x_test=Tensor::load("imdb_2000_tsX.bin");
    Tensor* y_test=Tensor::load("imdb_2000_tsY.bin");


    x_train->reshape_({x_train->shape[0],length,1}); //batch x timesteps x input_dim
    x_test->reshape_({x_test->shape[0],length,1}); //batch x timesteps x input_dim

    y_train->reshape_({y_train->shape[0],1,1}); //batch x timesteps x input_dim
    y_test->reshape_({y_test->shape[0],1,1}); //batch x timesteps x input_dim

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * batch_size);
        Tensor* x_mini_train = x_train->select({_range_, ":", ":"});
        Tensor* y_mini_train = y_train->select({_range_, ":", ":"});
        Tensor* x_mini_test  = x_test->select({_range_, ":", ":"});
        Tensor* y_mini_test  = y_test->select({_range_, ":", ":"});

        delete x_train;
        delete y_train;
        delete x_test;
        delete y_test;

        x_train = x_mini_train;
        y_train = y_mini_train;
        x_test  = x_mini_test;
        y_test  = y_mini_test;
    }

    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 1);
      evaluate(net,{x_test},{y_test});
    }

    delete net;

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;

    return EXIT_SUCCESS;
}
