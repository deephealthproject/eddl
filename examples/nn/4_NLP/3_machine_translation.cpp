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
// MT
// using EuTrans
//////////////////////////////////

int main(int argc, char **argv) {
    // Download Imdb
    //download_eutrans();

    // Settings
    int epochs = 10;
    int batch_size = 32;

    int ilength=3;
    int olength=5;
    int invs=100;
    int outvs=100;
    int embedding=64;

    // Define network
    layer in = Input({1}); //1 word
    layer l = in;

    //layer lE = RandomUniform(Embedding(l, invs, 1,embedding),-0.05,0.05);

    l=Dense(l,128);
    //l = LSTM(LSTM(lE,256),256);

    // Decoder
    l = LSTM(Decoder(LSTM(l,256),outvs),256);

    layer out = Softmax(Dense(l, outvs));

    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    plot(net, "model.pdf");



    optimizer opt=adam(0.001);
    //opt->set_clip_val(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          //CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          CS_CPU()
    );

    // View model
    summary(net);


    // Load dataset
    /*
    Tensor* x_train=Tensor::load("eutrans_trX.bin");
    Tensor* y_train=Tensor::load("eutrans_trY.bin");
    Tensor* x_test=Tensor::load("eutrans_tsX.bin");
    Tensor* y_test=Tensor::load("eutrans_tsY.bin");
*/
    Tensor *x_train=new Tensor({1000,1});
    Tensor *y_train=new Tensor({1000,olength*outvs});
    x_train->reshape_({x_train->shape[0],1}); //batch x timesteps x input_dim
    y_train->reshape_({x_train->shape[0],olength,outvs}); //batch x timesteps x input_dim

    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 1);
      //evaluate(net,{x_test},{y_test});
    }

}
