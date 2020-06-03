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
// Text generation
// Only Decoder
//////////////////////////////////

int main(int argc, char **argv) {

    // Settings
    int epochs = 10;
    int batch_size = 32;

    int olength=5;
    int invs=100;
    int outvs=100;
    int embedding=64;

    // Define network
    layer in = Input({1}); //1 word
    layer l = in;

    layer lE = RandomUniform(Embedding(l, invs, 1,embedding),-0.05,0.05);

    l = ReLu(Dense(lE,1024));

    // Decoder
    layer ind = Input({outvs});
    l = Decoder(LSTM(ind,256),l,"concat");

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
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
    );

    // View model
    summary(net);


    // Load dataset

    Tensor *x_train=new Tensor({1000,1});
    Tensor *y_train=new Tensor({1000,olength*outvs});
    y_train->reshape_({x_train->shape[0],olength,outvs}); //batch x timesteps x input_dim

    // Train model
    for(int i=0;i<epochs;i++) {
      fit(net, {x_train}, {y_train}, batch_size, 1);
      //evaluate(net,{x_test},{y_test});
    }

}
