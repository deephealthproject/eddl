/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
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
// mnist_auto_encoder.cpp:
// An autoencoder for mnist
// merging two networs
//////////////////////////////////

int main(int argc, char **argv) {

    // Download dataset
    download_mnist();

    // Settings
    int epochs = 5;
    int batch_size = 100;



    // base model
    // is a merge of two nets

    //encoder
    layer in = Input({784});
    layer l = Activation(Dense(in, 256), "relu");
    l = Activation(Dense(l, 128), "relu");

    model enc=Model({in},{l});
    setName(enc,"enc");

    // decoder
    in = Input({128});
    layer out = Activation(Dense(in, 64), "relu");

    model dec=Model({in},{out});
    setName(dec,"dec");

    // merge
    model base = Model({enc,dec});
    setName(base,"base");

    plot(base, "base.pdf");

    //////

    layer in1 = Input({784});
    layer in2 = Input({784});

    // get siameses
    layer out1 = getLayer(base,{in1});
    layer out2 = getLayer(base,{in2});

    // combine siameses with some operations
    l=Diff(out1,out2);
    l=ReLu(Dense(l,256));
    layer outs=Sigmoid(Dense(l,784));

    // build final model
    model siamese=Model({in1,in2},{outs});
    setName(siamese,"siamese");

    // Build model
    build(siamese,
          adam(0.0001), // Optimizer
          {"dice"}, // Losses
          {"dice"}, // Metrics
          CS_GPU({1}) // one GPU
          //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
          //CS_CPU()
    );
    summary(siamese);
    plot(siamese, "model.pdf");

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    // Preprocessing
    x_train->div_(255.0f);

    // Train model
    fit(siamese, {x_train,x_train}, {x_train}, batch_size, epochs);

    delete base;
    delete enc;
    delete dec;
    delete siamese;


}
